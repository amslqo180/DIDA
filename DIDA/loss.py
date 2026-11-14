import torch
import torch.nn.functional as F


def orthogonal_loss(x, y):
    """
    x,y: [B,D]
    希望内积接近0
    """
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)

    return (x * y).sum(dim=-1).pow(2).mean()

def batch_orthogonal_loss(x,y):
    """
    输入：
        x: [B, dim] 或 [B, N, dim]
        y: [B, dim] 或 [B, M, dim]
    输出：
        一个标量 Tensor，代表整个 Batch 的平均正交损失
    """
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)

    ### 转换3D张量，使用矩阵乘法
    if x.dim()==2:x=x.unsqueeze(1)  ###[B,1,dim]
    if y.dim()==2:y=y.unsqueeze(1)

    ### 使用批量矩阵乘法bmm计算内积[B,1,dim] x [B,dim,1] -> [B,1,1]
    inner_product = torch.bmm(x,y.transpose(-2,-1))

    ### 平方求均值（余弦相似度越接近0，平方越接近0）
    return inner_product.pow(2).mean()

def token_regularization_loss(
        style_embeds,
        main_embeds,
        sub_embeds,
        num_vectors=3,
        lambda_orth=1.0,
        lambda_orth_sub=1.0,
        lambda_comp=1.0,
):
    """
    输入参数：
        style_embeds:[B,dim]当前批次的所有风格嵌入
        cam_embeds:[B,dim]当前批次的所有相机视角嵌入
        main_embeds:[B,dim]当前批次的所有行人身份主嵌入
        sub_embeds:[B,num_vectors,dim]当前批次的所有行人子部位嵌入
    """
    ### 风格与身份 and 风格与相机 and 相机与身份 正交 (输入均为[B,dim]，输出为标量)
    loss_style_main=batch_orthogonal_loss(style_embeds,main_embeds)

    ### sub-tokens两两正交（输入sub_embeds:[B,num_vectors,dim]）
    ### 对自身计算批量内积：[B, num_vectors, dim] x [B, dim, num_vectors] -> [B, num_vectors, num_vectors]
    subs_norm = F.normalize(sub_embeds, dim=-1)

    ### 混淆矩阵，对角线全是1（自己和自己的相似度），非对角线是两两相似度
    inner_matrix = torch.bmm(subs_norm, subs_norm.transpose(-2,-1))

    # 我们只需要严格的上三角部分（排除对角线的1，避免自己跟自己算正交）
    # torch.triu_indices 可以直接帮我们提取出上三角的索引
    triu_indices = torch.triu_indices(row=num_vectors, col=num_vectors, offset=1)

    total_orth_sub_loss = inner_matrix[:,triu_indices[0],triu_indices[1]].pow(2).mean()

    ### 主token = 子token 聚合（Composition Loss）
    # 沿着子部位维度求平均: [B, num_vectors, dim] -> [B, dim]
    sub_mean = sub_embeds.mean(dim=1)

    total_comp_loss=F.mse_loss(
        F.normalize(main_embeds, dim=-1),
        F.normalize(sub_mean, dim=-1)
    )

    ### 最终加权融合 总loss
    total_loss=(
        lambda_orth*loss_style_main
        + lambda_orth*total_orth_sub_loss
        + lambda_comp*total_comp_loss
    )

    return total_loss


def token_anchor_loss(
    text_encoder,
    tokenizer,
    person_embeds,
    sub_embeds,
    style_embeds,
    cam_embeds,
    num_vectors=3,
    lambda_person=0.1,
    lambda_part=0.1,
    lambda_style=0.1,
    lambda_cam=0.1,
):
    """
    语义约束：
    <person_i> 接近 person
    <person_i_1~k> 接近 head/body/leg
    <style_IR>/<style_RGB> 接近 "style"
    <cam_x> 接近 "view"
    """

    embedding_layer = text_encoder.get_input_embeddings()

    # ===== 1. 获取 anchor token embedding =====
    person_id = tokenizer.encode("person", add_special_tokens=False)[0]
    head_id = tokenizer.encode("head", add_special_tokens=False)[0]
    body_id = tokenizer.encode("body", add_special_tokens=False)[0]
    leg_id = tokenizer.encode("leg", add_special_tokens=False)[0]

    person_anchor = embedding_layer.weight[person_id]
    part_anchors = torch.stack([
        embedding_layer.weight[head_id],
        embedding_layer.weight[body_id],
        embedding_layer.weight[leg_id],
    ])  # [3, dim]


    loss_person = F.mse_loss(person_embeds, person_anchor.expand_as(person_embeds), reduction="mean")
    target_part_anchors = part_anchors[:num_vectors].unsqueeze(0).expand_as(sub_embeds)
    loss_part = F.mse_loss(sub_embeds, target_part_anchors, reduction="mean")

    #  加权求和
    total_anchor_loss = (
            lambda_person * loss_person +
            lambda_part * loss_part
    )

    return total_anchor_loss




def compute_triplet_loss(anchor,positive,margin=0.3):
    """
    anchor:   fake_feat [B, D] -> 预测样本特征
    positive: real_feat [B, D] -> 同一位置对应的真实图原样本特征
    """
    batch_size=anchor.size(0)
    # 利用 randperm 快速随机置换打乱构建负样本（在大 Batch / 行人平衡采样 P*K 架构下高概率为异ID负样本）
    rand_idx = torch.randperm(batch_size, device=anchor.device)
    negative = positive[rand_idx]
    # 距离定义：平方欧氏距离
    pos_dist = (anchor - positive).pow(2).sum(dim=-1)
    neg_dist = (anchor - negative).pow(2).sum(dim=-1)

    # Triplet Hinge Loss 约束
    loss = F.relu(pos_dist - neg_dist + margin).mean()
    return loss


def compute_token_attention_separation_loss(part_attn_maps):
    """
    计算不同部位 sub-token Attention Map 之间的互斥/分离损失（惩罚重叠）。

    参数:
        part_attn_maps (Tensor): [B, num_parts, H, W]，归一化后的各部位 Attention Map
    返回:
        Tensor (Scalar): 互斥损失
    """
    B, K, H, W = part_attn_maps.shape
    if K <= 1:
        return torch.tensor(0.0, device=part_attn_maps.device)

    # 展平空间维度 -> [B, K, H*W]
    flat_attn = part_attn_maps.view(B, K, -1)

    # L2 归一化或直接计算点积重叠
    norm_attn = F.normalize(flat_attn, p=2, dim=-1)

    # 计算两两之间的相似度矩阵 -> [B, K, K]
    overlap_matrix = torch.bmm(norm_attn, norm_attn.transpose(-2, -1))

    # 提取上三角非对角线元素（排除自己和自己的重叠）
    triu_indices = torch.triu_indices(row=K, col=K, offset=1)
    sep_loss = overlap_matrix[:, triu_indices[0], triu_indices[1]].pow(2).mean()

    return sep_loss


# def compute_cross_attention_mask_loss(attn_maps, input_ids, target_masks, target_token_ids, lambda_sep=0.1):
#     """
#     计算 Cross Attention 与真实部位掩码之间的损失，并加上 sub-token Attention 互斥惩罚。
#
#     参数:
#         attn_maps (list of Tensor): UNet 捕获的 Attention Map
#         input_ids (Tensor): [B, text_seq_len] 文本 Token ID
#         target_masks (Tensor): [B, 4, H, W] 真实掩码 (0: Person, 1: Head, 2: Upper, 3: Lower)
#         target_token_ids (Tensor): [B, 4] 部位对应 Token ID
#         lambda_sep (float): 注意力互斥/分离损失的权重
#     """
#     if len(attn_maps) == 0:
#         return torch.tensor(0.0, device=input_ids.device)
#
#     total_mask_loss = 0.0
#     total_sep_loss = 0.0
#     count = 0
#
#     B = input_ids.shape[0]
#
#     for attn in attn_maps:
#         if attn.dim() == 4:
#             attn = attn.mean(dim=1)  # 对多头取平均 -> [B, H_s * W_s, seq_len]
#
#         num_spatial, seq_len = attn.shape[1], attn.shape[2]
#         h_s = w_s = int(num_spatial ** 0.5)
#
#         attn = attn.permute(0, 2, 1).view(B, seq_len, h_s, w_s)
#         downsampled_masks = F.interpolate(target_masks, size=(h_s, w_s), mode="nearest")
#
#         for b in range(B):
#             part_attns = []
#             for part_idx in range(4):
#                 tok_id = target_token_ids[b, part_idx]
#                 matched_indices = (input_ids[b] == tok_id).nonzero(as_tuple=True)[0]
#                 if len(matched_indices) == 0:
#                     continue
#
#                 tok_pos = matched_indices[0]
#                 pred_attn = attn[b, tok_pos]
#
#                 # 归一化注意力范围 [0, 1]
#                 pred_attn_norm = (pred_attn - pred_attn.min()) / (pred_attn.max() - pred_attn.min() + 1e-8)
#
#                 # 1. 区域覆盖 MSE 损失
#                 gt_mask = downsampled_masks[b, part_idx]
#                 total_mask_loss += F.mse_loss(pred_attn_norm, gt_mask)
#
#                 # 收集子部位（Head, Upper, Lower，即 index 1,2,3）用于互斥计算
#                 if part_idx > 0:
#                     part_attns.append(pred_attn_norm)
#
#             # 2. 计算同样本内 sub-tokens 之间的注意力分离/正交损失
#             if len(part_attns) > 1:
#                 stacked_parts = torch.stack(part_attns, dim=0).unsqueeze(0)  # [1, sub_K, h_s, w_s]
#                 total_sep_loss += compute_token_attention_separation_loss(stacked_parts)
#
#             count += 1
#
#     mean_mask_loss = total_mask_loss / (count + 1e-8)
#     mean_sep_loss = total_sep_loss / (count + 1e-8)
#
#     return mean_mask_loss + lambda_sep * mean_sep_loss


ATTENTION_SPATIAL_SHAPES = {
    2048: (64, 32),  # 修正：(H=64, W=32) 对应 512(高) x 256(宽) 下采样 8 倍
}


def compute_cross_attention_mask_loss(
        attn_maps,
        input_ids,
        target_masks,
        target_token_ids,
        margin=0.2,
        eps=1e-6,
):
    """
    基于 Margin 的区域 Cross-Attention 对齐损失 (纯掩码约束版)。

    参数:
        attn_maps (list of Tensor): UNet 捕获的 Cross-Attention 特征层列表
        input_ids (Tensor): [B, seq_len] 文本 Token ID
        target_masks (Tensor): [B, 4, H_gt, W_gt] 真实掩码 (0: Person, 1: Head, 2: Upper, 3: Lower)
        target_token_ids (Tensor): [B, 4] 4个目标部位对应的 Token ID
        margin (float): 区域相对注意力 Margin (期望区域内注意力密度高于区域外至少 margin)
        eps (float): 防止除以 0 的极小值
    """
    if len(attn_maps) == 0:
        return torch.tensor(0.0, device=input_ids.device, dtype=torch.float32, requires_grad=True)

    B, seq_len = input_ids.shape
    device = input_ids.device

    # -------------------------------------------------------------
    # 1. 向量化匹配 Token 在 input_ids 中的位置索引 [B, 4]
    # -------------------------------------------------------------
    # matches: [B, 4, seq_len]
    matches = (input_ids.unsqueeze(1) == target_token_ids.unsqueeze(2))

    # 找到每个 target_token 在文本序列中的位置 (若有多个匹配取首个)
    tok_positions = matches.float().argmax(dim=-1)  # [B, 4]
    has_target_mask = matches.any(dim=-1)  # [B, 4] 标识是否有有效匹配

    total_region_loss = 0.0
    valid_attn_count = 0

    # 掩码下采样缓存字典，避免重复插值
    mask_cache = {}

    for attn in attn_maps:
        # [B, Heads, HW, Seq] -> 平均头数 -> [B, HW, Seq]
        if attn.dim() == 4:
            attn = attn.mean(dim=1)
        elif attn.dim() != 3:
            continue

        B_attn, num_spatial, attn_seq_len = attn.shape
        if num_spatial not in ATTENTION_SPATIAL_SHAPES:
            continue

        h_s, w_s = ATTENTION_SPATIAL_SHAPES[num_spatial]

        # 转换为 [B, Seq, H, W]
        attn = attn.permute(0, 2, 1).reshape(B, attn_seq_len, h_s, w_s)

        # -------------------------------------------------------------
        # 2. 向量化提取 4 个部位的 Attention Map -> [B, 4, H, W]
        # -------------------------------------------------------------
        tok_pos_expanded = tok_positions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h_s, w_s)
        part_attns = torch.gather(attn, dim=1, index=tok_pos_expanded)  # [B, 4, H, W]

        # -------------------------------------------------------------
        # 3. 空间维度概率归一化
        # -------------------------------------------------------------
        attn_spatial_sum = part_attns.sum(dim=[-2, -1], keepdim=True) + eps
        part_attns_norm = part_attns / attn_spatial_sum  # [B, 4, H, W]

        # -------------------------------------------------------------
        # 4. 掩码尺寸调整与缓存 (采用双线性插值保留边缘软边界)
        # -------------------------------------------------------------
        if (h_s, w_s) not in mask_cache:
            downsampled_masks = F.interpolate(
                target_masks.float(), size=(h_s, w_s), mode="bilinear", align_corners=False
            ).to(dtype=attn.dtype, device=device)
            mask_cache[(h_s, w_s)] = downsampled_masks
        else:
            downsampled_masks = mask_cache[(h_s, w_s)]

        # -------------------------------------------------------------
        # 5. 向量化计算区域内/区域外的注意力密度与 Ranking Loss
        # -------------------------------------------------------------
        gt_masks = downsampled_masks  # [B, 4, H, W]
        inv_masks = 1.0 - gt_masks  # [B, 4, H, W]

        mask_area = gt_masks.sum(dim=[-2, -1], keepdim=True)  # [B, 4, 1, 1]
        outside_area = inv_masks.sum(dim=[-2, -1], keepdim=True)  # [B, 4, 1, 1]

        # # 计算区域内和区域外的平均注意力密度
        # attn_inside = (part_attns_norm * gt_masks).sum(dim=[-2, -1], keepdim=True) / (mask_area + eps)
        # attn_outside = (part_attns_norm * inv_masks).sum(dim=[-2, -1], keepdim=True) / (outside_area + eps)
        # 区域内分配到的注意力总和 (0 ~ 1 之间)
        attn_inside = (part_attns_norm * gt_masks).sum(dim=[-2, -1], keepdim=True)

        # 区域外分配到的注意力总和 (0 ~ 1 之间)
        attn_outside = (part_attns_norm * inv_masks).sum(dim=[-2, -1], keepdim=True)

        # print("gt_masks min/max/sum:", gt_masks.min().item(), gt_masks.max().item(), gt_masks.sum().item())
        # print("part_attns_norm sum:", part_attns_norm.sum(dim=[-2, -1]).mean().item())
        # print("attn_inside mean:", attn_inside.mean().item())
        # print("attn_outside mean:", attn_outside.mean().item())
        # print("margin setting:", margin)
        # 排序损失: 期望 attn_inside > attn_outside + margin
        region_loss_matrix = F.relu(margin - attn_inside + attn_outside).squeeze(-1).squeeze(-1)  # [B, 4]

        # 过滤掉极小掩码（防零点异常）以及文本中未匹配到 Token 的部位
        valid_regions = (mask_area.squeeze(-1).squeeze(-1) > eps) & \
                        (outside_area.squeeze(-1).squeeze(-1) > eps) & \
                        has_target_mask

        if valid_regions.any():
            total_region_loss += region_loss_matrix[valid_regions].mean()
            valid_attn_count += 1

    if valid_attn_count == 0:
        return torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)

    return total_region_loss / valid_attn_count




















