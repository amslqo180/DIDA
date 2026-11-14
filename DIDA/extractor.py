import os
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T

# 导入项目中原有的模块
from AGW.config import cfg
from AGW.modeling import build_model
from DEEN.model import embed_net


class MarketFeatureExtractor:
    def __init__(self, config_file, weight_path, device='cuda'):
        self.device = device

        # 1. 加载配置
        cfg.merge_from_file(config_file)
        cfg.MODEL.DEVICE = device
        cfg.freeze()

        # 2. 构建模型结构
        self.model = build_model(cfg, num_classes=751)

        # 3. 加载预训练权重 (去掉分类头参数)[cite: 3]
        if hasattr(self.model, 'module'):
            self.model.module.load_param(weight_path)
        else:
            self.model.load_param(weight_path)

        self.model.to(self.device)
        self.model.eval()  # 开启 eval 模式，此时 forward 只返回 (B, 2048) 的嵌入[cite: 3]

        # 4. 定义标准 Re-ID 图像预处理流程
        self.transform = T.Compose([
            T.Resize((256, 128)),  # Re-ID 标准高宽尺寸 (Height, Width)
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def extract_from_tensor(self, img_tensor):
        """
        输入格式: torch.Tensor, 形状 (B, C, H, W)，值域范围 [0, 1]
        """
        # 如果尺寸不是 (256, 128)，需要 Resize 到 Re-ID 常用尺寸
        if img_tensor.shape[-2:] != (256, 128):
            img_tensor = torch.nn.functional.interpolate(
                img_tensor, size=(256, 128), mode='bilinear', align_corners=False
            )

        # 归一化 (使用 ImageNet mean/std)
        mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(1, 3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        # 提取身份特征 (B, 2048)
        id_embeds = self.model(img_tensor.to(self.device))
        return id_embeds

    @torch.no_grad()
    def extract_from_pil(self, pil_images):
        """
        输入格式: PIL Image 列表 或 单张 PIL Image
        """
        if not isinstance(pil_images, list):
            pil_images = [pil_images]

        batch_tensor = torch.stack([self.transform(img) for img in pil_images]).to(self.device)
        id_embeds = self.model(batch_tensor)  # (B, 2048)
        return id_embeds


class DEENFeatureExtractor(nn.Module):
    """
    DEEN 特征提取器类：针对 SYSU/RegDB 等多模态数据集提取跨模态行人特征。
    """

    def __init__(self, model_path=None, dataset='sysu', arch='resnet50', num_classes=395, device='cuda'):
        super().__init__()
        self.dataset = dataset
        self.device = torch.device(device)
        self.feat_dim = 1024 if dataset == 'regdb' else 2048

        # 1. 初始化并加载 DEEN 网络
        self.deen = embed_net(num_classes, dataset=dataset, arch=arch)
        if model_path and os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location="cpu")
            if 'net' in state_dict:
                state_dict = state_dict['net']
            self.deen.load_state_dict(state_dict)
            print(f"Successfully loaded weights from {model_path}")
        else:
            if model_path:
                print(f"Warning: Weight file not found at {model_path}")

        # 2. 锁住参数，设置为评估模式
        self.deen.eval()
        self.deen.requires_grad_(False)
        self.deen.to(self.device)

        # 3. 注册 ImageNet 归一化的 mean 和 std 为 buffer
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        )

    @torch.no_grad()
    def extract_features(self, pixel_values, cam_ids):
        """
        根据 cam_ids 自动分流 RGB / IR 并提取特征
        pixel_values: 范围为 [-1, 1] 的图像 Tensor (B, C, H, W)
        cam_ids: 摄像头 ID Tensor (B,)
        """
        # 记录原始 dtype，用于最后还原
        orig_dtype = pixel_values.dtype

        # 统一把输入图像转换为 float32，避免与 DEEN 模型权重的 float32 冲突
        pixel_values = pixel_values.to(dtype=torch.float32)
        B = pixel_values.size(0)
        feats = torch.zeros((B, self.feat_dim), device=pixel_values.device, dtype=pixel_values.dtype)

        # 判断模态 Mask (摄像头 3 和 6 为红外 IR)
        ir_mask = (cam_ids == 3) | (cam_ids == 6)
        rgb_mask = ~ir_mask

        # 1. [-1, 1] -> [0, 1]
        x_rescaled = (pixel_values + 1.0) / 2.0

        # 2. ImageNet 归一化 (匹配正确的 dtype 和 device)
        mean = self.mean.to(dtype=pixel_values.dtype, device=pixel_values.device)
        std = self.std.to(dtype=pixel_values.dtype, device=pixel_values.device)
        x_norm = (x_rescaled - mean) / std

        # 3. 提取 RGB 特征
        if rgb_mask.any():
            x_rgb = x_norm[rgb_mask]
            n_samples = x_rgb.size(0)

            feat, feat_att = self.deen(x_rgb, None, modal=1)
            feat1 = feat[:n_samples]
            feat2 = feat_att[:n_samples]
            feat3 = feat[n_samples:n_samples * 2]
            feat4 = feat_att[n_samples:n_samples * 2]
            feat5 = feat[n_samples * 2:]
            feat6 = feat_att[n_samples * 2:]
            f_rgb_fused = (feat1 + feat2 + feat3 + feat4 + feat5 + feat6) / 6.0

            feats[rgb_mask] = f_rgb_fused.to(dtype=pixel_values.dtype)

        # 4. 提取 IR 特征
        if ir_mask.any():
            x_ir = x_norm[ir_mask]
            n_samples = x_ir.size(0)

            feat, feat_att = self.deen(None, x_ir, modal=2)
            feat1 = feat[:n_samples]
            feat2 = feat_att[:n_samples]
            feat3 = feat[n_samples:n_samples * 2]
            feat4 = feat_att[n_samples:n_samples * 2]
            feat5 = feat[n_samples * 2:]
            feat6 = feat_att[n_samples * 2:]
            f_ir_fused = (feat1 + feat2 + feat3 + feat4 + feat5 + feat6) / 6.0

            feats[ir_mask] = f_ir_fused.to(dtype=pixel_values.dtype)

        return feats.to(dtype=orig_dtype)

    def forward(self, pixel_values, cam_ids):
        return self.extract_features(pixel_values, cam_ids)