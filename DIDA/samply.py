import os
import random
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler,UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
import safetensors.torch

# ==========================================
# 1. 视角与背景文本池 (保留原有功能)
# ==========================================
VIEWPOINTS = [
    "front view, facing the camera",
    "back view, walking away from the camera",
    "side view, walking profile",
    "three-quarter view, turned slightly away",
]

BACKGROUNDS = [
    "on a busy city sidewalk",
    "crossing a pedestrian zebra lane",
    "in an indoor concrete parking lot",
    "at a public park crossroad",
    "near a subway station entrance"
]
def load_inference_pipeline(base_model_path, output_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 加载训练时保存的 Tokenizer (包含新增的 person tokens)
    tokenizer = CLIPTokenizer.from_pretrained(base_model_path,subfolder="tokenizer")
    # 2. 加载TI embedding
    ti_path=os.path.join(output_dir,"learned_embeds.safetensors")
    state_dict=safetensors.torch.load_file(ti_path)
    learned_tokens=list(state_dict.keys())
    # 3. 添加TI token
    tokenizer.add_tokens(learned_tokens)

    # 2. 加载基础 Text Encoder 并调整大小
    text_encoder = CLIPTextModel.from_pretrained(base_model_path, subfolder="text_encoder", torch_dtype=torch.float16)
    text_encoder.resize_token_embeddings(len(tokenizer))

    unet = UNet2DConditionModel.from_pretrained(
        os.path.join(output_dir, "unet"),
        torch_dtype=torch.float16
    )

    # 3. 加载 Pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        unet=unet,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)

    # # 4. 加载 TI 嵌入权重
    # ti_path = os.path.join(output_dir, "learned_embeds.safetensors")
    # state_dict = safetensors.torch.load_file(ti_path)
    # 将权重填入 text_encoder
    token_embeds = pipeline.text_encoder.get_input_embeddings().weight.data
    for token, embed in state_dict.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        token_embeds[token_id] = embed.to(device=device, dtype=torch.float16)

    # 5. 加载 LoRA 权重 (现在支持标准加载了)
    # pipeline.load_lora_weights(os.path.join(output_dir, "unet_lora"))
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    return pipeline

def generate_sysu_expansion(
        pipeline,
        save_root,
        num_vectors,
        person_id,
        cam_id,
        num_img,
        label_to_id,
        batch_size=8,        # 一次生成的图片张数，可根据显存调整（如 4 或 8）
        lambda_div=0.3,      # 多样性引导步长 \lambda_d
        tau=5.0,             # Pairwise Repulsion 温度系数 \tau
        guide_ratio=0.9      # 前 40% 时间步施加多样性引导
):
    device = pipeline.device
    weight_dtype = pipeline.unet.dtype

    # 构建 Person Tokens
    person_tokens = [f"<person_{person_id}>"] + [f"<person_{person_id}_{v}>" for v in range(1, num_vectors + 1)]
    person_str = " ".join(person_tokens)

    # 保存路径
    person_dir_name = str(label_to_id[person_id]).zfill(4)
    target_dir = os.path.join(save_root, 'cam8', person_dir_name)
    os.makedirs(target_dir, exist_ok=True)

    cam_num = int(cam_id[-1]) if cam_id[-1].isdigit() else 9
    image_style = 'IR'

    print(f"\nGenerating {num_img} diverse images (Batch size: {batch_size}) for [CAM: {cam_id}, Person: {person_dir_name}]...")

    # 记忆库：存储已生成的 z0 (维度: [N, D])
    z0_bank = []

    # 冻结 UNet 参数以优化梯度传递
    pipeline.unet.requires_grad_(False)

    total_batches = (num_img + batch_size - 1) // batch_size
    img_counter = 1

    for b_idx in range(total_batches):
        cur_batch_size = min(batch_size, num_img - (b_idx * batch_size))

        # 1. 构建当前 Batch 的 Prompt 列表
        prompts = []
        for i in range(cur_batch_size):
            global_img_idx = b_idx * batch_size + i
            viewpoint_modifier = VIEWPOINTS[global_img_idx % len(VIEWPOINTS)]
            background_modifier = random.choice(BACKGROUNDS)
            prompts.append(f"a <style_{image_style}> photo of {person_str}, {viewpoint_modifier}, {background_modifier}")

        # 2. 批量文本编码
        text_inputs = pipeline.tokenizer(prompts, padding='max_length',
                                         max_length=pipeline.tokenizer.model_max_length,
                                         truncation=True, return_tensors='pt')
        text_embeddings = pipeline.text_encoder(text_inputs.input_ids.to(device))[0].to(dtype=weight_dtype)

        uncond_inputs = pipeline.tokenizer([""] * cur_batch_size, padding='max_length',
                                           max_length=pipeline.tokenizer.model_max_length,
                                           truncation=True, return_tensors='pt')
        uncond_embeddings = pipeline.text_encoder(uncond_inputs.input_ids.to(device))[0].to(dtype=weight_dtype)
        current_text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # 3. 采样初始高斯噪声 z_T，Shape: [cur_batch_size, C, H, W]
        height, width = 512, 256
        latents = torch.randn((cur_batch_size, pipeline.unet.config.in_channels, height // 8, width // 8),
                              device=device, dtype=weight_dtype)

        pipeline.scheduler.set_timesteps(25, device=device)
        timesteps = pipeline.scheduler.timesteps
        total_steps = len(timesteps)

        # 重置 Scheduler 索引
        if hasattr(pipeline.scheduler, '_step_index'):
            pipeline.scheduler._step_index = None

        latents = latents * pipeline.scheduler.init_noise_sigma

        # 4. 去噪 Loop
        for step_idx, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

            with torch.no_grad():
                noise_pred = pipeline.unet(latent_model_input, t,
                                           encoder_hidden_states=current_text_embeddings).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            cfg_noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)

            # ====== 早期采样轨迹多样性引导 (Pairwise Repulsion) ======
            if len(z0_bank) > 0 and int(total_steps * 0.6)<step_idx < int(total_steps * guide_ratio):
                # 记录按样本独立的原始模长
                orig_norms = latents.view(cur_batch_size, -1).norm(dim=1, keepdim=True).detach()  # [B, 1]
                latents = latents.detach().requires_grad_(True)

                # 单步推导 \hat{z}_0
                alpha_prod_t = pipeline.scheduler.alphas_cumprod[t].to(device).to(weight_dtype)
                beta_prod_t = 1.0 - alpha_prod_t
                pred_z0 = (latents - (beta_prod_t ** 0.5) * cfg_noise_pred) / (alpha_prod_t ** 0.5)

                # 展平与 L2 归一化
                cur_z0_flat = F.normalize(pred_z0.view(cur_batch_size, -1), p=2, dim=1)            # [B, D]
                bank_tensor = F.normalize(torch.stack(z0_bank, dim=0).detach(), p=2, dim=1)  # [N, D]

                # 批量计算 Cosine 相似度矩阵: [B, N]
                cos_sim = torch.mm(cur_z0_flat, bank_tensor.T)

                # Pairwise Repulsion Loss
                cov_f32 = cos_sim.to(torch.float32)
                loss_div = torch.mean(torch.exp(tau * cov_f32))

                # 计算 Latent 梯度
                grad = torch.autograd.grad(loss_div, latents)[0]

                # 清理异常数值并按样本归一化梯度
                grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
                grad_flat = grad.view(cur_batch_size, -1)
                grad_norms = grad_flat.norm(p=2, dim=1, keepdim=True) + 1e-8
                grad_normalized = (grad_flat / grad_norms).view_as(latents)

                # 施加排斥更新
                latents = latents - lambda_div * grad_normalized

                # 重新校准每个样本的模长
                latents_flat = latents.view(cur_batch_size, -1)
                latents_mean = latents_flat.mean(dim=1, keepdim=True)
                latents_norm = latents_flat.norm(dim=1, keepdim=True) + 1e-8
                latents_flat = (latents_flat - latents_mean) / latents_norm * orig_norms
                latents = latents_flat.view_as(latents).detach()

            # 执行标准 Diffusion 下一步采样
            with torch.no_grad():
                latents = pipeline.scheduler.step(cfg_noise_pred, t, latents).prev_sample

        # 5. 当前 Batch 去噪结束，将生成的每个样本提取最终 z0 存入记忆库
        final_z0 = latents.detach()
        for i in range(cur_batch_size):
            z0_bank.append(final_z0[i].view(-1).detach().clone())

        # 6. 批量解码保存图像
        with torch.no_grad():
            dec_latents = final_z0 / pipeline.vae.config.scaling_factor
            images = pipeline.vae.decode(dec_latents).sample
            images = (images / 2 + 0.5).clamp(0, 1)
            images = torch.nan_to_num(images, nan=0.0, posinf=1.0, neginf=0.0)
            images = images.cpu().permute(0, 2, 3, 1).float().numpy()

        for i in range(cur_batch_size):
            img_pil = Image.fromarray((images[i] * 255).astype(np.uint8))
            generate_num = str(img_counter).zfill(4)
            img_pil.save(os.path.join(target_dir, f"{generate_num}.png"))
            img_counter += 1

    del z0_bank
    torch.cuda.empty_cache()


def main():
    pipe = load_inference_pipeline("../paper_code/stable-diffusion-v1-5",
                                   "./text-inversion-model/checkpoint-15000_1ti2att-cond")
    data_root = "../data/SYSU_Market/"

    id_file = os.path.join(data_root, 'exp', 'train_id.txt')
    with open(id_file, 'r') as f:
        content = f.read()
        raw_ids = content.replace('\n', ',').split(',')
    ids = [i.zfill(4) for i in raw_ids if i.strip()]

    set_ids = sorted(list(set([int(i) for i in ids])))
    id_to_label = {pid: i for i, pid in enumerate(set_ids)}
    label_to_id = {i: pid for i, pid in enumerate(set_ids)}

    sysu_cams = ['cam3', 'cam6']
    market_cams = ['cam7']

    for id in ids:
        for cam in market_cams:
            img_dir = os.path.join(data_root, cam, id)
            if not os.path.exists(img_dir):
                continue
            label = id_to_label[int(id)]
            # print(f"generate image for {os.path.join(data_root, cam, id)}")
            generate_sysu_expansion(pipeline=pipe, save_root=data_root, num_vectors=3, person_id=label, cam_id=cam,
                                    num_img=30, label_to_id=label_to_id)

if __name__ == '__main__':
    main()