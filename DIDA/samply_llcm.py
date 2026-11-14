#!/usr/bin/env python3

import os
import random
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
import safetensors.torch

# ==========================================
# 视角与背景文本池
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

    # 1. 加载 Tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer")

    # 2. 加载 Textual Inversion (TI) embedding
    ti_path = os.path.join(output_dir, "learned_embeds.safetensors")
    state_dict = safetensors.torch.load_file(ti_path)
    learned_tokens = list(state_dict.keys())

    # 3. 添加 TI token
    tokenizer.add_tokens(learned_tokens)

    # 4. 加载 Text Encoder 并调整 Token 嵌入维度
    text_encoder = CLIPTextModel.from_pretrained(base_model_path, subfolder="text_encoder", torch_dtype=torch.float16)
    text_encoder.resize_token_embeddings(len(tokenizer))

    unet = UNet2DConditionModel.from_pretrained(
        os.path.join(output_dir, "unet"),
        torch_dtype=torch.float16
    )

    # 5. 加载 Pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        unet=unet,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)

    # 6. 填入 TI 权重
    token_embeds = pipeline.text_encoder.get_input_embeddings().weight.data
    for token, embed in state_dict.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        token_embeds[token_id] = embed.to(device=device, dtype=torch.float16)

    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    return pipeline


def generate_nir_for_label(
        pipeline,
        data_root,
        num_vectors,
        label,           # 连续标签 (整型, 如 713)
        pid,             # 真实的行人 ID 文件夹名称 (字符串, 如 '1152')
        num_img=30,      # 为该 ID 生成的红外图像数量
        batch_size=8,
        lambda_div=0.3,
        tau=5.0,
        guide_ratio=0.9,
        update_idx_file=True
):
    """
    为指定的 Label 和 PID 生成红外 (NIR) 图像并更新 train_nir.txt
    """
    device = pipeline.device
    weight_dtype = pipeline.unet.dtype

    # 构建 Person Tokens，注意 token 映射使用的是连续的 label
    person_tokens = [f"<person_{label}>"] + [f"<person_{label}_{v}>" for v in range(1, num_vectors + 1)]
    person_str = " ".join(person_tokens)

    # 目标保存路径：nir/1152/
    target_dir = os.path.join(data_root, 'nir', pid)
    os.makedirs(target_dir, exist_ok=True)

    # 固定模态风格为 IR / NIR
    image_style = 'IR'

    print(f"\n[NIR Generation] Generating {num_img} infrared images for [Label: {label}, PID: {pid}]...")

    z0_bank = []
    pipeline.unet.requires_grad_(False)

    total_batches = (num_img + batch_size - 1) // batch_size
    img_counter = 1
    generated_idx_entries = []

    for b_idx in range(total_batches):
        cur_batch_size = min(batch_size, num_img - (b_idx * batch_size))

        # 1. 构造红外 Prompt
        prompts = []
        for i in range(cur_batch_size):
            global_img_idx = b_idx * batch_size + i
            viewpoint_modifier = VIEWPOINTS[global_img_idx % len(VIEWPOINTS)]
            background_modifier = random.choice(BACKGROUNDS)
            prompts.append(f"a <style_{image_style}> photo of {person_str}, {viewpoint_modifier}, {background_modifier}")

        # 2. 文本编码
        text_inputs = pipeline.tokenizer(prompts, padding='max_length',
                                         max_length=pipeline.tokenizer.model_max_length,
                                         truncation=True, return_tensors='pt')
        text_embeddings = pipeline.text_encoder(text_inputs.input_ids.to(device))[0].to(dtype=weight_dtype)

        uncond_inputs = pipeline.tokenizer([""] * cur_batch_size, padding='max_length',
                                           max_length=pipeline.tokenizer.model_max_length,
                                           truncation=True, return_tensors='pt')
        uncond_embeddings = pipeline.text_encoder(uncond_inputs.input_ids.to(device))[0].to(dtype=weight_dtype)
        current_text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # 3. 初始高斯噪声
        height, width = 512, 256
        latents = torch.randn((cur_batch_size, pipeline.unet.config.in_channels, height // 8, width // 8),
                              device=device, dtype=weight_dtype)

        pipeline.scheduler.set_timesteps(25, device=device)
        timesteps = pipeline.scheduler.timesteps
        total_steps = len(timesteps)

        if hasattr(pipeline.scheduler, '_step_index'):
            pipeline.scheduler._step_index = None

        latents = latents * pipeline.scheduler.init_noise_sigma

        # 4. 去噪 Loop + 多样性引导
        for step_idx, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

            with torch.no_grad():
                noise_pred = pipeline.unet(latent_model_input, t,
                                           encoder_hidden_states=current_text_embeddings).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            cfg_noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)

            # 多样性引导 (Pairwise Repulsion)
            if len(z0_bank) > 0 and int(total_steps * 0.6) < step_idx < int(total_steps * guide_ratio):
                orig_norms = latents.view(cur_batch_size, -1).norm(dim=1, keepdim=True).detach()
                latents = latents.detach().requires_grad_(True)

                alpha_prod_t = pipeline.scheduler.alphas_cumprod[t].to(device).to(weight_dtype)
                beta_prod_t = 1.0 - alpha_prod_t
                pred_z0 = (latents - (beta_prod_t ** 0.5) * cfg_noise_pred) / (alpha_prod_t ** 0.5)

                cur_z0_flat = F.normalize(pred_z0.view(cur_batch_size, -1), p=2, dim=1)
                bank_tensor = F.normalize(torch.stack(z0_bank, dim=0).detach(), p=2, dim=1)

                cos_sim = torch.mm(cur_z0_flat, bank_tensor.T)
                loss_div = torch.mean(torch.exp(tau * cos_sim.to(torch.float32)))

                grad = torch.autograd.grad(loss_div, latents)[0]
                grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
                grad_flat = grad.view(cur_batch_size, -1)
                grad_norms = grad_flat.norm(p=2, dim=1, keepdim=True) + 1e-8
                grad_normalized = (grad_flat / grad_norms).view_as(latents)

                latents = latents - lambda_div * grad_normalized

                latents_flat = latents.view(cur_batch_size, -1)
                latents_mean = latents_flat.mean(dim=1, keepdim=True)
                latents_norm = latents_flat.norm(dim=1, keepdim=True) + 1e-8
                latents_flat = (latents_flat - latents_mean) / latents_norm * orig_norms
                latents = latents_flat.view_as(latents).detach()

            with torch.no_grad():
                latents = pipeline.scheduler.step(cfg_noise_pred, t, latents).prev_sample

        # 5. 存入 Memory Bank
        final_z0 = latents.detach()
        for i in range(cur_batch_size):
            z0_bank.append(final_z0[i].view(-1).detach().clone())

        # 6. 解码并保存图像
        with torch.no_grad():
            dec_latents = final_z0 / pipeline.vae.config.scaling_factor
            images = pipeline.vae.decode(dec_latents).sample
            images = (images / 2 + 0.5).clamp(0, 1)
            images = torch.nan_to_num(images, nan=0.0, posinf=1.0, neginf=0.0)
            images = images.cpu().permute(0, 2, 3, 1).float().numpy()

        for i in range(cur_batch_size):
            img_pil = Image.fromarray((images[i] * 255).astype(np.uint8))

            # 生成符合 LLCM 风格的红外文件名，如 1152_c11_f0001_nir.jpg
            new_image_name = f"{pid}_c11_m{label-713+1:04d}_f{img_counter:04d}_nir.jpg"
            img_pil.save(os.path.join(target_dir, new_image_name))

            # 追加的索引路径：nir/1152/1152_c11_f0001_nir.jpg 713
            rel_path = f"nir/{pid}/{new_image_name}"
            generated_idx_entries.append(f"{rel_path} {label}\n")
            img_counter += 1

    # 7. 追加保存到 train_nir.txt
    if update_idx_file:
        idx_dir = os.path.join(data_root, "idx")
        os.makedirs(idx_dir, exist_ok=True)
        target_idx_file = os.path.join(idx_dir, "train_nir.txt")

        with open(target_idx_file, "a") as f:
            f.writelines(generated_idx_entries)
        print(f"追加 {len(generated_idx_entries)} 条索引至: {target_idx_file}")

    del z0_bank
    torch.cuda.empty_cache()


def main():
    pipe = load_inference_pipeline(
        base_model_path="../paper_code/stable-diffusion-v1-5",
        output_dir="./text-inversion-model/checkpoint-15000_1ti2att-cond"
    )

    data_root = "/home/lk/PyCharmMiscProject/data/LLCM_Market"
    idx_vis_file = os.path.join(data_root, "idx", "train_vis.txt")
    start_label = 713  # 起始阈值：筛选 >= 713 的标签

    if not os.path.exists(idx_vis_file):
        print(f"Error: 找不到索引文件 '{idx_vis_file}'")
        return

    # ----------------------------------------------------------------------
    # 解析 train_vis.txt，提取 label >= 713 的条目并建立 label_to_pid 字典
    # ----------------------------------------------------------------------
    label_to_pid = {}

    with open(idx_vis_file, "r") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue

            parts = line_str.split()
            if len(parts) < 2:
                continue

            rel_path, label_str = parts[0], parts[1]
            label = int(label_str)

            # 只保留 label >= 713 的新增数据
            if label >= start_label:
                # 相对路径格式如: vis/1152/1152_c10_m0001_f0001_vis.jpg
                path_parts = rel_path.split('/')
                if len(path_parts) >= 2:
                    pid = path_parts[1]  # 提取 PID，如 '1152'
                    if label not in label_to_pid:
                        label_to_pid[label] = pid

    sorted_labels = sorted(label_to_pid.keys())
    print(f"解析完成！找到 {len(sorted_labels)} 个待生成红外图像的 ID (Label 范围: {sorted_labels[0]} ~ {sorted_labels[-1]})")

    # ----------------------------------------------------------------------
    # 遍历生成红外图像
    # ----------------------------------------------------------------------
    for label in sorted_labels:
        pid = label_to_pid[label]
        generate_nir_for_label(
            pipeline=pipe,
            data_root=data_root,
            num_vectors=3,
            label=label,
            pid=pid,
            num_img=30,  # 每个 ID 生成的红外图片张数
            batch_size=8,
            update_idx_file=True
        )

    print("\n所有红外图像生成并索引更新完毕！")


if __name__ == '__main__':
    main()