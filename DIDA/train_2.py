#!/usr/bin/env python
# coding=utf-8
# Textual Inversion fine-tuning script (single GPU version)
import sys
import os
agw_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AGW')
if agw_dir not in sys.path:
    sys.path.insert(0, agw_dir)
import argparse
import logging
import math
import os
import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import PIL
import safetensors
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict,get_peft_model, PeftModel
from torch import nn
from torch.utils.data import Dataset, BatchSampler
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available, convert_state_dict_to_diffusers
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from loss import token_regularization_loss, token_anchor_loss, compute_triplet_loss, compute_cross_attention_mask_loss
from dataset import TextualInversionDataset
from Sampler import IdentitySampler
from DEEN.model import embed_net
from extractor import MarketFeatureExtractor
from safetensors.torch import save_file

warnings.filterwarnings("ignore",message="You have disabled the safety checker")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=torch.serialization.SourceChangeWarning)
if is_wandb_available():
    import wandb

# check_min_version("0.39.0.dev0")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_text_features(text_encoder, token_ids):
    """
    token_ids: (B, N) or (B,)
    """
    embeds = text_encoder.get_input_embeddings()(token_ids)  # (B,N,D) or (B,D)

    if embeds.dim() == 3:
        embeds = embeds.mean(dim=1)  # multi-token → 平均

    return embeds  # (B,D)

def extract_deen_features(deen,pixel_values,cam_ids):
    """
    使用 DEEN 模型提取多模态行人特征。
    根据相机的 ID (3, 6 为红外，其余为 RGB) 进行图像分流并进行标准 ImageNet 归一化。
    """
    B=pixel_values.size(0)
    feat_dim=1024 if deen.dataset=='regdb' else 2048
    feats = torch.zeros((B, feat_dim), device=pixel_values.device, dtype=pixel_values.dtype)

    # 区分红外相机(IR)与可见光相机(RGB)
    ir_mask = (cam_ids == 3) | (cam_ids == 6)
    rgb_mask = ~ir_mask

    # 扩散模型的图像像素在 [-1, 1] 之间，需要恢复到 [0, 1] 然后进行 ImageNet 归一化
    x_rescaled = (pixel_values + 1.0) / 2.0
    mean = torch.tensor([0.485, 0.456, 0.406], device=pixel_values.device,dtype=pixel_values.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=pixel_values.device,dtype=pixel_values.dtype).view(1, 3, 1, 1)
    x_norm = (x_rescaled - mean) / std

    # 提取可见光图像特征 (modal=1)
    if rgb_mask.any():
        x_rgb = x_norm[rgb_mask]
        n_samples=x_rgb.size(0)
        with torch.no_grad():
            feat, feat_att = deen(x_rgb, None, modal=1)
            # 严格按照论文测试代码的切片方式，拆解出全部 6 组特征
            feat1 = feat[:n_samples]
            feat2 = feat_att[:n_samples]
            feat3 = feat[n_samples:n_samples * 2]
            feat4 = feat_att[n_samples:n_samples * 2]
            feat5 = feat[n_samples * 2:]
            feat6 = feat_att[n_samples * 2:]

            f_rgb_fused = (feat1 + feat2 + feat3 + feat4 + feat5 + feat6) / 6.0

        feats[rgb_mask] = f_rgb_fused.to(dtype=pixel_values.dtype)

    # 提取红外图像特征 (modal=2)
    if ir_mask.any():
        x_ir = x_norm[ir_mask]
        n_samples = x_ir.size(0)
        with torch.no_grad():
            feat, feat_att = deen(None, x_ir, modal=2)
            feat1 = feat[:n_samples]
            feat2 = feat_att[:n_samples]
            feat3 = feat[n_samples:n_samples * 2]
            feat4 = feat_att[n_samples:n_samples * 2]
            feat5 = feat[n_samples * 2:]
            feat6 = feat_att[n_samples * 2:]
            f_ir_fused = (feat1 + feat2 + feat3 + feat4 + feat5 + feat6) / 6.0
        feats[ir_mask] = f_ir_fused.to(dtype=pixel_values.dtype)

    return feats

def extract_agw_features(agw_model, pixel_values, cam_ids):
    """
    使用 AGW (embed_net) 模型提取多模态行人特征。
    根据相机的 ID (3, 6 为红外，其余为 RGB) 进行图像分流并进行标准 ImageNet 归一化。
    """
    B = pixel_values.size(0)
    feat_dim = 2048  # AGW 的 pool_dim 固定为 2048
    feats = torch.zeros((B, feat_dim), device=pixel_values.device, dtype=pixel_values.dtype)

    # 区分红外相机(IR)与可见光相机(RGB)
    ir_mask = (cam_ids == 3) | (cam_ids == 6)
    rgb_mask = ~ir_mask

    # 扩散模型的图像像素在 [-1, 1] 之间，需要恢复到 [0, 1] 然后进行 ImageNet 归一化
    x_rescaled = (pixel_values + 1.0) / 2.0
    mean = torch.tensor([0.485, 0.456, 0.406], device=pixel_values.device, dtype=pixel_values.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=pixel_values.device, dtype=pixel_values.dtype).view(1, 3, 1, 1)
    x_norm = (x_rescaled - mean) / std

    # 提取可见光图像特征 (modal=1)
    if rgb_mask.any():
        x_rgb = x_norm[rgb_mask]
        with torch.no_grad():
            # AGW 测试模式下返回: (l2norm(x_pool), l2norm(feat))
            # 通常使用经过 bottleneck 和 l2norm 后的特征 (即第二个返回值)
            _, feat_rgb = agw_model(x_rgb, None, modal=1)
        feats[rgb_mask] = feat_rgb.to(dtype=pixel_values.dtype)

    # 提取红外图像特征 (modal=2)
    if ir_mask.any():
        x_ir = x_norm[ir_mask]
        with torch.no_grad():
            _, feat_ir = agw_model(None, x_ir, modal=2)
        feats[ir_mask] = feat_ir.to(dtype=pixel_values.dtype)

    return feats

def save_model_card(repo_id, images=None, base_model=None, repo_folder=None):
    img_str = ""
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            img_str += f"![img_{i}](./image_{i}.png)\n"
    model_description = f"""
# Textual inversion text2image fine-tuning - {repo_id}
These are textual inversion adaption weights for {base_model}.\n
{img_str}
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )
    tags = ["stable-diffusion", "stable-diffusion-diffusers", "text-to-image", "diffusers", "textual_inversion",
            "diffusers-training"]
    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))


def log_validation(args,label_to_id,device, weight_dtype, step):
    num_ids = 10
    rand_ids = random.sample(range(1307), num_ids)
    # rand_ids=[5,10,200,250,530,600,650,850,1000,1200]
    # temp_rng = random.Random(step)
    # test_cam_rgb = temp_rng.choice([1, 2, 4, 5, 7])
    # test_cam_ir = temp_rng.choice([3, 6])

    # logger.info(
    #     f"[Validation Settings] Step: {step} | Selected RGB Cam: <cam_{test_cam_rgb}> | Selected IR Cam: <cam_{test_cam_ir}>")

    base_model_path=args.pretrained_model_name_or_path
    output_dir_name=os.path.join(args.output_dir,f"checkpoint-{step}")
    # 1. 加载训练时保存的 Tokenizer (包含新增的 person tokens)
    tokenizer = CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer")
    # 2. 加载TI embedding
    ti_path = os.path.join(output_dir_name, "learned_embeds.safetensors")
    state_dict = safetensors.torch.load_file(ti_path)
    learned_tokens = list(state_dict.keys())
    # 3. 添加TI token
    tokenizer.add_tokens(learned_tokens)

    # 2. 加载基础 Text Encoder 并调整大小
    text_encoder = CLIPTextModel.from_pretrained(base_model_path, subfolder="text_encoder", torch_dtype=torch.float16)
    text_encoder.resize_token_embeddings(len(tokenizer))

    unet = UNet2DConditionModel.from_pretrained(
    os.path.join(output_dir_name, "unet"),
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

    # 将权重填入 text_encoder
    token_embeds = pipeline.text_encoder.get_input_embeddings().weight.data
    for token, embed in state_dict.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        token_embeds[token_id] = embed.to(device=device, dtype=torch.float16)

    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    ### 目录结构
    epoch_dir = os.path.join(args.output_dir, "validation", f"step_{step}")
    os.makedirs(epoch_dir, exist_ok=True)

    generator = None if args.seed is None else torch.Generator(device=device).manual_seed(args.seed)

    for rand_id in rand_ids:
        person_tokens = [f"<person_{rand_id}>"]
        for j in range(1, args.num_vectors + 1):
            person_tokens.append(f"<person_{rand_id}_{j}>")
        person_string = " ".join(person_tokens)

        validation_prompt_rgb = f"a <style_RGB> photo of a {person_string}"
        validation_prompt_ir = f"a <style_IR> photo of a {person_string}"
        logger.info(f"Validation RGB prompt: {validation_prompt_rgb}")
        logger.info(f"Validation IR prompt: {validation_prompt_ir}")
        images_rgb = []
        images_ir = []

        for _ in range(args.num_validation_images):
            autocast_ctx = nullcontext() if torch.backends.mps.is_available() else torch.autocast(device.type)
            with autocast_ctx:
                image = pipeline(validation_prompt_rgb, height=args.img_h, width=args.img_w, num_inference_steps=25,
                                 generator=generator).images[0]
            images_rgb.append(image)

        for _ in range(args.num_validation_images):
            autocast_ctx = nullcontext() if torch.backends.mps.is_available() else torch.autocast(device.type)
            with autocast_ctx:
                image = pipeline(validation_prompt_ir, height=args.img_h, width=args.img_w, num_inference_steps=25,
                                 generator=generator).images[0]
            images_ir.append(image)

        # numpy 拼接
        np_rgb = [np.asarray(img) for img in images_rgb]
        np_ir = [np.asarray(img) for img in images_ir]

        row_rgb = np.concatenate(np_rgb, axis=1)
        row_ir = np.concatenate(np_ir, axis=1)

        merged_image = np.concatenate([row_rgb, row_ir], axis=0)

        merged_image = Image.fromarray(
            merged_image.astype(np.uint8)
        )
        save_path = os.path.join(
            epoch_dir,
            f"{label_to_id[rand_id]}.png"
        )

        merged_image.save(save_path)

        logger.info(f"Validation image saved to: {save_path}")

    del pipeline
    torch.cuda.empty_cache()
    return images_rgb + images_ir


def save_progress(text_encoder, placeholder_token_ids, placeholder_tokens, args, save_path, safe_serialization=True):
    logger.info("Saving embeddings")
    learned_embeds = (
        text_encoder.get_input_embeddings()
        .weight[placeholder_token_ids]
    )
    learned_embeds_dict = {token: learned_embeds[i].detach().cpu()
                           for i, token in enumerate(placeholder_tokens)
                           }
    if safe_serialization:
        safetensors.torch.save_file(learned_embeds_dict, save_path, metadata={"format": "pt"})
    else:
        torch.save(learned_embeds_dict, save_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_name", type=str, default="sysu", choices=["sysu", "llcm"],help="The dataset to train on. Choice between 'sysu' or 'llcm'.")
    parser.add_argument("--num_vectors", type=int, default=1)
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--placeholder_token", type=str, nargs="+", required=True)
    parser.add_argument("--initializer_token", type=str, nargs="+", required=True)
    parser.add_argument("--num_camers", type=int, default=7)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="Dropout probability for LoRA layers")
    parser.add_argument("--stage2_lr_decay", type=float, default=0.5, help="Stage 2 learning rate decay factor")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="text-inversion-model")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--img_h", type=int, default=384)
    parser.add_argument("--img_w", type=int, default=144)
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--P", type=int, default=8)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=10000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--ti_lr", type=float, default=5e-4)
    parser.add_argument("--lora_lr", type=float, default=1e-4)
    parser.add_argument("--scale_lr", action="store_true", default=False)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--validation_true", action="store_true")
    parser.add_argument("--num_validation_images", type=int, default=4)
    parser.add_argument("--validation_steps", type=int, default=100)
    parser.add_argument("--validation_epochs", type=int, default=None)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--no_safe_serialization", action="store_true")
    parser.add_argument("--feature_dropout_prob", type=float, default=0.5)
    parser.add_argument("--lambda_attn", type=float, default=0.1)

    args = parser.parse_args()
    if args.train_data_dir is None:
        raise ValueError("You must specify a train data directory.")
    return args


class CrossAttnCaptureProcessor:
    def __init__(
            self,
            attn_maps,
            layer_name="unknown",
            capture=True,
            capture_spatial_sizes=None,
    ):
        self.attn_maps = attn_maps
        self.layer_name = layer_name
        self.capture = capture
        self.capture_spatial_sizes = capture_spatial_sizes

    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
            *args,
            **kwargs,
    ):
        residual = hidden_states

        # 1. Spatial normalization
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size = hidden_states.shape[0]

        is_cross = encoder_hidden_states is not None

        # 2. Cross-Attention encoder states 归一化
        if is_cross:
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        else:
            encoder_hidden_states = hidden_states

        query_len = hidden_states.shape[1]

        # -------------------------------------------------------------------
        # 判断当前层是否需要捕获 Cross-Attention Map
        # -------------------------------------------------------------------
        should_capture = (
                is_cross
                and self.capture
                and (self.capture_spatial_sizes is None or query_len in self.capture_spatial_sizes)
        )

        # 3. Q / K / V 投影
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        heads = attn.heads
        head_dim = query.shape[-1] // heads

        # 统一转为原生 4D 形状: [B, heads, seq_len, head_dim]
        query = query.view(batch_size, -1, heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, heads, head_dim).transpose(1, 2)

        # -------------------------------------------------------------------
        # 分支 A (极速通路): 不需要捕获的层，直接走 PyTorch 2.0 SDPA 加速
        # -------------------------------------------------------------------
        if not should_capture:
            # 使用 PyTorch 2.0+ FlashAttention / Memory-Efficient Attention 算子
            hidden_states = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False
            )
            # 还原维度: [B, heads, seq_len, head_dim] -> [B, seq_len, inner_dim]
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, heads * head_dim)

        # -------------------------------------------------------------------
        # 分支 B (捕获通路): 只有目标 Cross-Attention 层才手动计算 Softmax
        # -------------------------------------------------------------------
        else:
            scale = attn.scale if hasattr(attn, "scale") and attn.scale is not None else (1.0 / (head_dim ** 0.5))

            # 计算点积矩阵: [B, heads, query_len, key_len]
            attention_scores = torch.matmul(query, key.transpose(-1, -2)) * scale

            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask

            # 【防溢出保护】在 float32 下计算 Softmax，防止 FP16 训练时出现 NaN/Inf
            attention_probs = F.softmax(attention_scores, dim=-1, dtype=torch.float32).to(query.dtype)

            # 捕获 4D Attention Map [B, heads, query_len, key_len]
            #切断 loss_attn 到 UNet (Q/K) 的梯度
            self.attn_maps.append(attention_probs)

            # 计算加权输出
            hidden_states = torch.matmul(attention_probs, value)
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, heads * head_dim)

        # 4. 后续输出投影与 Rescale
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

def main():
    args = parse_args()

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError("Cannot use both --report_to=wandb and --hub_token.")
    if args.report_to == "wandb" and not is_wandb_available():
        raise ImportError("Install wandb to use it for logging.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    os.makedirs(args.output_dir, exist_ok=True)
    repo_id = None
    if args.push_to_hub:
        repo_id = create_repo(
            repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
        ).repo_id

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Load tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    else:
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")

    # Load models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder",
                                                 revision=args.revision)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision,
                                        variant=args.variant)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet",
                                                revision=args.revision, variant=args.variant)
    ### 冻结vae和unet
    vae.requires_grad_(False)
    # unet.requires_grad_(False)

    ### 预训练模型
    if args.dataset_name.lower()=='sysu':
        num_person=395
        num_label=296
        deen_model_path="./DEEN/sysu_deen_p4_n6_lr_0.1_seed_0_best.t"
        deen=embed_net(num_person,dataset=args.dataset_name,arch='resnet50')
    elif args.dataset_name.lower()=='llcm':
        num_person=713
        num_label = 713
        deen_model_path = "./DEEN/llcm_deen_p4_n6_lr_0.1_seed_0_best.t"
        deen = embed_net(num_person, dataset=args.dataset_name, arch='resnet50')

    if os.path.exists(deen_model_path):
        deen.load_state_dict(torch.load(deen_model_path, map_location="cpu")['net'])
        logger.info(f"Loaded DEEN weights from {deen_model_path}")
    else:
        logger.warning(f"DEEN weight file not found at {deen_model_path}, using uninitialized model.")
    deen.eval()
    deen.requires_grad_(False)
    deen.to(device, dtype=weight_dtype)
    ### 行人特征维度
    deen_feat_dim = 2048
    deen.training = False
    clip_dim = text_encoder.config.hidden_size


    # 建立特征投影网络 (MLP)，负责将 DEEN 特征对齐到 CLIP Text 空间
    proj_layer = nn.Sequential(
        nn.Linear(deen_feat_dim, clip_dim),
        nn.LayerNorm(clip_dim),
        nn.GELU(),
        nn.Linear(clip_dim, clip_dim)
    ).to(device, dtype=weight_dtype)

    config_file = './AGW/AGW_baseline.yml'
    weight_path = './AGW/market1501_AGW.pth'
    agw_extractor = MarketFeatureExtractor(config_file, weight_path)
    logger.info(f"Loaded AGW Market model from {weight_path}")
    agw_feat_dim = 2048
    agw_proj_layer = nn.Sequential(
        nn.Linear(agw_feat_dim, clip_dim),
        nn.LayerNorm(clip_dim),
        nn.GELU(),
        nn.Linear(clip_dim, clip_dim)
    ).to(device, dtype=weight_dtype)

    placeholder_tokens = []
    placeholder_tokens.append(args.placeholder_token[0].format('IR'))
    placeholder_tokens.append(args.placeholder_token[0].format('RGB'))
    ### placeholder_tokens=[‘<style_IR>’,'<style_RGB>']

    if args.dataset_name.lower()=='sysu':
        ### 遍历所有行人，并为每个行人创建一组token ['<person_{}>']
        id_file = os.path.join(args.train_data_dir, 'exp', 'train_id.txt')
        with open(id_file, 'r') as f:
            content = f.read()
            raw_ids = content.replace('\n', ',').split(',')
        ids = [i.zfill(4) for i in raw_ids if i.strip()]
        ### 将标签转化为连续标签
        set_ids = sorted(list(set([int(i) for i in ids])))
    elif args.dataset_name.lower()=='llcm':
        idx_files=[
            os.path.join(args.train_data_dir, 'idx', 'train_vis.txt'),
            os.path.join(args.train_data_dir, 'idx', 'train_nir.txt')
        ]
        raw_labels=[]
        for idx_file in idx_files:
            with open(idx_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 2:
                            raw_labels.append(int(parts[1]))
        set_ids = sorted(list(set(raw_labels)))

    id_to_label = {pid: i for i, pid in enumerate(set_ids)}
    label_to_id = {i: pid for i, pid in enumerate(set_ids)}

    for id in set_ids:
        placeholder_tokens.append(args.placeholder_token[1].format(id_to_label[id]))
        # add dummy tokens for multi-vector
        for i in range(1, args.num_vectors + 1):
            placeholder_tokens.append(
                f"<person_{id_to_label[id]}_{i}>")  ### placeholder_token=[‘‘<style_IR>’,'<style_RGB>','<cam_1>',...,'<person_{0}>','<person_{0}_1>','<person_{0}_2>','<person_{0}_3>','<person_{1}',...]
    if args.num_vectors < 1:
        raise ValueError(f"--num_vectors has to be >= 1, but is {args.num_vectors}")

    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)

    logger.info(f'Added {num_added_tokens} tokens')

    initializer_token_ids = []
    for token in args.initializer_token:
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if len(token_ids) > 1:
            raise ValueError("The initializer token must be a single token.")
        initializer_token_ids.append(token_ids[0])
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(
        placeholder_tokens)  ### placeholder_token_ids=[49408, 49409, 49410, 49411, 49412,....]

    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data
    ### 初始化新增token
    with torch.no_grad():
        token_embeds[placeholder_token_ids[0]] = token_embeds[initializer_token_ids[0]].clone()
        token_embeds[placeholder_token_ids[1]] = token_embeds[initializer_token_ids[0]].clone()
        current_idx = 2

        part_embeds = [
            token_embeds[initializer_token_ids[1]].clone(),# person 全局
            token_embeds[initializer_token_ids[2]].clone(),  # head
            token_embeds[initializer_token_ids[3]].clone(),  # body
            token_embeds[initializer_token_ids[4]].clone()  # leg
        ]
        part_idx=0
        for ids in placeholder_token_ids[current_idx:]:
            token_embeds[ids]=part_embeds[part_idx%len(part_embeds)].clone()
            part_idx+=1

    # Freeze everything except new token embeddings
    text_module = text_encoder.text_model if hasattr(text_encoder, "text_model") else text_encoder
    text_module.encoder.requires_grad_(False)
    text_module.final_layer_norm.requires_grad_(False)
    text_module.embeddings.position_embedding.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.train()
        text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning("xFormers 0.0.16 may cause problems during training. Please update to at least 0.0.17.")
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # ==========================================================
    # Cross-Attention Map Capture
    # ==========================================================
    attn_maps = []
    original_attn_processors = unet.attn_processors
    new_attn_processors = {}
    for name, processor in original_attn_processors.items():
        if "attn2" in name:
            new_attn_processors[name] = CrossAttnCaptureProcessor(
                attn_maps=attn_maps,
                layer_name=name,
                capture=True,
                capture_spatial_sizes={2048},
            )
        else:
            new_attn_processors[name] = processor
    unet.set_attn_processor(
        new_attn_processors
    )

    # Move to device
    unet.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device,dtype=weight_dtype)

    if args.scale_lr:
        args.learning_rate = args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size

    optimizer = torch.optim.AdamW(
        [
            {'params': unet.parameters(), "lr": args.lora_lr},
            {'params': text_encoder.get_input_embeddings().parameters(), "lr": args.ti_lr},
            {'params':proj_layer.parameters(),'lr':args.lora_lr},
            {'params':agw_proj_layer.parameters(),'lr':args.lora_lr},
        ],
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset
    train_dataset = TextualInversionDataset(
        data_root=args.train_data_dir,
        tokenizer=tokenizer,
        img_h=args.img_h,
        img_w=args.img_w,
        num_vector=args.num_vectors,
        repeats=args.repeats,
        center_crop=args.center_crop,
        mode_set="train",
        dataset_name=args.dataset_name
    )
    sampler = IdentitySampler(train_dataset, num_instances_per_batch=args.P, num_instances_per_id=args.K)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=BatchSampler(sampler, batch_size=args.P * args.K, drop_last=True),
        num_workers=args.dataloader_num_workers,
        pin_memory=True
    )

    if args.validation_epochs is not None:
        warnings.warn(
            f"FutureWarning: validation_epochs is deprecated. Use validation_steps instead.",
            FutureWarning, stacklevel=2,
        )
        args.validation_steps = args.validation_epochs * len(train_dataset)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
        num_cycles=args.lr_num_cycles,
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size = {args.P*args.K}")
    logger.info(f"  Weight dtype = {args.mixed_precision}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Resume from checkpoint
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = sorted(
                [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")],
                key=lambda x: int(x.split("-")[1]),
            )
            path = dirs[-1] if dirs else None

        if path is None:
            logger.info(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
        else:
            logger.info(f"Resuming from checkpoint {path}")
            checkpoint = torch.load(os.path.join(args.output_dir, path, "training_state.pt"), map_location=device)
            global_step = checkpoint["global_step"]
            text_encoder.get_input_embeddings().weight.data[placeholder_token_ids] = checkpoint["learned_embeds"]
            unet.load_state_dict(checkpoint["unet"])
            if "proj_layer" in checkpoint:
                proj_layer.load_state_dict(checkpoint["proj_layer"])
            if "agw_proj_layer" in checkpoint:
                agw_proj_layer.load_state_dict(checkpoint["agw_proj_layer"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(range(0, args.max_train_steps), initial=global_step, desc="Steps")

    # Keep original embeddings as reference
    orig_embeds_params = text_encoder.get_input_embeddings().weight.data.clone()

    images = []
    lr_decayed=False
    for epoch in range(first_epoch, args.num_train_epochs):
        text_encoder.train()
        unet.train()
        proj_layer.train()
        agw_proj_layer.train()
        accum_steps = 0

        for step, batch in enumerate(train_dataloader):
            # Convert images to latent space
            ### 取出嵌入层，分类损失
            labels = batch["label"].to(device,dtype=torch.long)
            cam_ids = batch["cam_id"].to(device,dtype=torch.long)
            pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)
            gt_masks = batch.get("masks", None)
            if gt_masks is not None:
                gt_masks = gt_masks.to(device, dtype=weight_dtype)

            # 构建 4 个 Token ID 矩阵 [B, 4]
            target_token_ids = []
            for label in labels:
                p_main = f"<person_{label.item()}>"
                p_head = f"<person_{label.item()}_1>"
                p_upper = f"<person_{label.item()}_2>"
                p_lower = f"<person_{label.item()}_3>"
                ids = tokenizer.convert_tokens_to_ids([p_main, p_head, p_upper, p_lower])
                target_token_ids.append(ids)
            target_token_ids = torch.tensor(target_token_ids, device=device, dtype=torch.long)

            bsz = pixel_values.shape[0]

            vae.to(dtype=torch.float32)
            latents = vae.encode(pixel_values.to(dtype=torch.float32)).latent_dist.sample().detach()
            latents = latents * vae.config.scaling_factor
            latents=latents.to(dtype=weight_dtype)

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            attn_maps.clear()
            encoder_hidden_states = text_encoder(batch["input_ids"].to(device))[0].to(dtype=weight_dtype)

            # 判断当前训练阶段：是否超过 70% 的总步数
            is_second_stage = (global_step / args.max_train_steps) >= 0.7

            if is_second_stage and not lr_decayed:
                logger.info(
                    f"Entering stage 2 (Step {global_step}). Decaying learning rate by factor {args.stage2_lr_decay}...")
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= args.stage2_lr_decay
                lr_decayed = True
            if is_second_stage:
                # 筛选出 label < 296 的行人掩码
                deen_mask = labels < num_label  # [B]
                # 初始化特征张量 [B, CLIP_DIM]
                proj_feats = torch.zeros(
                    (bsz, text_encoder.config.hidden_size),
                    device=device,
                    dtype=weight_dtype
                )
                # 仅对 SYSU 数据集 (label < 296) 的样本提取特征
                if deen_mask.any():
                    valid_pixels = pixel_values[deen_mask]
                    valid_cams = cam_ids[deen_mask]

                    # 提取 DEEN 特征并投影到 CLIP 空间
                    deen_feats_valid = extract_deen_features(deen, valid_pixels, valid_cams)
                    proj_feats[deen_mask] = proj_layer(deen_feats_valid)

                agw_mask=labels>=num_label
                if agw_mask.any() and agw_extractor is not None:
                    valid_pixels=pixel_values[agw_mask]
                    pixels_01 = (valid_pixels + 1.0) / 2.0
                    agw_feats_valid = agw_extractor.extract_from_tensor(pixels_01).to(dtype=weight_dtype)
                    proj_feats[agw_mask] = agw_proj_layer(agw_feats_valid)

                feature_mask = (torch.rand(bsz, 1, 1, device=device) > args.feature_dropout_prob).to(dtype=weight_dtype)

                proj_feats_expanded=proj_feats.unsqueeze(1)  #[B,1,768]
                masked_proj_feats=proj_feats_expanded*feature_mask

                encoder_hidden_states=torch.cat([encoder_hidden_states, masked_proj_feats], dim=1)

            # Unet前向预测
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            loss_diff = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            if is_second_stage:
                loss_attn = args.lambda_attn * compute_cross_attention_mask_loss(
                        attn_maps=attn_maps,
                        input_ids=batch["input_ids"].to(device),
                        target_masks=gt_masks,
                        target_token_ids=target_token_ids
                )
                loss = loss_diff + loss_attn
            else:
                loss = loss_diff
            # Gradient accumulation
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            accum_steps += 1

            if accum_steps == args.gradient_accumulation_steps:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                accum_steps = 0

                # Restore original non-placeholder embeddings
                index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
                index_no_updates[placeholder_token_ids] = False
                with torch.no_grad():
                    text_encoder.get_input_embeddings().weight[index_no_updates] = orig_embeds_params[index_no_updates]

                progress_bar.update(1)
                global_step += 1
                progress_bar.set_postfix(
                    stage="2nd(30%)" if is_second_stage else "1st(70%)",
                    loss=loss.item() * args.gradient_accumulation_steps,
                    loss_diff=loss_diff.item(),
                )
                # Save full checkpoint
                if global_step % args.checkpointing_steps == 0:
                    if args.checkpoints_total_limit is not None:
                        checkpoints = sorted(
                            [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")],
                            key=lambda x: int(x.split("-")[1]),
                        )
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            for ckpt in checkpoints[:len(checkpoints) - args.checkpoints_total_limit + 1]:
                                shutil.rmtree(os.path.join(args.output_dir, ckpt))

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)
                    torch.save({
                        "global_step": global_step,
                        ### TI
                        "learned_embeds": text_encoder.get_input_embeddings().weight.data[
                            placeholder_token_ids].clone(),
                        ### LoRA
                        "unet": unet.state_dict(),
                        "proj_layer":proj_layer.state_dict(),
                        "agw_proj_layer":agw_proj_layer.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    }, os.path.join(save_path, "training_state.pt"))

                # Validation
                if args.validation_true is not None and global_step % args.validation_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)

                    ### 保存当前step独立的TI词嵌入
                    ckpt_weight_name = "learned_embeds.bin" if args.no_safe_serialization else "learned_embeds.safetensors"
                    save_progress(
                        text_encoder,
                        placeholder_token_ids,
                        placeholder_tokens,
                        args,
                        os.path.join(save_path, ckpt_weight_name),
                        safe_serialization=not args.no_safe_serialization
                    )

                    ### 保存当前step独立的LoRA权重文件夹
                    unet.save_pretrained(os.path.join(save_path, 'unet'))
                    logger.info(f"Saved checkpoint to {save_path}")
                    images = log_validation(args, label_to_id,device, weight_dtype, global_step)

            if global_step >= args.max_train_steps:
                break

    logger.info("Training finished...")


if __name__ == "__main__":
    main()