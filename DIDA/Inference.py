import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, UNet2DConditionModel
from peft import PeftModel
from transformers import CLIPTextModel, CLIPTokenizer
import safetensors.torch


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

    # 将权重填入 text_encoder
    token_embeds = pipeline.text_encoder.get_input_embeddings().weight.data
    for token, embed in state_dict.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        token_embeds[token_id] = embed.to(device=device, dtype=torch.float16)

    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    return pipeline


def generate_sysu_expansion(pipeline, save_root, num_vectors, person_id, cam_id, num_img, label_to_id):
    generate_cam=['cam8']
    # 构建文本
    # 假设训练时使用了 num_vectors，构建对应的 token 字符串
    person_tokens = [f"<person_{person_id}>"]
    for v in range(1, num_vectors + 1):
        person_tokens.append(f"<person_{person_id}_{v}>")
    person_str = " ".join(person_tokens)

    for cam in generate_cam:
        image_style="IR"
        cam_num=int(cam[-1])
        prompt = f"a <style_{image_style}> photo of a {person_str}"

        # 保存路径
        person_dir_name = str(label_to_id[person_id]).zfill(4)
        target_dir = os.path.join(save_root, cam, person_dir_name)
        os.makedirs(target_dir, exist_ok=True)

        print(f"Generating {prompt}...")
        # 生成 4 张
        images = pipeline(prompt, num_inference_steps=25, num_images_per_prompt=num_img, height=512, width=256).images
        for i, img in enumerate(images):
            img.save(os.path.join(target_dir, f"{i}.png"))


def main():
    pipe = load_inference_pipeline("../paper_code/stable-diffusion-v1-5", "./text-inversion-model/checkpoint-15000_1ti2att-cond")
    data_root = "../data/SYSU_Market/"
    ### 遍历所有行人，并为每个行人创建一组token ['<person>']
    id_file = os.path.join(data_root, 'exp', 'train_id.txt')
    with open(id_file, 'r') as f:
        content = f.read()
        raw_ids = content.replace('\n', ',').split(',')
    ids = [i.zfill(4) for i in raw_ids if i.strip()]

    ### 将标签转化为连续标签
    set_ids = sorted(list(set([int(i) for i in ids])))
    id_to_label = {pid: i for i, pid in enumerate(set_ids)}
    label_to_id = {i: pid for i, pid in enumerate(set_ids)}

    cams = ['cam7']
    for id in ids:
        for cam in cams:
            img_dir = os.path.join(data_root, cam, id)
            if not os.path.exists(img_dir):
                continue
            label = id_to_label[int(id)]
            print(f"generate iamge for {os.path.join(data_root, cam, id)}")
            generate_sysu_expansion(pipeline=pipe, save_root=data_root, num_vectors=3, person_id=label, cam_id=cam,
                                    num_img=30, label_to_id=label_to_id)


if __name__ == '__main__':
    main()
