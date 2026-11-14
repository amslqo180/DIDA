import random

import torch
import numpy as np
from packaging import version
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import PIL
from PIL import Image

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }

image_text_templates=[
    "a {} photo of a {}",
    "a {} rendering of a {}",
    "a {} cropped photo of the {}",
    "the {} photo of a {}",
    "a {} photo of a clean {}",
    "a {} photo of a dirty {}",
    "a {} dark photo of the {}",
    "a {} photo of my {}",
    "a {} photo of the cool {}",
    "a {} close-up photo of a {}",
    "a {} bright photo of the {}",
    "a {} cropped photo of a {}",
    "a {} photo of the {}",
    "a {} good photo of the {}",
    "a {} photo of one {}",
    "a {} close-up photo of the {}",
    "a {} rendition of the {}",
    "a {} photo of the clean {}",
    "a {} rendition of a {}",
    "a {} photo of a nice {}",
    "a {} good photo of a {}",
    "a {} photo of the nice {}",
    "a {} photo of the small {}",
    "a {} photo of the weird {}",
    "a {} photo of the large {}",
    "a {} photo of a cool {}",
    "a {} photo of a small {}",
]

class TextualInversionDataset(Dataset):
    def __init__(self, data_root,tokenizer,img_h,img_w,repeats=100,interpolation='bicubic',flip_p=0.5,
                 mode_set='train',num_vector=3,center_crop=False,dataset_name='sysu',mask_root=None):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.img_h = img_h
        self.img_w = img_w
        self.num_vector = num_vector
        self.center_crop = center_crop
        self.flip_p = flip_p
        self.dataset_name=dataset_name.lower()
        self.images=[]    ### [[图片路径，label,id,cam],...]
        self.all_id=[]
        if mask_root is not None:
            self.mask_root = mask_root
        else:
            if self.dataset_name == 'sysu':
                self.mask_root = os.path.join(os.path.dirname(data_root.rstrip('/')), "SYSU_Market_Masks")
            else:
                self.mask_root = os.path.join(os.path.dirname(data_root.rstrip('/')), "LLCM_Market_Masks")

        if self.dataset_name == 'sysu':
            ### 读取训练集id
            id_file=os.path.join(data_root,'exp',f'{mode_set}_id.txt')
            with open(id_file,'r') as f:
                content = f.read()
                raw_ids=content.replace('\n',',').split(',')
            ids=[i.zfill(4)for i in raw_ids if i.strip()]

            ### 将标签转化为连续标签
            set_ids=sorted(list(set([int(i) for i in ids])))
            self.all_id=set_ids
            self.id_to_label={pid:i for i,pid in enumerate(set_ids)}
            self.label_to_id={i:pid for i,pid in enumerate(set_ids)}

            ### 读取图片
            self.cams=['cam1','cam2','cam3','cam4','cam5','cam6','cam7']
            for id in ids:
                for cam in self.cams:
                    img_dir=os.path.join(data_root,cam,id)
                    if not os.path.exists(img_dir):
                        continue
                    cam_id=int(cam.replace('cam',''))
                    is_ir=(cam_id in [3,6])
                    for img in os.listdir(img_dir):
                        img_path=os.path.join(img_dir,img)
                        label=self.id_to_label[int(id)]
                        self.images.append([img_path,label,id,cam_id,is_ir])

        elif self.dataset_name == 'llcm':
            idx_files=[os.path.join(data_root,'idx',f'{mode_set}_vis.txt'),
                       os.path.join(data_root,'idx',f'{mode_set}_nir.txt')]
            raw_entries=[]
            for idx_files in idx_files:
                if os.path.exists(idx_files):
                    with open(idx_files,'r') as f:
                        for line in f:
                            line=line.strip()
                            if line:
                                parts=line.split()
                                if len(parts)>=2:
                                    rel_path,label_str=parts[0],parts[1]
                                    raw_entries.append((rel_path,int(label_str)))

            raw_labels=[entry[1]for entry in raw_entries]
            set_ids=sorted(list(set(raw_labels)))
            self.all_id=set_ids
            self.id_to_label={pid:i for i,pid in enumerate(set_ids)}
            self.label_to_id={i:pid for i,pid in enumerate(set_ids)}

            for rel_path, raw_label in raw_entries:
                img_path = os.path.join(data_root, rel_path)
                if not os.path.exists(img_path):
                    continue

                label = self.id_to_label[raw_label]

                # 提取 PID (如 vis/0250/xxx.jpg 中的 '0250')
                path_parts = rel_path.split('/')
                pid = path_parts[1] if len(path_parts) > 1 else str(raw_label)

                # 解析相机 ID
                filename = os.path.basename(rel_path)
                cam_id = 0
                if '_c' in filename:
                    try:
                        cam_id = int(filename.split('_c')[1].split('_')[0])
                    except ValueError:
                        cam_id = 0

                # 模态识别：包含 'nir' 或是 'test_nir' 即为红外
                is_ir = ('nir' in rel_path.lower())

                self.images.append([img_path, label, pid, cam_id, is_ir])
        else:
            raise ValueError(f"不支持的 dataset_name: {dataset_name}，请指定 'sysu' 或 'llcm'")

        self.num_images = len(self.images)
        self._length=self.num_images

        if mode_set=="train":
            self._length=self.num_images*repeats

        ### 建立连续标签（label）到样本索引（indices）的映射
        self.label_to_indices={}
        for idx,(_,label,_,_,_) in enumerate(self.images):
            if label not in self.label_to_indices:
                self.label_to_indices[label]=[]
            self.label_to_indices[label].append(idx)

        self.unique_labels=list(self.label_to_indices.keys())

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        self.template = image_text_templates
        self.flip_transform=transforms.RandomHorizontalFlip(p=self.flip_p)


    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        example = {}
        idx=idx%self.num_images
        img_path,label,pid,cam_id,is_ir=self.images[idx]
        image=Image.open(img_path)

        if not image.mode=="RGB":
            image = image.convert("RGB")

        rel_path=os.path.relpath(img_path,self.data_root)
        mask_path = os.path.join(self.mask_root, "masks", os.path.splitext(rel_path)[0] + ".png")

        if os.path.exists(mask_path):
            mask_label=Image.open(mask_path)
        else:
            mask_label = Image.fromarray(np.zeros((image.height, image.width), dtype=np.uint8))

        if self.center_crop:
            crop = min(image.height, image.width)
            h, w = image.height, image.width
            image = image.crop(((w - crop) // 2, (h - crop) // 2, (w + crop) // 2, (h + crop) // 2))
            mask_label = mask_label.crop(((w - crop) // 2, (h - crop) // 2, (w + crop) // 2, (h + crop) // 2))

        image = image.resize((self.img_w, self.img_h), resample=self.interpolation)
        mask_label = mask_label.resize((self.img_w, self.img_h), resample=PIL_INTERPOLATION["nearest"])

        # 随机水平翻转 (图片与掩码必须保持同步翻转)
        if random.random() < self.flip_p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask_label = mask_label.transpose(Image.FLIP_LEFT_RIGHT)


        # --- 3. 构造 4 个二值掩码 (Full, Head, Upper, Lower) ---
        mask_np = np.array(mask_label)
        m_head = (mask_np == 1).astype(np.float32)
        m_upper = (mask_np == 2).astype(np.float32)
        m_lower = (mask_np == 3).astype(np.float32)
        m_full = ((mask_np > 0)).astype(np.float32)  # 行人整体掩码

        # 组合成 4 通道张量 [4, H, W]
        masks = np.stack([m_full, m_head, m_upper, m_lower], axis=0)
        example["masks"] = torch.from_numpy(masks)  # 维度: [4, img_h, img_w]

        style_token = "<style_IR>" if is_ir else "<style_RGB>"
        person_tokens=[f"<person_{label}>"]
        for i in range(1, self.num_vector + 1):
            person_tokens.append(
                f"<person_{label}_{i}>"
            )
        person_string = " ".join(person_tokens)
        text=random.choice(image_text_templates)
        text=text.format(style_token,person_string)
        # logging.info(f"文本为{text}")
        ### tokenizer处理文本，转化成数字ID
        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        # logging.info(f"序列为{example['input_ids']}")
        img_np=np.array(image).astype(np.uint8)
        img_np = (img_np / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(img_np).permute(2,0,1)
        example['label']=self.images[idx][1]
        example['pid']=self.images[idx][2]
        example['cam_id']=self.images[idx][3]
        return example


