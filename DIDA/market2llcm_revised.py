#!/usr/bin/env python3

import os
import shutil
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Convert Market-1501 dataset to LLCM format and merge into a new dataset")

    # Hyperparameters:
    parser.add_argument(
        "--market_gt_bbox_dir",
        type=str,
        default="/home/lk/PyCharmMiscProject/data/Market-1501-v15.09.15/gt_bbox",
        help="Path of the gt_bbox folder in the Market-1501 dataset"
    )
    parser.add_argument(
        "--llcm_dir",
        type=str,
        default="/home/lk/PyCharmMiscProject/data/LLCM",
        help="Root directory path of the ORIGINAL LLCM dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/lk/PyCharmMiscProject/data/LLCM_Market",
        help="Path where the NEW merged dataset will be created"
    )
    parser.add_argument(
        "--max_llcm_id",
        type=int,
        default=1151,
        help="Current maximum LLCM ID"
    )
    parser.add_argument(
        "--start_label",
        type=int,
        default=713,
        help="Starting label for newly added Market-1501 IDs"
    )
    parser.add_argument(
        "--min_images_per_id",
        type=int,
        default=12,
        help="Threshold for minimum number of images per ID (default 12)"
    )

    args = parser.parse_args()

    market_gt_bbox_dir = args.market_gt_bbox_dir
    llcm_dir = args.llcm_dir
    output_dir = args.output_dir
    start_label = args.start_label
    max_llcm_id = args.max_llcm_id
    min_images_per_id = args.min_images_per_id

    # ----------------------------------------------------------------------
    # 核心步骤 1：非破坏性操作，完整复制原 LLCM 数据集到新目录
    # ----------------------------------------------------------------------
    if not os.path.exists(llcm_dir):
        print(f"Error: Original LLCM directory '{llcm_dir}' does not exist.")
        return

    print(f"Copying original LLCM dataset to new path: {output_dir} ...")
    if os.path.exists(output_dir):
        print(f"Warning: Output directory '{output_dir}' already exists. Overwriting...")
        shutil.rmtree(output_dir)
    shutil.copytree(llcm_dir, output_dir)

    vis_dir = os.path.join(output_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    # ----------------------------------------------------------------------
    # 核心步骤 2：收集 Market-1501 图片并按行人 ID 分组
    # ----------------------------------------------------------------------
    id_to_images = {}
    all_files = sorted(os.listdir(market_gt_bbox_dir))

    for image_file in all_files:
        if image_file[:4].isdigit():
            market_id = int(image_file[:4])
            if market_id <= 0:  # 过滤 -1 或 0000 噪声 ID
                continue
        else:
            print(f"Skipping file that does not meet the format: {image_file}")
            continue

        if market_id not in id_to_images:
            id_to_images[market_id] = []
        id_to_images[market_id].append(image_file)

    # 筛选满足最小图片张数要求的 ID
    selected_market_ids = sorted([
        m_id for m_id, imgs in id_to_images.items()
        if len(imgs) >= min_images_per_id
    ])

    print("Processing and copying Market-1501 images into LLCM structure...")
    new_idx_entries = []

    # ----------------------------------------------------------------------
    # 核心步骤 3：分配新文件夹 ID 与对应的 Label，复制并重命名图片
    # ----------------------------------------------------------------------
    for i, market_id in enumerate(selected_market_ids, start=1):
        new_id_num = max_llcm_id + i  # 行人文件夹编号，如 1152, 1153...
        current_label = start_label + (i - 1)  # 对应训练 label：713, 714, 715...

        new_id_str = f"{new_id_num:04d}"

        # 在 vis 目录下新建行人 ID 文件夹，形如 vis/1152/
        id_folder_path = os.path.join(vis_dir, new_id_str)
        os.makedirs(id_folder_path, exist_ok=True)

        images_list = sorted(id_to_images[market_id])
        for idx, image_file in enumerate(images_list, start=1):
            old_image_path = os.path.join(market_gt_bbox_dir, image_file)

            new_image_name = f"{new_id_str}_c10_m{market_id:04d}_f{idx:04d}_vis.jpg"
            new_image_path = os.path.join(id_folder_path, new_image_name)
            shutil.copy(old_image_path, new_image_path)

            rel_path = f"vis/{new_id_str}/{new_image_name}"
            # 存入列表：相对路径 + 当前分配的 label
            new_idx_entries.append(f"{rel_path} {current_label}\n")

    # ----------------------------------------------------------------------
    # 核心步骤 4：以追加模式 ('a') 追加到 idx/train_vis.txt
    # ----------------------------------------------------------------------
    target_idx_file = os.path.join(output_dir, "idx", "train_vis.txt")

    # 使用 "a" (append) 模式以确保保留原 txt 中的内容
    with open(target_idx_file, "a") as f:
        f.writelines(new_idx_entries)

    print(f"\n成功合并！共融合了 {len(selected_market_ids)} 个 Market-1501 ID。")
    print(f"Label 范围：{start_label} ~ {start_label + len(selected_market_ids) - 1}")
    print(f"新生成的图片存放在: {vis_dir}")
    print(f"追加后的训练索引已写入: {target_idx_file}")


if __name__ == "__main__":
    main()