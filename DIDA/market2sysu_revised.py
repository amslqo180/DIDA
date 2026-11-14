#!/usr/bin/env python3

import os
import shutil
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Convert Market-1501 dataset to SYSU-MM01 format and merge into a new dataset")

    # Hyperparameters:
    parser.add_argument(
        "--market_gt_bbox_dir",
        type=str,
        default="/home/lk/PyCharmMiscProject/data/Market-1501-v15.09.15/gt_bbox",
        help="Path of the gt_bbox folder in the Market-1501 dataset"
    )
    parser.add_argument(
        "--sysu_dir",
        type=str,
        default="/home/lk/PyCharmMiscProject/data/SYSU-MM01",
        help="Root directory path of the ORIGINAL SYSU-MM01 dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/lk/PyCharmMiscProject/data/SYSU_Market",
        help="Path where the NEW merged dataset will be created"
    )
    parser.add_argument(
        "--max_sysu_id",
        type=int,
        default=533,
        help="Current maximum SYSU ID (default 533, can be changed dynamically)"
    )
    parser.add_argument(
        "--min_images_per_id",
        type=int,
        default=12,
        help="Threshold for minimum number of images per ID (default 12)"
    )

    args = parser.parse_args()

    market_gt_bbox_dir = args.market_gt_bbox_dir
    sysu_dir = args.sysu_dir
    output_dir = args.output_dir

    # ----------------------------------------------------------------------
    # 核心修改：非破坏性操作。先完整复制原 SYSU 数据集到新目录
    # ----------------------------------------------------------------------
    if not os.path.exists(sysu_dir):
        print(f"Error: Original SYSU directory '{sysu_dir}' does not exist.")
        return

    print(f"Copying original SYSU dataset to new path: {output_dir} ...")
    if os.path.exists(output_dir):
        print(f"Warning: Output directory '{output_dir}' already exists. Overwriting...")
        shutil.rmtree(output_dir)
    shutil.copytree(sysu_dir, output_dir)

    # 后续所有操作均在 output_dir（新数据集）中进行
    new_cam_folder = "cam7"
    new_cam_dir = os.path.join(output_dir, new_cam_folder)
    os.makedirs(new_cam_dir, exist_ok=True)

    max_sysu_id = args.max_sysu_id
    min_images_per_id = args.min_images_per_id

    # 1. Collect all image files corresponding to each old_id
    id_to_images = {}  # key: market_id, value: list of image filenames

    all_files = os.listdir(market_gt_bbox_dir)
    all_files = sorted(all_files)

    for image_file in all_files:
        if image_file[:4].isdigit():
            market_id = int(image_file[:4])
        else:
            print(f"Skipping file that does not meet the format: {image_file}")
            continue

        if market_id not in id_to_images:
            id_to_images[market_id] = []
        id_to_images[market_id].append(image_file)

    # 2. Filter out old_ids that meet min_images_per_id
    selected_market_ids = []
    for market_id, images_list in id_to_images.items():
        if len(images_list) >= min_images_per_id:
            selected_market_ids.append(market_id)

    # 3. Sort selected_market_ids as needed
    selected_market_ids.sort()

    # 4. Assign new IDs and copy files
    id_mapping = {}
    for i, market_id in enumerate(selected_market_ids, start=1):
        new_sysu_id = max_sysu_id + i  # Assign IDs sequentially
        id_mapping[market_id] = new_sysu_id

        # Create a new folder for this ID
        new_id_dir = os.path.join(new_cam_dir, f"{new_sysu_id:04d}")
        os.makedirs(new_id_dir, exist_ok=True)

        # Copy all images of this market_id and rename them in order
        images_list = id_to_images[market_id]
        images_list.sort()
        for idx, image_file in enumerate(images_list, start=1):
            old_image_path = os.path.join(market_gt_bbox_dir, image_file)
            new_image_path = os.path.join(new_id_dir, f"{idx:04d}.jpg")
            shutil.copy(old_image_path, new_image_path)

    # 5. Update train_id.txt (在新生成的 output_dir 文件夹内修改)
    train_id_file = os.path.join(output_dir, "exp", "train_id.txt")
    if os.path.exists(train_id_file):
        with open(train_id_file, "r") as f:
            train_ids = f.read().strip().split(",")
    else:
        train_ids = []

    # Append the newly added IDs to the train_id list
    new_train_ids = train_ids + [str(max_sysu_id + i) for i in range(1, len(selected_market_ids) + 1)]

    with open(train_id_file, "w") as f:
        f.write(",".join(new_train_ids))

    print(f"Success! {len(selected_market_ids)} Market-1501 IDs have been merged.")
    print(f"Original dataset '{sysu_dir}' remains untouched.")
    print(f"New dataset with cam7 and updated train_id.txt is saved at: {output_dir}")


if __name__ == "__main__":
    main()