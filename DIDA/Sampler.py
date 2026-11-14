from torch.utils.data import Sampler
import random


class IdentitySampler(Sampler):
    """
    P X K Sampler:
    每个 Batch 包含 P 个不同的行人 (num_instances_per_batch)，
    每个行人包含 K 张图片 (num_instances_per_id)。
    Batch Size = P * K
    """

    def __init__(self, dataset, num_instances_per_batch, num_instances_per_id):
        self.dataset = dataset
        self.num_instances_per_batch = num_instances_per_batch  # P (几个人)
        self.num_instances_per_id = num_instances_per_id  # K (每人几张图)
        self.batch_size = num_instances_per_batch * num_instances_per_id

        self.label_to_indices = dataset.label_to_indices
        self.labels = dataset.unique_labels

    def __iter__(self):
        # 每个 epoch 开始时，随机打乱行人顺序
        random.shuffle(self.labels)

        # 复制一份映射字典，方便在抽取时对每个行人的图片序列进行打乱
        pooled_indices = {label: list(indices) for label, indices in self.label_to_indices.items()}
        for label in pooled_indices:
            random.shuffle(pooled_indices[label])

        final_indices = []

        # 循环抽取，直到可抽取的行人不足 P 个
        available_labels = [l for l in self.labels if len(pooled_indices[l]) >= self.num_instances_per_id]

        while len(available_labels) >= self.num_instances_per_batch:
            # 1. 随机选出 P 个行人
            selected_labels = random.sample(available_labels, self.num_instances_per_batch)

            # 2. 从这 P 个行人中，每人吐出 K 张图的索引
            for label in selected_labels:
                for _ in range(self.num_instances_per_id):
                    final_indices.append(pooled_indices[label].pop(0))

            # 3. 动态更新哪些行人还有足够的图片可供下一次抽取
            available_labels = [l for l in available_labels if len(pooled_indices[l]) >= self.num_instances_per_id]

        return iter(final_indices)

    def __len__(self):
        # 粗略估计一个 epoch 产生的样本总数（P * K 的倍数）
        # 由于每次舍弃尾部不够分的样本，实际长度会比图片总数略小
        num_batches = len(self.dataset.images) // self.batch_size
        return num_batches * self.batch_size