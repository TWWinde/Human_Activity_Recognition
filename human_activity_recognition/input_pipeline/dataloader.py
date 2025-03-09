import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import Counter

class HumanActivityDataset(Dataset):
    def __init__(self, data_dir, seq_length=100, stride=50, mode="train"):
        """
        data_dir: 数据存放的目录 (包含 RawData/)
        seq_length: 每个时间序列样本的长度
        stride: 滑动窗口步长
        users: 选取的数据用户 (用于划分训练集和测试集)
        """
        self.seq_length = seq_length
        self.stride = stride
        self.data = []  # 存放 (样本, 标签)
        self.mode = mode
        if self.mode == "train":
            users = set(range(1, 26))
        elif self.mode == "test":
            users = set(range(26, 30))
        else:
            raise ValueError("Invalid mode. Must be 'train' or 'test'.")

        labels_path = os.path.join(data_dir, "RawData", "labels.txt")
        self._parse_labels(labels_path, data_dir, users)

    def _parse_labels(self, labels_path, data_dir, users):
        """
        解析 labels.txt 并加载数据
        """
        with open(labels_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                
                exp_id, user_id, activity, start, end = map(int, parts)

                # 仅加载指定用户的数据
                if users is not None and user_id not in users:
                    continue

                acc_path = os.path.join(data_dir, "RawData", f"acc_exp{exp_id:02d}_user{user_id:02d}.txt")
                gyro_path = os.path.join(data_dir, "RawData", f"gyro_exp{exp_id:02d}_user{user_id:02d}.txt")

                if os.path.exists(acc_path) and os.path.exists(gyro_path):
                    acc_data = np.loadtxt(acc_path)  # shape: (num_samples, 3)
                    gyro_data = np.loadtxt(gyro_path)  # shape: (num_samples, 3)

                    # 确保数据对齐
                    min_length = min(len(acc_data), len(gyro_data))
                    acc_data, gyro_data = acc_data[:min_length], gyro_data[:min_length]

                    # 拼接成 6D 数据 (加速度X,Y,Z + 角速度X,Y,Z)
                    full_data = np.hstack((acc_data, gyro_data))  # shape: (num_samples, 6)

                    # 提取对应活动的时间段
                    activity_data = full_data[start:end]

                    # 滑动窗口切割数据
                    for i in range(0, len(activity_data) - self.seq_length, self.stride):
                        sample = activity_data[i:i + self.seq_length]  # 形状: (seq_length, 6)
                        self.data.append((sample, activity-1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        features, label = self.data[index]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def get_dataloaders(data_dir, batch_size=64, seq_length=100, stride=50):
    """
    加载训练集和测试集数据，进行数据划分，并对少数类别进行过采样。
    """

    train_dataset = HumanActivityDataset(data_dir, seq_length, stride, mode="train")
    test_dataset = HumanActivityDataset(data_dir, seq_length, stride, mode="test")

    label_counts = Counter([int(label) for _, label in train_dataset])  # 确保 label 是 int 类型

    # 计算类别权重 (出现少的类别赋予更高的采样权重)
    total_samples = sum(label_counts.values())
    class_weights = {label: total_samples / count for label, count in label_counts.items()}

    # 为每个样本分配权重
    sample_weights = [class_weights[int(label)] for _, label in train_dataset]  # 确保 lookup 也用 int

    # 过采样训练数据
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader