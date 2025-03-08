import os
import datetime
import logging
if not logging.getLogger().hasHandlers():  # 避免重复初始化
    logging.basicConfig(level=logging.INFO)

logging.basicConfig(level=logging.INFO)
import torch
import torch.nn as nn
from tqdm import tqdm

class Trainer:
    def __init__(self, config, model, train_loader, val_loader, device="mps" if torch.cuda.is_available() else "cpu"):
        logging.info(f"Starting training...")

        # 设备
        self.device = device
        self.model = model.to(self.device)

        # 数据加载
        self.train_loader = train_loader
        self.val_loader = val_loader

        # 训练参数
        self.total_steps = config.total_steps
        self.epoch_steps = config.n_epochs
        self.log_interval = config.log_interval
        self.ckpt_interval = config.ckpt_interval
        self.loss_weight = config.loss_weight
        self.acc_weight = config.acc_weight

        # 优化器 & 损失函数
        self.loss_object = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

        # TensorBoard & Logging
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(os.path.dirname(__file__), "logs", current_time)
        logging.info(f"TensorBoard logs will be stored in: {log_dir}")

        # Checkpoint 目录
        self.checkpoint_dir = config.checkpoint_paths
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        logging.info(f"Checkpoints will be stored in: {self.checkpoint_dir}")

        # 记录最佳模型
        self.best_val_accuracy = 0.0

    def train_step(self, features, labels):
        """单步训练"""
        self.model.train()
        features, labels = features.to(self.device), labels.to(self.device)

        self.optimizer.zero_grad()
        predictions = self.model(features)

        loss = self.loss_object(predictions, labels)
        loss.backward()
        self.optimizer.step()

        # 计算准确率
        _, predicted = torch.max(predictions, 1)
        accuracy = (predicted == labels).float().mean().item()

        return loss.item(), accuracy

    def val_step(self, features, labels):
        """单步验证"""
        self.model.eval()
        with torch.no_grad():
            features, labels = features.to(self.device), labels.to(self.device)
            predictions = self.model(features)

            loss = self.loss_object(predictions, labels)
            _, predicted = torch.max(predictions, 1)
            accuracy = (predicted == labels).float().mean().item()

        return loss.item(), accuracy

    def train(self):
        """完整训练循环"""
        logging.info("\n================ Starting Training ================")

        step = 0
        for epoch in range(1, self.epoch_steps + 1):
            train_loss, train_accuracy = 0.0, 0.0

            # 遍历训练集
            for step, (features, labels) in enumerate(tqdm(self.train_loader, total=self.total_steps), 1):
                step += 1
                loss, accuracy = self.train_step(features, labels)
                train_loss += loss
                train_accuracy += accuracy

                # 日志记录
                if step % self.log_interval == 0:
                    val_loss, val_accuracy = 0.0, 0.0
                    for val_features, val_labels in self.val_loader:
                        v_loss, v_accuracy = self.val_step(val_features, val_labels)
                        val_loss += v_loss
                        val_accuracy += v_accuracy

                    val_loss /= len(self.val_loader)
                    val_accuracy /= len(self.val_loader)

                    logging.info(f"Epoch {epoch}, Step {step}, Train Loss: {train_loss/self.log_interval:.4f}, Train Accuracy: {train_accuracy/self.log_interval:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

                    # 重置训练损失 & 准确率
                    train_loss, train_accuracy = 0.0, 0.0

                    # 保存最优模型
                    if val_accuracy > self.best_val_accuracy:
                        self.best_val_accuracy = val_accuracy
                        ckpt_path = os.path.join(self.checkpoint_dir, "best_model.pth")
                        torch.save(self.model.state_dict(), ckpt_path)
                        logging.info(f"Best model saved to {ckpt_path}")

                # 定期保存 Checkpoint
                if step % self.ckpt_interval == 0:
                    ckpt_path = os.path.join(self.checkpoint_dir, f"checkpoint_{step}.pth")
                    torch.save(self.model.state_dict(), ckpt_path)
                    logging.info(f"Checkpoint saved at {ckpt_path}")

        # 训练结束，保存最终模型
        final_ckpt_path = os.path.join(self.checkpoint_dir, "final_model.pth")
        torch.save(self.model.state_dict(), final_ckpt_path)
        logging.info(f"Final model saved to {final_ckpt_path}")

        logging.info("\n================ Finished Training ================")