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
    def __init__(self, config, model, train_loader, val_loader, device=None):
        logging.info(f"Starting training...")

        # 设备
        self.device = device if device else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = model.to(self.device)

        # 数据加载
        self.train_loader = train_loader
        self.val_loader = val_loader
        print(f"训练集样本数量: {len(train_loader.dataset)}")
        print(f"验证集样本数量: {len(val_loader.dataset)}")
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
        self.start_epoch = 1
        self.start_step = 0

        # 尝试加载 checkpoint
        self._load_checkpoint()

    def _save_checkpoint(self, epoch, step, best=False):
        """保存 checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_accuracy': self.best_val_accuracy
        }

        filename = "best_model.pth" if best else f"checkpoint_{step}.pth"
        save_path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, save_path)
        logging.info(f"✅ Checkpoint saved: {save_path}")

    def _load_checkpoint(self):
        """尝试加载已保存的 checkpoint"""
        best_model_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        last_checkpoint_path = os.path.join(self.checkpoint_dir, "last_checkpoint.pth")

        if os.path.exists(last_checkpoint_path):
            checkpoint_path = last_checkpoint_path
        elif os.path.exists(best_model_path):
            checkpoint_path = best_model_path
        else:
            logging.info("⚠️ No checkpoint found. Starting fresh!")
            return

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.start_epoch = checkpoint.get('epoch', 1)
        self.start_step = checkpoint.get('step', 0)
        self.best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0)

        logging.info(f"✅ Loaded checkpoint from {checkpoint_path}. Resuming from epoch {self.start_epoch}, step {self.start_step}")

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

        step = self.start_step
        train_loss, train_accuracy = 0.0, 0.0
        for epoch in range(self.start_epoch, self.epoch_steps + 1):
            
            for features, labels in self.train_loader:
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
                    logging.info(f"Epoch {epoch}, Step {step}, "
                                 f"Train Loss: {train_loss / self.log_interval:.4f}, "
                                 f"Train Accuracy: {train_accuracy * 100/ self.log_interval:.2f}%, "
                                 f"Val Loss: {val_loss:.4f}, "
                                 f"Val Accuracy: {val_accuracy*100:.2f}%")

                    # 重置训练损失 & 准确率
                    train_loss, train_accuracy = 0.0, 0.0

                    # 保存最优模型
                    if val_accuracy > self.best_val_accuracy:
                        self.best_val_accuracy = val_accuracy
                        self._save_checkpoint(epoch, step, best=True)

                # 定期保存 checkpoint
                if step % self.ckpt_interval == 0:
                    self._save_checkpoint(epoch, step, best=False)
                    torch.save({
                        'epoch': epoch,
                        'step': step,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'best_val_accuracy': self.best_val_accuracy
                    }, os.path.join(self.checkpoint_dir, "last_checkpoint.pth"))

        # 训练结束保存最终模型
        self._save_checkpoint(self.epoch_steps, step, best=False)
        logging.info("\n================ Finished Training ================")