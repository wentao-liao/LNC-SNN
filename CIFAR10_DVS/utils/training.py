import torch
import torch.nn.functional as F
from torch.cuda import amp
from spikingjelly.clock_driven import functional, encoding


class Trainer:
    def __init__(self, model, optimizer, device, T=50, amp_enabled=False):
        """
        初始化训练器

        Args:
            model: 神经网络模型
            optimizer: 优化器
            device: 训练设备（CPU/GPU）
            T: 时间步长
            amp_enabled: 是否启用自动混合精度训练
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.T = T
        self.scaler = amp.GradScaler() if amp_enabled else None
        self.encoder = encoding.PoissonEncoder()

    def train_step(self, batch):
        """单个训练步骤"""
        img, label = batch
        img = img.to(self.device)
        label = label.to(self.device)
        label_onehot = F.one_hot(label, 10).float()
        self.optimizer.zero_grad()

        # 使用自动混合精度训练
        if self.scaler is not None:
            with amp.autocast():
                out_fr = self._forward_pass(img)

                loss = F.mse_loss(out_fr, label_onehot)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            out_fr = self._forward_pass(img)
            loss = F.mse_loss(out_fr, label_onehot)
            loss.backward()
            self.optimizer.step()

        # 计算准确率
        acc = (out_fr.argmax(1) == label).float().mean()

        # 清理显存
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return loss.item(), acc.item()

    def evaluate(self, test_loader):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        total_acc = 0
        total_samples = 0

        with torch.no_grad():
            for img, label in test_loader:
                img = img.to(self.device)
                label = label.to(self.device)
                label_onehot = F.one_hot(label, 10).float()
                out_fr = self._forward_pass(img)
                loss = F.mse_loss(out_fr, label_onehot)
                acc = (out_fr.argmax(1) == label).float().sum()

                batch_size = label.numel()
                total_samples += batch_size
                total_loss += loss.item() * batch_size
                total_acc += acc.item()

        avg_loss = total_loss / total_samples
        avg_acc = total_acc / total_samples

        # 清理显存
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return avg_loss, avg_acc

    def _forward_pass(self, img):
        """前向传播"""
        out_fr = 0.
        for t in range(self.T):
            encoded_img = self.encoder(img)
            out_fr += self.model(encoded_img)

            # 每个时间步后清理显存
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        out_fr = out_fr / self.T

        # 重置网络状态
        functional.reset_net(self.model)
        if hasattr(self.model, 'reset_states'):
            self.model.reset_states()

        return out_fr