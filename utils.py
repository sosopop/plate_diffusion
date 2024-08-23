import numpy as np
import torch
import torch.nn as nn

class NoiseStdAdjuster:
    def __init__(self, initial_noise_figure=0.0, adjust_batch_interval=300):
        self.noise_figure = initial_noise_figure
        self.g_loss_std = 0.0
        self.d_loss_std = 0.0
        self.g_loss_mean = 0.0
        self.d_loss_mean = 0.0
        self.loss_ratio = 0.0
        self.scale = 1.0
        self.generator_loss_history = []
        self.discriminator_loss_history = []
        self.adjust_batch_interval = adjust_batch_interval
        self.current_batch_step = 0

    def update(self, generator_loss, discriminator_loss):
        self.generator_loss_history.append(generator_loss)
        self.discriminator_loss_history.append(discriminator_loss)
        self.current_batch_step += 1
        if self.current_batch_step >= self.adjust_batch_interval:
            self.current_batch_step = 0
            self.adjust_noise_figure()

        return self.noise_figure, self.g_loss_std, self.d_loss_std, self.loss_ratio
    
    def set_batch_interval(self, batch_interval):
        self.adjust_batch_interval = batch_interval
        
    def set_scale_step(self, scale):
        self.scale = scale

    def adjust_noise_figure(self):
        # 计算生成器损失平均值
        self.g_loss_mean = np.mean(self.generator_loss_history)
        # 计算生成器损失标准差
        self.g_loss_std = np.std(self.generator_loss_history)
        # 计算判别器损失平均值
        self.d_loss_mean = np.mean(self.discriminator_loss_history)
        # 计算判别器损失标准差
        self.d_loss_std = np.std(self.discriminator_loss_history)

        # 计算生成器和判别器损失的平均值比例
        self.loss_ratio = self.g_loss_mean / self.d_loss_mean
        if self.loss_ratio > 10.0:
            self.noise_figure = self.noise_figure + 0.1 * self.scale
        elif self.loss_ratio < 2.5:
            self.noise_figure = self.noise_figure - 0.1 * self.scale

        # 清空历史记录
        self.generator_loss_history.clear()
        self.discriminator_loss_history.clear()

        return self.noise_figure

def test_ajust_noise_figure():
    # 使用示例
    adjuster = NoiseStdAdjuster(initial_noise_figure=1.0, adjust_batch_interval=1000)

    # 模拟生成器和判别器损失更新
    for epoch in range(5000):
        generator_loss = np.random.rand() * 5  # 模拟生成器损失
        discriminator_loss = np.random.rand() / 2  # 模拟判别器损失
        
        noise_figure, gen_loss_std, disc_loss_std, loss_ratio = adjuster.update(generator_loss, discriminator_loss)
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}:")
            print(f"  Noise Std: {noise_figure}")
            print(f"  Generator Loss Std: {gen_loss_std}")
            print(f"  Discriminator Loss Std: {disc_loss_std}")
            print(f"  Loss Ratio: {loss_ratio}")


def discriminator_loss_function(d_real, d_fake):
    D_real_loss = nn.functional.binary_cross_entropy(d_real, torch.ones_like(d_real), reduction='sum')
    D_fake_loss = nn.functional.binary_cross_entropy(d_fake, torch.zeros_like(d_fake), reduction='sum')
    return D_real_loss, D_fake_loss

def generator_loss_function(d_fake):
    D_fake_loss = nn.functional.binary_cross_entropy(d_fake, torch.ones_like(d_fake), reduction='sum')
    return D_fake_loss


def test_soft_label_loss_function():
    # 模拟的判别器输出
    d_real = torch.tensor([0.9, 0.85, 0.8])
    d_fake = torch.tensor([0.1, 0.2, 0.15])

    # 计算判别器损失
    d_r_loss, d_f_loss = discriminator_loss_function(d_real, d_fake)
    print(f"Discriminator Loss: {d_r_loss.item() + d_f_loss.item()}")

    # 计算生成器损失
    g_loss = generator_loss_function(d_fake)
    print(f"Generator Loss: {g_loss.item()}")
    g_loss = generator_loss_function(d_fake)
    print(f"Generator Loss with Soft Label: {g_loss.item()}")
    

if __name__ == '__main__':
    test_soft_label_loss_function()
    test_ajust_noise_figure()