import torch
import torch.nn as nn

class NoisePredictionUNet(nn.Module):
    def __init__(self):
        super(NoisePredictionUNet, self).__init__()

        # 使用 BigGAN 残差块构建 UNet
        self.down_blocks = nn.ModuleList([
            BigGANResidualBlock(3, 64),  # 初始通道数为 3 (图像通道数)
            BigGANResidualBlock(64, 128),
            BigGANResidualBlock(128, 256),
        ])
        self.up_blocks = nn.ModuleList([
            BigGANResidualBlock(256 + 128, 128),  # 上采样时与下采样特征拼接
            BigGANResidualBlock(128 + 64, 64),
            BigGANResidualBlock(64 + 3, 3),  # 最后输出通道数为 3 (噪声通道数)
        ])

        # 加入自注意力机制增强模型对图像全局信息的捕捉能力
        self.self_attention = SelfAttention(256)

        # 上采样层
        self.up_samples = nn.ModuleList([
            nn.Upsample(scale_factor=2),
            nn.Upsample(scale_factor=2),
        ])

    def forward(self, x, t):
        # t 是时间步编码，可以是位置编码或其他编码方式

        # 下采样
        skip_connections = []
        for down_block in self.down_blocks:
            x = down_block(x, t)
            skip_connections.append(x)
            x = nn.AvgPool2d(2)(x)

        # 自注意力
        x = self.self_attention(x)

        # 上采样
        for up_block, skip_connection, up_sample in zip(self.up_blocks, reversed(skip_connections), self.up_samples):
            x = up_sample(x)
            x = torch.cat([x, skip_connection], dim=1)  # 与下采样特征拼接
            x = up_block(x, t)

        return x  # 输出预测的噪声


class BigGANResidualBlock(nn.Module):
    # 参考 BigGAN 的残差块设计
    def __init__(self, in_channels, out_channels):
        super(BigGANResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(in_channels)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        # 加入时间步编码 t
        h = self.conv1(nn.functional.relu(self.batch_norm1(x)))
        h += t  # 将 t 广播加到 h 上
        h = self.conv2(nn.functional.relu(self.batch_norm2(h)))
        return h + self.shortcut(x)


class SelfAttention(nn.Module):
    # 自注意力机制
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        v = self.value(x).view(B, -1, H * W)
        attention = torch.softmax(torch.bmm(q, k), dim=-1)
        out = torch.bmm(v, attention.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out + x

model = NoisePredictionUNet()
image = torch.randn(1, 3, 128, 128)  # 示例图像
t = torch.randint(0, 1000, (1,))  # 示例时间步编码
predicted_noise = model(image, t)