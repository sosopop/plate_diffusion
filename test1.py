import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

class EfficientAttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4, reduction_factor=2):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.reduction_factor = reduction_factor
        
        self.norm = nn.GroupNorm(1, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        reduced_h, reduced_w = h // self.reduction_factor, w // self.reduction_factor
        
        x = self.norm(x)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reduce spatial dimensions of k and v
        k = F.adaptive_avg_pool2d(k, (reduced_h, reduced_w))
        v = F.adaptive_avg_pool2d(v, (reduced_h, reduced_w))
        
        q = einops.rearrange(q, 'b (h c) x y -> b h c (x y)', h=self.num_heads)
        k = einops.rearrange(k, 'b (h c) x y -> b h c (x y)', h=self.num_heads)
        v = einops.rearrange(v, 'b (h c) x y -> b h c (x y)', h=self.num_heads)

        attn = torch.einsum('bhci,bhcj->bhij', q, k) * (self.channels ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        out = torch.einsum('bhij,bhcj->bhci', attn, v)
        out = einops.rearrange(out, 'b h c (x y) -> b (h c) x y', x=h, y=w)
        
        return self.proj(out) + x

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = EfficientAttentionBlock(128)
        self.down2 = Down(128, 256)
        self.sa2 = EfficientAttentionBlock(256)
        self.down3 = Down(256, 256)
        self.sa3 = EfficientAttentionBlock(256)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = EfficientAttentionBlock(128)
        self.up2 = Up(256, 64)
        self.sa5 = EfficientAttentionBlock(64)
        self.up3 = Up(128, 64)
        self.sa6 = EfficientAttentionBlock(64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output

class DiffusionModel(nn.Module):
    def __init__(self, img_size=128, img_channels=3, device="cuda"):
        super().__init__()
        self.img_size = img_size
        self.img_channels = img_channels
        self.device = device

        self.model = UNet(c_in=img_channels, c_out=img_channels, device=device)

    def forward(self, x, t):
        return self.model(x, t)

# 使用示例
device = "cuda" if torch.cuda.is_available() else "cpu"
model = DiffusionModel(img_size=128, img_channels=3, device=device).to(device)

# 假设的输入
batch_size = 4
x = torch.randn(batch_size, 3, 128, 128).to(device)
t = torch.randint(0, 1000, (batch_size,)).to(device)

# 前向传播
output = model(x, t)
print(output.shape)  # 应该输出 torch.Size([4, 3, 128, 128])