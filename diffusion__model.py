import torch
import torch.nn as nn
import torch.nn.functional as F
from license_plate_dataset import LicensePlateVocab

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = x.view(-1, self.channels, x.shape[2] * x.shape[3]).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size[0], size[1])

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
            nn.Linear(
                emb_dim,
                out_channels
            ),
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
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
    
class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, text_dim=64, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = AttentionBlock(128)
        self.down2 = Down(128, 256)
        self.sa2 = AttentionBlock(256)
        self.down3 = Down(256, 256)
        self.sa3 = AttentionBlock(256)

        self.bot1 = DoubleConv(320, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = AttentionBlock(128)
        self.up2 = Up(256, 64)
        self.sa5 = AttentionBlock(64)
        self.up3 = Up(128, 64)
        self.sa6 = AttentionBlock(64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)
        self.emb_text = nn.Embedding(LicensePlateVocab.vocab_size, text_dim, padding_idx=LicensePlateVocab.pad_idx)
        self.pos_encoder = PositionalEncoding(text_dim, 16)
        self.token_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=text_dim, nhead=2),
            num_layers=2
        )

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, token):
        emb_tgt = self.emb_text(token)
        emb_tgt = self.pos_encoder(emb_tgt)
        emb_tgt = self.token_encoder(emb_tgt)

        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)
        B, _, H, W = x4.shape 
        emb_tgt = emb_tgt.mean(dim=1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W) 
        x4 = torch.cat([x4, emb_tgt], dim=1)  # 沿着通道维度连接

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        # x = self.sa6(x)
        output = self.outc(x)
        return output

class DiffusionModel(nn.Module):
    def __init__(self, img_size=128, img_channels=3, num_timesteps=1000, device="cuda"):
        super(DiffusionModel, self).__init__()
        self.device = device
        self.num_timesteps = num_timesteps
        self.beta = torch.linspace(1e-4, 0.02, num_timesteps)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.img_size = img_size
        self.img_channels = img_channels
        self.backbone = UNet(c_in=img_channels, c_out=img_channels, device=device)

    def to(self, device):
        self.backbone.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        
    def q_sample(self, x0, t):
        noise = torch.randn_like(x0)
        return (
            torch.sqrt(self.alpha_bar[t])[:, None, None, None] * x0 +
            torch.sqrt(1 - self.alpha_bar[t])[:, None, None, None] * noise
        ), noise
        
    def p_sample(self, xt, t, label):
        predicted_noise = self.backbone(xt, t / self.num_timesteps, label)
        alpha_t = self.alpha[t][:, None, None, None]
        alpha_bar_t = self.alpha_bar[t][:, None, None, None]
        
        mean = (xt - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_t)
        var = self.beta[t][:, None, None, None]
        
        return mean + torch.sqrt(var) * torch.randn_like(xt)
    
    def sample(self, n_samples, img_shape, labels):
        device = next(self.backbone.parameters()).device
        xt = torch.randn(n_samples, *img_shape).to(device)
        labels = labels.to(device)
        
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
            xt = self.p_sample(xt, t_batch, labels)
        
        return xt.clamp(0, 1)
    
    def forward(self, x, t, token):
        return self.backbone(x, t, token)

if __name__ == "__main__":
    # 使用示例
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DiffusionModel(img_size=128, img_channels=3, device=device)
    model.to(device)

    # 假设的输入
    batch_size = 4
    x = torch.randn(batch_size, 3, 128, 128).to(device)
    t = torch.randint(0, 1000, (batch_size,)).to(device)

    tokens = torch.tensor(LicensePlateVocab.text_to_sequence("鲁AB650B")).to(device)
    tokens = tokens.repeat(batch_size, 1)

    t = torch.randint(0, model.num_timesteps, (4,), device=device)
    x_t, noise = model.q_sample(x, t)
    
    # 前向传播
    output = model(x, t, tokens)
    print(output.shape)  # 应该输出 torch.Size([4, 3, 128, 128])