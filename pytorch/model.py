import math

import torch
import torch.nn as nn

from utils import append_dims


class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob=0.0):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, output_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout_prob)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
            nn.GroupNorm(32, output_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout_prob)
        )

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(x)


class DownBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob=0.0):
        super(DownBlock, self).__init__()
        self.conv_block = ConvBlock(input_dim, output_dim, dropout_prob=dropout_prob)
        self.down_sample = nn.MaxPool2d(2)

    def forward(self, x):
        skip_out = self.conv_block(x)
        down_out = self.down_sample(skip_out)
        return down_out, skip_out


class UpBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob=0.0):
        super(UpBlock, self).__init__()
        self.up_sample = nn.ConvTranspose2d(input_dim, output_dim, 2, 2)
        self.conv_block = ConvBlock(2 * output_dim, output_dim, dropout_prob=dropout_prob)

    def forward(self, x, skip_input):
        x = self.up_sample(x)
        x = torch.concat([x, skip_input], dim=1)
        x = self.conv_block(x)
        return x


class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super(LearnedSinusoidalPosEmb, self).__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        freq = x @ self.weights.unsqueeze(0)
        return torch.cat([append_dims(x, freq.ndim), freq.sin(), freq.cos()], dim=-1)


class UNet(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, dropout_prob=0.0):
        super(UNet, self).__init__()
        self.time_mlp = nn.Sequential(
            LearnedSinusoidalPosEmb(128),
            nn.SiLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(129, 512)
        )
        self.cond_emb = nn.Sequential(
            nn.Embedding(11, 128),
            nn.SiLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(128, 512),
        )
        self.down1 = DownBlock(input_dim, 128, dropout_prob=dropout_prob)
        self.down2 = DownBlock(128, 256, dropout_prob=dropout_prob)
        self.conv_block = ConvBlock(256, 512, dropout_prob=dropout_prob)
        self.up1 = UpBlock(512, 256, dropout_prob=dropout_prob)
        self.up2 = UpBlock(256, 128, dropout_prob=dropout_prob)
        self.act = nn.SiLU()
        self.output = nn.Conv2d(128, output_dim, kernel_size=1)

    def forward(self, x, t, labels=None):
        if labels is None:
            # All labels are set as unconditional
            labels = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
        elif self.training:
            # Bias labels for conditioning, set some as zero for unconditional training
            no_condition = torch.rand_like(labels, device=x.device, dtype=x.dtype) < 0.1
            labels = (labels + 1).masked_fill(no_condition, 0)
        else:
            # Bias labels for conditioning, ID zero is unconditional
            labels = labels + 1

        t, labels = append_dims(t, x.ndim), append_dims(labels, x.ndim - 1)
        emb = self.time_mlp(t) + self.cond_emb(labels)
        scale, shift = torch.chunk(emb.permute(0, 3, 1, 2), 2, dim=1)

        x = x.permute(0, 3, 1, 2)
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x = (x * scale) + shift
        x = self.conv_block(x)
        x = self.up1(x, skip2)
        x = self.up2(x, skip1)
        x = self.act(x)
        x = self.output(x)
        x = x.permute(0, 2, 3, 1)
        return x
