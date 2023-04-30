import os
import math

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision

from tqdm import tqdm
from matplotlib.animation import FuncAnimation

import argparse


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
        freq = torch.einsum('b,d->bd', x, self.weights) * 2 * math.pi
        return torch.cat([freq.sin(), freq.cos()], dim=-1)


class UNet(nn.Module):
    def __init__(self, dropout_prob=0.0):
        super(UNet, self).__init__()
        self.time_mlp = nn.Sequential(
            LearnedSinusoidalPosEmb(128),
            nn.SiLU(),
            nn.Linear(128, 256)
        )
        self.cond_emb = nn.Sequential(
            nn.Embedding(11, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
        )
        self.down1 = DownBlock(1, 128, dropout_prob=dropout_prob)
        self.down2 = DownBlock(128, 256, dropout_prob=dropout_prob)
        self.conv_block = ConvBlock(256, 512, dropout_prob=dropout_prob)
        self.up1 = UpBlock(512, 256, dropout_prob=dropout_prob)
        self.up2 = UpBlock(256, 128, dropout_prob=dropout_prob)
        self.act = nn.SiLU()
        self.output = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x, t, label=None):
        if label is None:
            label = torch.ones(x.shape[0], dtype=torch.int64, device=x.device)
        elif self.training:
            label = (label + 1).masked_fill(torch.rand_like(label, device=x.device, dtype=x.dtype) < 0.5, 0)
        else:
            label = label + 1
        time_emb = self.time_mlp(t)
        cond = self.cond_emb(label)
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x = (x * cond[:, :, None, None]) + time_emb[:, :, None, None]
        x = self.conv_block(x)
        x = self.up1(x, skip2)
        x = self.up2(x, skip1)
        x = self.act(x)
        x = self.output(x)
        return x


class ConsistencyMNIST(nn.Module):
    def __init__(self, score_model, score_model_ema, continuous=False, sigma_min=0.002, sigma_max=80.0, sigma_data=0.5,
                 rho=7.0):
        super(ConsistencyMNIST, self).__init__()
        self.score_model = score_model
        self.score_model_ema = score_model_ema

        self.continuous = continuous
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho

    @torch.no_grad()
    def update_ema(self, mu=0.95):
        for weight, ema_weight in zip(self.score_model.parameters(), self.score_model_ema.parameters()):
            ema_weight.mul_(mu).add_(weight, alpha=1 - mu)

    def rho_schedule(self, u):
        # u [0,1]
        rho_inv = 1.0 / self.rho
        sigma_max_pow_rho_inv = self.sigma_max ** rho_inv
        sigmas = (sigma_max_pow_rho_inv + u * (self.sigma_min ** rho_inv - sigma_max_pow_rho_inv)) ** self.rho
        return sigmas

    def get_snr(self, sigma):
        return sigma ** -2.0

    def get_weights(self, snr):
        return snr + 1.0 / self.sigma_data ** 2.0

    def get_scaling(self, sigma):
        c_skip = self.sigma_data ** 2 / ((sigma - self.sigma_min) ** 2 + self.sigma_data ** 2)
        c_out = (sigma - self.sigma_min) * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in

    def loss_weight(self, t):
        return (t ** 2.0 + self.sigma ** 2.0) * torch.rsqrt(t * self.sigma)

    def denoise(self, model, x_t, sigmas, **model_kwargs):
        c_skip, c_out, c_in = self.get_scaling(sigmas)
        rescaled_t = 0.25 * torch.log(sigmas + 1e-44)
        model_output = model(c_in.view(-1, 1, 1, 1) * x_t, rescaled_t, **model_kwargs)
        denoised = c_out.view(-1, 1, 1, 1) * model_output + c_skip.view(-1, 1, 1, 1) * x_t
        return model_output, denoised

    def loss_t(self, x, t, t_next, **model_kwargs):
        z = torch.randn_like(x, device=x.device)

        x_t = x + z * t.view(-1, 1, 1, 1)
        x_t_next = (x + z * t_next.view(-1, 1, 1, 1)).detach()

        dropout_state = torch.get_rng_state()
        _, denoised_x = self.denoise(self.score_model, x_t, t, **model_kwargs)
        with torch.no_grad():
            torch.set_rng_state(dropout_state)
            _, target_x = self.denoise(self.score_model_ema, x_t_next, t_next, **model_kwargs)
            target_x = target_x.detach()

        snrs = self.get_snr(t)
        weights = self.get_weights(snrs)

        return (((denoised_x - target_x) ** 2.0).mean(dim=[1, 2, 3]) * weights).mean()

    def compute_loss(self, x, num_scales, **model_kwargs):
        offset = 1.0 / (num_scales - 1)

        if self.continuous:
            rand_u_1 = torch.rand((x.shape[0],), dtype=x.dtype, device=x.device, requires_grad=False)
            rand_u_1 = rand_u_1 * (1.0 - offset)
        else:
            rand_u_1 = torch.randint(0, num_scales - 1, (x.shape[0],), device=x.device, requires_grad=False)
            rand_u_1 = rand_u_1 / (num_scales - 1)
        rand_u_2 = torch.clamp(rand_u_1 + offset, 0.0, 1.0)

        t_1 = self.rho_schedule(rand_u_1)
        t_2 = self.rho_schedule(rand_u_2)

        return self.loss_t(x, t_1, t_2, **model_kwargs)

    @torch.no_grad()
    def forward(self, x, ts, return_list=False, **model_kwargs):
        x_list = []

        _, x = self.denoise(self.score_model_ema, x, ts[0].unsqueeze(0), **model_kwargs)

        if return_list:
            x_list.append(x)

        for t in ts[1:]:
            t = t.unsqueeze(0)
            z = torch.randn_like(x)
            x = x + math.sqrt(t ** 2 - self.sigma_min ** 2) * z
            _, x = self.denoise(self.score_model_ema, x, t, **model_kwargs)

            if return_list:
                x_list.append(x.clamp(-1.0, 1.0))

        return x_list if return_list else x.clamp(-1.0, 1.0)


def plot_images(images, subplot_shape, name, path):
    fig, axes = plt.subplots(*subplot_shape)
    fig.suptitle(name, fontsize=16)
    axes = axes.flatten()

    for ax, img in zip(axes, images):
        ax.imshow(img, cmap='gray')
        ax.axis('off')

    plt.savefig(path)


def plot_images_animation(images_list, subplot_shape, name, path):
    fig, axes = plt.subplots(*subplot_shape)
    fig.suptitle(name, fontsize=16)
    axes = axes.flatten()

    def animate(i):
        plots = []
        images = images_list[i]
        for ax, img in zip(axes, images):
            plots.append(ax.imshow(img, cmap='gray'))
            plots.append(ax.axis('off'))
        return plots

    anim = FuncAnimation(fig, animate, frames=len(images_list), interval=10, blit=False, repeat=True)
    anim.save(path, writer='pillow', fps=10)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ep', '--epochs', type=int, default=100)
    parser.add_argument('-b', '--batch_size', type=int, default=256)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-4)
    parser.add_argument('-acc', '--accumulation_steps', type=int, default=1)

    parser.add_argument('-c', '--continuous', type=bool, default=True)
    parser.add_argument('-do', '--dropout_prob', type=float, default=0.0)
    parser.add_argument('-smin', '--sigma_min', type=float, default=0.002)
    parser.add_argument('-smax', '--sigma_max', type=float, default=80.0)
    parser.add_argument('-sdat', '--sigma_data', type=float, default=0.6)
    parser.add_argument('-rho', '--rho', type=float, default=7.0)

    parser.add_argument('-mu0', '--mu_zero', type=float, default=0.9)
    parser.add_argument('-s0', '--s_zero', type=float, default=2.0)
    parser.add_argument('-s1', '--s_one', type=float, default=150.0)

    parser.add_argument('-ckpt', '--checkpoint', type=str, required=True)
    parser.add_argument('-d', '--data_path', type=str, required=True)

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    score_model = UNet(dropout_prob=args.dropout_prob)
    score_model_ema = UNet(dropout_prob=args.dropout_prob)
    score_model_ema.load_state_dict(score_model.state_dict())

    model = ConsistencyMNIST(
        score_model=score_model,
        score_model_ema=score_model_ema,
        continuous=args.continuous,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        sigma_data=args.sigma_data,
        rho=args.rho
    )
    model.to(device)

    if os.path.exists(args.checkpoint):
        print(f"Restoring Checkpoint: {args.checkpoint}.")
        checkpoint = torch.load(args.checkpoint)
    else:
        print(f"Starting new training run: {args.checkpoint}.")
        checkpoint = {}

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])

    dataset = torchvision.datasets.MNIST(
        root=args.data_path,
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )

    optim = torch.optim.AdamW(
        model.score_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    if 'optimizer_state_dict' in checkpoint:
        optim.load_state_dict(checkpoint['optimizer_state_dict'])

    for ep in range(checkpoint.get('epochs', 0), args.epochs):
        model.train()

        elapsed = ep / args.epochs
        N = math.ceil(math.sqrt((elapsed * ((args.s_one + 1) ** 2 - args.s_zero ** 2)) + args.s_zero ** 2) - 1) + 1
        mu = math.exp(args.s_zero * math.log(args.mu_zero) / N)

        pbar = tqdm(dataloader)
        pbar.set_description(f"epoch: {ep}")

        for idx, (img, label) in enumerate(pbar):
            img = (img * 2) - 1
            img = img.to(device)
            label = label.to(device)

            loss = model.compute_loss(img, N, label=label)

            pbar.set_postfix({
                "loss": loss.item()
            })

            (loss / args.accumulation_steps).backward()

            if ((idx + 1) % args.accumulation_steps == 0) or (idx + 1 == len(dataloader)):
                optim.step()
                model.update_ema(mu)
                optim.zero_grad()
                torch.cuda.empty_cache()

        checkpoint = {
            'epochs': ep + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict()
        }
        torch.save(checkpoint, args.checkpoint)

        model.score_model_ema.eval()
        num_steps = 10
        with torch.no_grad():
            ts = model.rho_schedule(torch.arange(num_steps, device=device) / num_steps)
            x_init = torch.randn(16, 1, 28, 28, device=device) * ts[0]
            label = torch.tensor([x if x < 10 else -1 for x in range(16)], dtype=torch.int64, device=device)
            x_list = model(x_init, ts, label=label, return_list=True)
            x_list = [((x + 1) / 2).clamp(0.0, 1.0).permute(0, 2, 3, 1).cpu().numpy() for x in x_list]

        plot_images(x_list[-1], (4, 4), f"Epoch: {ep}, Steps: {num_steps}", f"epoch-{ep}_steps-{num_steps}.png")
        plot_images_animation(x_list, (4, 4), f"Epoch: {ep}, Steps: {num_steps}", f"epoch-{ep}_steps-{num_steps}.gif")


if __name__ == "__main__":
    train()
