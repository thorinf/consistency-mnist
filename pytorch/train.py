import argparse
import math
import os

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from model import UNet
from consistency import Consistency
from utils import count_parameters, plot_images, plot_images_animation


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
    parser.add_argument('-sdat', '--sigma_data', type=float, default=0.5)
    parser.add_argument('-rho', '--rho', type=float, default=7.0)

    parser.add_argument('-mu0', '--mu_zero', type=float, default=0.95)
    parser.add_argument('-s0', '--s_zero', type=float, default=2.0)
    parser.add_argument('-s1', '--s_one', type=float, default=20.0)

    parser.add_argument('-ckpt', '--checkpoint', type=str, required=True)
    parser.add_argument('-d', '--data_path', type=str, required=True)

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    score_model = UNet(dropout_prob=args.dropout_prob)
    score_model.to(device)

    if os.path.exists(args.checkpoint):
        print(f"Restoring Checkpoint: {args.checkpoint}.")
        checkpoint = torch.load(args.checkpoint)
    else:
        print(f"Starting new training run: {args.checkpoint}.")
        checkpoint = {}

    if 'model_state_dict' in checkpoint:
        score_model.load_state_dict(checkpoint['model_state_dict'])

    num_params = count_parameters(score_model)
    print(f"Total number of parameters: {num_params:,}")

    ema_score_model = UNet(dropout_prob=args.dropout_prob)
    ema_score_model.to(device)

    if 'ema_model_state_dict' in checkpoint:
        ema_score_model.load_state_dict(checkpoint['ema_model_state_dict'])
    else:
        ema_score_model.load_state_dict(score_model.state_dict())

    consistency = Consistency(
        score_model=score_model,
        ema_score_model=ema_score_model,
        continuous=args.continuous,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        sigma_data=args.sigma_data,
        rho=args.rho
    )

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
        persistent_workers=True
    )

    optim = torch.optim.AdamW(
        score_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    if 'optimizer_state_dict' in checkpoint:
        optim.load_state_dict(checkpoint['optimizer_state_dict'])

    global_step = checkpoint.get('global_step', 0)

    for ep in range(checkpoint.get('epochs', 0), args.epochs):
        score_model.train()
        ema_score_model.train()

        pbar = tqdm(dataloader)
        pbar.set_description(f"epoch: {ep}")

        elapsed = ep / args.epochs
        N = math.ceil(math.sqrt((elapsed * ((args.s_one + 1) ** 2 - args.s_zero ** 2)) + args.s_zero ** 2) - 1) + 1
        mu = math.exp(args.s_zero * math.log(args.mu_zero) / N)

        for idx, (data, labels) in enumerate(pbar):
            data = data.permute(0, 2, 3, 1) * 2 - 1
            data = data.to(device)
            labels = labels.to(device)

            loss = consistency.compute_loss(data, N, labels=labels)

            pbar.set_postfix({
                "loss": loss.item()
            })

            (loss / args.accumulation_steps).backward()

            if ((idx + 1) % args.accumulation_steps == 0) or (idx + 1 == len(dataloader)):
                optim.step()
                consistency.update_ema(mu)
                optim.zero_grad()
                global_step += 1

        checkpoint = {
            'epochs': ep + 1,
            'global_step': global_step,
            'model_state_dict': score_model.state_dict(),
            'ema_model_state_dict': ema_score_model.state_dict(),
            'optimizer_state_dict': optim.state_dict()
        }
        torch.save(checkpoint, args.checkpoint)

        if (ep + 1) % 5 != 0:
            continue

        score_model.eval()
        ema_score_model.eval()
        num_steps = 10
        with torch.no_grad():
            ts = consistency.rho_schedule(torch.arange(num_steps, device=device) / num_steps)
            x_init = torch.randn((16, 28, 28, 1), device=device) * ts[0]
            labels = torch.tensor([x if x < 10 else -1 for x in range(16)], dtype=torch.int64, device=device)
            x_list = consistency.sample(x_init, ts, labels=labels, return_list=True)
            x_list = [((x + 1) / 2).clamp(0.0, 1.0).cpu().numpy() for x in x_list]

        plot_images(
            images=x_list[-1],
            subplot_shape=(4, 4),
            name=f"Epoch: {ep}, Steps: {num_steps}",
            path=f"figures/epoch-{ep}_steps-{num_steps}.png",
            labels=labels.tolist()
        )
        plot_images_animation(
            images_list=x_list,
            subplot_shape=(4, 4),
            name=f"Epoch: {ep}, Steps: {num_steps}",
            path=f"figures/epoch-{ep}_steps-{num_steps}.gif",
            labels=labels.tolist()
        )


if __name__ == "__main__":
    train()
