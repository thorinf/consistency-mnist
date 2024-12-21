import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def append_dims(tensor, target_dims):
    assert isinstance(target_dims, int), f"Expected 'target_dims' to be an integer, but received {type(target_dims)}."
    tensor_dims = tensor.ndim
    assert tensor_dims <= target_dims, f"Tensor has {tensor_dims} dimensions, but target has {target_dims} dimensions."
    return tensor[(...,) + (None,) * (target_dims - tensor_dims)]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def update_model_ema(model: nn.Module, ema_model: nn.Module, mu: float = 0.95) -> None:
    for weight, ema_weight in zip(model.parameters(), ema_model.parameters()):
        ema_weight.mul_(mu).add_(weight, alpha=1 - mu)


def plot_images(images, subplot_shape, name, path, labels=None):
    fig_width = subplot_shape[1] * 2.0
    fig_height = subplot_shape[0] * 2.0
    fig, axes = plt.subplots(*subplot_shape, figsize=(fig_width, fig_height))
    fig.suptitle(name, fontsize=16)
    axes = axes.flatten()

    for i, (ax, img) in enumerate(zip(axes, images)):
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        if labels is not None:
            ax.set_title(labels[i], fontsize=8)

    plt.savefig(path)
    plt.close()


def plot_images_animation(images_list, subplot_shape, name, path, labels=None):
    fig_width = subplot_shape[1] * 2.0
    fig_height = subplot_shape[0] * 2.0
    fig, axes = plt.subplots(*subplot_shape, figsize=(fig_width, fig_height))
    fig.suptitle(name, fontsize=16)
    axes = axes.flatten()

    def animate(i):
        plots = []
        images = images_list[i]
        for i, (ax, img) in enumerate(zip(axes, images)):
            plots.append(ax.imshow(img, cmap="gray"))
            plots.append(ax.axis("off"))
            if labels is not None:
                plots.append(ax.set_title(labels[i], fontsize=8))
        return plots

    anim = FuncAnimation(fig, animate, frames=len(images_list), interval=10, blit=False, repeat=True)
    anim.save(path, writer="pillow", fps=10)
    plt.close()
