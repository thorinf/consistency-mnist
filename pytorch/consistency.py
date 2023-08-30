import math
from typing import Any, List, Tuple, TypeVar, Union

import torch
import torch.nn as nn

from utils import append_dims

T = TypeVar('T', float, torch.Tensor)


class Consistency:
    def __init__(
            self,
            sigma_min: float = 0.002,
            sigma_max: float = 80.0,
            sigma_data: float = 0.5,
            rho: float = 7.0,
            continuous: bool = False
    ) -> None:
        super(Consistency, self).__init__()
        self.continuous = continuous
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho

    def rho_schedule(self, u: T) -> T:
        # u [0,1]
        rho_inv = 1.0 / self.rho
        sigma_max_pow_rho_inv = self.sigma_max ** rho_inv
        sigmas = (sigma_max_pow_rho_inv + u * (self.sigma_min ** rho_inv - sigma_max_pow_rho_inv)) ** self.rho
        return sigmas

    @staticmethod
    def get_snr(sigma: T) -> T:
        return sigma ** -2.0

    def get_weights(self, snr: T) -> T:
        return snr + 1.0 / self.sigma_data ** 2.0

    def get_scaling(self, sigma: T) -> Tuple[T, T, T]:
        c_skip = self.sigma_data ** 2 / ((sigma - self.sigma_min) ** 2 + self.sigma_data ** 2)
        c_out = (sigma - self.sigma_min) * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in

    def denoise(
            self,
            model: nn.Module,
            x_t: torch.Tensor,
            sigmas: torch.Tensor,
            **model_kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        c_skip, c_out, c_in = self.get_scaling(sigmas)
        rescaled_t = append_dims(0.25 * torch.log(sigmas + 1e-44), x_t.ndim)
        model_output = model(append_dims(c_in, x_t.ndim) * x_t, rescaled_t, **model_kwargs)
        denoised = append_dims(c_out, model_output.ndim) * model_output + append_dims(c_skip, x_t.ndim) * x_t
        return model_output, denoised

    def loss_t(
            self,
            model: nn.Module,
            target_model: nn.Module,
            x_target: torch.Tensor,
            t: torch.Tensor,
            t_next: torch.Tensor,
            **model_kwargs: Any
    ) -> torch.Tensor:
        z = torch.randn_like(x_target, device=x_target.device)

        x_t = x_target + z * append_dims(t, z.ndim)
        x_t_next = (x_target + z * append_dims(t_next, z.ndim)).detach()

        dropout_state = torch.get_rng_state()
        _, denoised_x = self.denoise(model, x_t, t, **model_kwargs)
        with torch.no_grad():
            torch.set_rng_state(dropout_state)
            _, target_x = self.denoise(target_model, x_t_next, t_next, **model_kwargs)
            target_x = target_x.detach()

        snrs = self.get_snr(t)
        weights = self.get_weights(snrs)
        mse = ((denoised_x - target_x) ** 2.0).mean(-1)

        consistency_loss = (append_dims(weights, mse.ndim) * mse)
        return consistency_loss.mean()

    def compute_loss(
            self,
            model: nn.Module,
            target_model: nn.Module,
            x_target: torch.Tensor,
            num_scales: int,
            **model_kwargs: Any
    ) -> torch.Tensor:
        offset = 1.0 / (num_scales - 1)

        if self.continuous:
            rand_u_1 = torch.rand((x_target.shape[0],), dtype=x_target.dtype, device=x_target.device,
                                  requires_grad=False)
            rand_u_1 = rand_u_1 * (1.0 - offset)
        else:
            rand_u_1 = torch.randint(0, num_scales - 1, (x_target.shape[0],), device=x_target.device,
                                     requires_grad=False)
            rand_u_1 = rand_u_1 / (num_scales - 1)
        rand_u_2 = torch.clamp(rand_u_1 + offset, 0.0, 1.0)

        t_1 = self.rho_schedule(rand_u_1)
        t_2 = self.rho_schedule(rand_u_2)

        return self.loss_t(model, target_model, x_target, t_1, t_2, **model_kwargs)

    @torch.no_grad()
    def sample(
            self,
            model: nn.Module,
            x_start: torch.Tensor,
            ts: torch.Tensor,
            z: torch.Tensor = None,
            x_min: float = -1.0,
            x_max: float = 1.0,
            return_all: bool = False,
            **model_kwargs: Any
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        x_list = []

        t = append_dims(ts[0], x_start.ndim)
        _, x = self.denoise(model, x_start, t, **model_kwargs)

        if return_all:
            x_list.append(x)

        for t in ts[1:]:
            t = append_dims(t, x.ndim)
            z = torch.randn_like(x) if z is None else z
            x = x + z * (t ** 2 - self.sigma_min ** 2).sqrt()
            _, x = self.denoise(model, x, t, **model_kwargs)
            x = x.clamp(x_min, x_max)

            if return_all:
                x_list.append(x)

        return x_list if return_all else x
