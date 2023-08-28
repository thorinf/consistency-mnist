import math

import torch

from utils import append_dims


class Consistency:
    def __init__(self, score_model, ema_score_model, continuous=False, sigma_min=0.002, sigma_max=80.0, sigma_data=0.5,
                 rho=7.0):
        super(Consistency, self).__init__()
        self.score_model = score_model
        self.ema_score_model = ema_score_model

        self.continuous = continuous
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho

    @torch.no_grad()
    def update_ema(self, mu=0.95):
        for weight, ema_weight in zip(self.score_model.parameters(), self.ema_score_model.parameters()):
            ema_weight.mul_(mu).add_(weight, alpha=1 - mu)

    def rho_schedule(self, u):
        # u [0,1]
        rho_inv = 1.0 / self.rho
        sigma_max_pow_rho_inv = self.sigma_max ** rho_inv
        sigmas = (sigma_max_pow_rho_inv + u * (self.sigma_min ** rho_inv - sigma_max_pow_rho_inv)) ** self.rho
        return sigmas

    @staticmethod
    def get_snr(sigma):
        return sigma ** -2.0

    def get_weights(self, snr):
        return snr + 1.0 / self.sigma_data ** 2.0

    def get_scaling(self, sigma):
        c_skip = self.sigma_data ** 2 / ((sigma - self.sigma_min) ** 2 + self.sigma_data ** 2)
        c_out = (sigma - self.sigma_min) * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in

    def denoise(self, model, x_t, sigmas, **model_kwargs):
        c_skip, c_out, c_in = self.get_scaling(sigmas)
        rescaled_t = 0.25 * torch.log(sigmas + 1e-44)
        model_output = model(append_dims(c_in, x_t.ndim) * x_t, rescaled_t, **model_kwargs)
        denoised = append_dims(c_out, model_output.ndim) * model_output + append_dims(c_skip, x_t.ndim) * x_t
        return model_output, denoised

    def loss_t(self, x, t, t_next, **model_kwargs):
        z = torch.randn_like(x, device=x.device)

        x_t = x + z * append_dims(t, z.ndim)
        x_t_next = (x + z * append_dims(t_next, z.ndim)).detach()

        dropout_state = torch.get_rng_state()
        _, denoised_x = self.denoise(self.score_model, x_t, t, **model_kwargs)
        with torch.no_grad():
            torch.set_rng_state(dropout_state)
            _, target_x = self.denoise(self.ema_score_model, x_t_next, t_next, **model_kwargs)
            target_x = target_x.detach()

        snrs = self.get_snr(t)
        weights = self.get_weights(snrs)
        mse = ((denoised_x - target_x) ** 2.0).mean(-1)

        consistency_loss = (append_dims(weights, mse.ndim) * mse)
        return consistency_loss.mean()

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
    def sample(self, x, ts, return_list=False, **model_kwargs):
        x_list = []

        _, x = self.denoise(self.ema_score_model, x, ts[0].unsqueeze(0), **model_kwargs)

        if return_list:
            x_list.append(x)

        for t in ts[1:]:
            t = t.unsqueeze(0)
            z = torch.randn_like(x)
            x = x + math.sqrt(t ** 2 - self.sigma_min ** 2) * z
            _, x = self.denoise(self.ema_score_model, x, t, **model_kwargs)
            x = x.clamp(-1.0, 1.0)

            if return_list:
                x_list.append(x)

        return x_list if return_list else x
