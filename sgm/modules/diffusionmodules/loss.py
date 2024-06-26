from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange, repeat

from ...modules.autoencoding.lpips.loss.lpips import LPIPS
from ...modules.encoders.modules import GeneralConditioner, ConcatTimestepEmbedderND
from ...util import append_dims, instantiate_from_config
from .denoiser import Denoiser


def logit_normal_sampler(m, s=1, beta_m=15, sample_num=1000000):
    y_samples = torch.randn(sample_num) * s + m
    x_samples = beta_m * (torch.exp(y_samples) / (1 + torch.exp(y_samples)))
    return x_samples


def mu_t(t, a=5, mu_max=1):
    t = t.to("cpu")
    return 2 * mu_max * t**a - mu_max


def get_sigma_s(t, a, beta_m):
    mu = mu_t(t, a=a)
    sigma_s = logit_normal_sampler(m=mu, sample_num=t.shape[0], beta_m=beta_m)
    return sigma_s


class StandardDiffusionLoss(nn.Module):
    def __init__(
        self,
        sigma_sampler_config: dict,
        loss_weighting_config: dict,
        loss_type: str = "l2",
        offset_noise_level: float = 0.0,
        batch2model_keys: Optional[Union[str, List[str]]] = None,
        lambda_lower: float = 1.0,
        fix_image_leak: bool = False,
    ):
        super().__init__()

        assert loss_type in ["l2", "l1", "lpips"]

        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)
        self.loss_weighting = instantiate_from_config(loss_weighting_config)

        self.loss_type = loss_type
        self.offset_noise_level = offset_noise_level
        self.lambda_lower = lambda_lower

        if loss_type == "lpips":
            self.lpips = LPIPS().eval()

        if not batch2model_keys:
            batch2model_keys = []

        if isinstance(batch2model_keys, str):
            batch2model_keys = [batch2model_keys]

        self.batch2model_keys = set(batch2model_keys)

        self.fix_image_leak = fix_image_leak
        if fix_image_leak:
            self.beta_m = 15
            self.a = 5
            self.noise_encoder = ConcatTimestepEmbedderND(256)

    def get_noised_input(self, sigmas_bc: torch.Tensor, noise: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        noised_input = input + noise * sigmas_bc
        return noised_input

    def forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        conditioner: GeneralConditioner,
        input: torch.Tensor,
        batch: Dict,
    ) -> torch.Tensor:
        cond = conditioner(batch)
        return self._forward(network, denoiser, cond, input, batch)

    def _forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        cond: Dict,
        input: torch.Tensor,
        batch: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        additional_model_inputs = {key: batch[key] for key in self.batch2model_keys.intersection(batch)}
        sigmas = self.sigma_sampler(input.shape[0]).to(input)

        noise = torch.randn_like(input)
        if self.offset_noise_level > 0.0:
            offset_shape = (
                (input.shape[0], 1, input.shape[2]) if self.n_frames is not None else (input.shape[0], input.shape[1])
            )
            noise = noise + self.offset_noise_level * append_dims(
                torch.randn(offset_shape, device=input.device),
                input.ndim,
            )
        sigmas_bc = append_dims(sigmas, input.ndim)
        noised_input = self.get_noised_input(sigmas_bc, noise, input)

        if self.fix_image_leak:
            noise_aug_strength = get_sigma_s(sigmas / 700, self.a, self.beta_m)
            noise_aug = append_dims(noise_aug_strength, 4).to(input.device)
            noise = torch.randn_like(noise_aug)
            cond["concat"] = self.get_noised_input(noise_aug, noise, cond["concat"])
            noise_emb = self.noise_encoder(noise_aug_strength).to(input.device)
            # cond["vector"] = noise_emb if "vector" not in cond else torch.cat([cond["vector"], noise_emb], dim=1)
            cond["vector"] = noise_emb
            # print(cond["concat"].shape, cond["vector"].shape, noise.shape, noise_aug.shape, noise_emb.shape)

        model_output = denoiser(network, noised_input, sigmas, cond, **additional_model_inputs)
        w = append_dims(self.loss_weighting(sigmas), input.ndim)
        return self.get_loss(model_output, input, w)

    def get_loss(self, model_output, target, w):
        if target.ndim == 5:
            target = rearrange(target, "b c t h w -> (b t) c h w")
            B = w.shape[0]
            w = repeat(w, "b () () () () -> (b t) () () ()", t=target.shape[0] // B)
            # model_output = rearrange(model_output, "b c t h w -> (b t) c h w")
            # w = rearrange(w, "b ... -> b t ...")

        if self.lambda_lower != 1.0:
            weight_lower = torch.ones_like(model_output, device=w.device)
            weight_lower[:, :, model_output.shape[2] // 2 :] *= self.lambda_lower
            w = weight_lower * w

        if self.loss_type == "l2":
            return torch.mean((w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1)
        elif self.loss_type == "l1":
            return torch.mean((w * (model_output - target).abs()).reshape(target.shape[0], -1), 1)
        elif self.loss_type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss
        else:
            raise NotImplementedError(f"Unknown loss type {self.loss_type}")
