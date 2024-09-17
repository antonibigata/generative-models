from typing import Dict, List, Optional, Tuple, Union

import math
import torch
import torch.nn as nn
from einops import rearrange, repeat
import lpips

from ...modules.autoencoding.lpips.loss.lpips import LPIPS
from ...modules.encoders.modules import GeneralConditioner, ConcatTimestepEmbedderND
from ...util import append_dims, instantiate_from_config, default
from .denoiser import Denoiser
from ...modules.autoencoding.temporal_ae import VideoDecoder


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
        lambda_upper: float = 1.0,
        fix_image_leak: bool = False,
        add_lpips: bool = False,
        weight_pixel: float = 0.0,
        n_frames_pixel: Optional[int] = 1,
        what_pixel_losses: Optional[List[str]] = [],
        disable_first_stage_autocast: bool = True,
    ):
        super().__init__()

        assert loss_type in ["l2", "l1", "lpips"]

        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)
        self.loss_weighting = instantiate_from_config(loss_weighting_config)

        self.loss_type = loss_type
        self.offset_noise_level = offset_noise_level
        self.lambda_lower = lambda_lower
        self.lambda_upper = lambda_upper
        self.add_lpips = add_lpips
        self.weight_pixel = weight_pixel
        self.n_frames_pixel = n_frames_pixel
        self.what_pixel_losses = what_pixel_losses

        self.en_and_decode_n_samples_a_time = 1
        self.disable_first_stage_autocast = disable_first_stage_autocast

        if loss_type == "lpips":
            self.lpips = LPIPS().eval()

        if add_lpips or "lpips" in what_pixel_losses:
            self.lpips = lpips.LPIPS(net="alex").eval()

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

    def decode_first_stage(self, z, first_stage_model):
        if len(z.shape) == 5:
            z = rearrange(z, "b c t h w -> (b t) c h w")

        z = 1.0 / 0.18215 * z
        n_samples = default(self.en_and_decode_n_samples_a_time, z.shape[0])

        n_rounds = math.ceil(z.shape[0] / n_samples)
        all_out = []
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for n in range(n_rounds):
                if isinstance(first_stage_model.decoder, VideoDecoder):
                    kwargs = {"timesteps": len(z[n * n_samples : (n + 1) * n_samples])}
                else:
                    kwargs = {}
                out = first_stage_model.decode(z[n * n_samples : (n + 1) * n_samples], **kwargs)
                all_out.append(out)
        out = torch.cat(all_out, dim=0)
        # out = rearrange(out, "b c h w -> b h w c")
        torch.cuda.empty_cache()
        return out

    def forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        conditioner: GeneralConditioner,
        input: torch.Tensor,
        batch: Dict,
        first_stage_model: nn.Module = None,
    ) -> torch.Tensor:
        cond = conditioner(batch)
        return self._forward(network, denoiser, cond, input, batch, first_stage_model)

    def _forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        cond: Dict,
        input: torch.Tensor,
        batch: Dict,
        first_stage_model: nn.Module = None,
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
        mask = cond.get("mask", None)
        w = append_dims(self.loss_weighting(sigmas), input.ndim)
        return self.get_loss(model_output, input, w, mask, first_stage_model, batch.get("original_frames", None))

    def get_loss(self, model_output, target, w, mask=None, first_stage_model=None, original_frames=None):
        if mask is not None:
            print(w.shape, mask.shape)
            w = w * mask

        T = 1
        if target.ndim == 5:
            target = rearrange(target, "b c t h w -> (b t) c h w")
            B = w.shape[0]
            T = target.shape[0] // B
            w = repeat(w, "b () () () () -> (b t) () () ()", t=T)
            # model_output = rearrange(model_output, "b c t h w -> (b t) c h w")
            # w = rearrange(w, "b ... -> b t ...")

        # other_losses_mask = torch.clone(w)
        or_w = w.clone()

        if self.lambda_lower != 1.0:
            weight_lower = torch.ones_like(model_output, device=w.device)
            weight_lower[:, :, model_output.shape[2] // 2 :] *= self.lambda_lower
            w = weight_lower * w

        if self.lambda_upper != 1.0:
            weight_upper = torch.ones_like(model_output, device=w.device)
            weight_upper[:, :, : model_output.shape[2] // 2] *= self.lambda_upper
            w = weight_upper * w
        loss_dict = {}

        if self.loss_type == "l2":
            loss = torch.mean((w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1)
        elif self.loss_type == "l1":
            loss = torch.mean((w * (model_output - target).abs()).reshape(target.shape[0], -1), 1)
        elif self.loss_type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
        else:
            raise NotImplementedError(f"Unknown loss type {self.loss_type}")

        loss_dict[self.loss_type] = loss.clone()
        loss_dict["loss"] = loss

        if self.add_lpips:
            loss_dict["lpips"] = w[:, 0, 0, 0] * self.lpips(
                (model_output[:, :3] * 0.18215).clip(-1, 1),
                (target[:, :3] * 0.18215).clip(-1, 1),
            ).reshape(-1)
            loss_dict["loss"] += loss_dict["lpips"].mean()

        if self.weight_pixel > 0.0:
            assert original_frames is not None
            # Randomly select n_frames_pixel frames
            selected_frames = torch.randperm(T)[: self.n_frames_pixel]
            selected_model_output = rearrange(model_output, "(b t) ... -> b t ...", t=T)[:, selected_frames]
            selected_model_output = rearrange(selected_model_output, "b t ... -> (b t) ...")
            selected_original_frames = original_frames[:, :, selected_frames]
            selected_original_frames = rearrange(selected_original_frames, "b c t ... -> (b t) c ...")
            selected_w = rearrange(or_w, "(b t) ... -> b t ...", t=T)[:, selected_frames]
            selected_w = rearrange(selected_w, "b t ... -> (b t) ...")
            decoded_frames = self.decode_first_stage(selected_model_output, first_stage_model)
            # print(decoded_frames.shape, selected_original_frames.shape, selected_w.shape)

            for loss_name in self.what_pixel_losses:
                if loss_name == "l2":
                    loss_pixel = torch.mean(
                        (selected_w * (decoded_frames - selected_original_frames) ** 2).reshape(
                            selected_original_frames.shape[0], -1
                        ),
                        1,
                    )
                    loss_dict["pixel_l2"] = self.weight_pixel * loss_pixel.mean()
                    loss += self.weight_pixel * loss_pixel.mean()
                elif loss_name == "lpips":
                    loss_pixel = (
                        self.lpips(decoded_frames, selected_original_frames).reshape(-1) * selected_w[:, 0, 0, 0]
                    )
                    loss_dict["pixel_lpips"] = loss_pixel.mean()
                    loss += self.weight_pixel * loss_pixel.mean()
                else:
                    raise NotImplementedError(f"Unknown pixel loss type {loss_name}")

        return loss_dict
