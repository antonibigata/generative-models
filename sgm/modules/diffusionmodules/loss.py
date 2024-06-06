from typing import Dict, List, Optional, Tuple, Union

import math
import torch
import torch.nn as nn
from einops import rearrange, repeat

from ...modules.autoencoding.lpips.loss.lpips import LPIPS
from ...modules.encoders.modules import GeneralConditioner
from ...util import append_dims, instantiate_from_config, default
from .denoiser import Denoiser
from ...modules.lipreader.lightnining import ModelModule
from ...modules.lipreader.preparation.detectors.retinaface.video_process import VideoProcess
from ...modules.lipreader.datamodule.transforms import VideoTransform
from ...modules.autoencoding.temporal_ae import VideoDecoder


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
    ):
        super().__init__()

        assert loss_type in ["l2", "l1", "lpips"]

        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)
        self.loss_weighting = instantiate_from_config(loss_weighting_config)

        self.loss_type = loss_type
        self.offset_noise_level = offset_noise_level
        self.lambda_lower = lambda_lower
        self.lambda_upper = lambda_upper

        if loss_type == "lpips":
            self.lpips = LPIPS().eval()

        if not batch2model_keys:
            batch2model_keys = []

        if isinstance(batch2model_keys, str):
            batch2model_keys = [batch2model_keys]

        self.batch2model_keys = set(batch2model_keys)

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
        first_stage_model: nn.Module = None,
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

        model_output = denoiser(network, noised_input, sigmas, cond, **additional_model_inputs)
        mask = cond.get("mask", None)
        w = append_dims(self.loss_weighting(sigmas), input.ndim)
        return self.get_loss(model_output, input, w, mask)

    def get_loss(self, model_output, target, w, mask=None):
        if mask is not None:
            print(w.shape, mask.shape)
            w = w * mask

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

        if self.lambda_upper != 1.0:
            weight_upper = torch.ones_like(model_output, device=w.device)
            weight_upper[:, :, : model_output.shape[2] // 2] *= self.lambda_upper
            w = weight_upper * w

        if self.loss_type == "l2":
            loss = torch.mean((w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1)
        elif self.loss_type == "l1":
            loss = torch.mean((w * (model_output - target).abs()).reshape(target.shape[0], -1), 1)
        elif self.loss_type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            loss = loss
        else:
            raise NotImplementedError(f"Unknown loss type {self.loss_type}")

        return {"loss": loss}


class StandardWithLipLoss(StandardDiffusionLoss):
    def __init__(
        self,
        sigma_sampler_config: dict,
        loss_weighting_config: dict,
        loss_type: str = "l2",
        offset_noise_level: float = 0.0,
        batch2model_keys: Optional[Union[str, List[str]]] = None,
        lambda_lower: float = 1.0,
        lambda_upper: float = 1.0,
        weight_path: str = "",
        lip_weight: float = 1.0,
        en_and_decode_n_samples_a_time: Optional[int] = 1,
        disable_first_stage_autocast: bool = True,
        n_frames: Optional[int] = 14,
    ):
        super().__init__(
            sigma_sampler_config,
            loss_weighting_config,
            loss_type,
            offset_noise_level,
            batch2model_keys,
            lambda_lower,
            lambda_upper,
        )

        def get_lightning_module(ckpt_path):
            modelmodule = ModelModule()
            modelmodule.model.load_state_dict(torch.load(ckpt_path, map_location=lambda storage, loc: storage))
            return modelmodule

        self.lip_model = get_lightning_module(weight_path)
        self.lip_model.eval()
        # No grad
        for param in self.lip_model.parameters():
            param.requires_grad = False

        self.n_frames = n_frames

        self.lip_weight = lip_weight
        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time
        self.disable_first_stage_autocast = disable_first_stage_autocast

        self.video_process = VideoProcess(convert_gray=False)
        self.video_transform = VideoTransform(subset="val", max_noise_level=250)

    # @torch.no_grad()
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

    def forward_without_loss(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        conditioner: GeneralConditioner,
        input: torch.Tensor,
        batch: Dict,
        first_stage_model: nn.Module = None,
    ) -> torch.Tensor:
        cond = conditioner(batch)
        return self._forward(network, denoiser, cond, input, batch, first_stage_model, no_loss=True)

    def _forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        cond: Dict,
        input: torch.Tensor,
        batch: Dict,
        first_stage_model: nn.Module = None,
        no_loss: bool = False,
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

        model_output = denoiser(network, noised_input, sigmas, cond, **additional_model_inputs)
        mask = cond.get("mask", None)
        w = append_dims(self.loss_weighting(sigmas), input.ndim)
        if no_loss:
            return model_output, input, w, mask, batch.get("landmarks"), batch.get("lip_emb")
        return self.get_loss(
            model_output, input, w, mask, batch.get("landmarks"), batch.get("lip_emb"), first_stage_model
        )

    def get_loss(self, model_output, target, w, mask=None, landmarks=None, lip_emb=None, first_stage_model=None):
        loss_dict = {}

        if mask is not None:
            print(w.shape, mask.shape)
            w = w * mask

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

        if self.lambda_upper != 1.0:
            weight_upper = torch.ones_like(model_output, device=w.device)
            weight_upper[:, :, : model_output.shape[2] // 2] *= self.lambda_upper
            w = weight_upper * w

        if self.loss_type == "l2":
            loss = torch.mean((w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1)
        elif self.loss_type == "l1":
            loss = torch.mean((w * (model_output - target).abs()).reshape(target.shape[0], -1), 1)
        elif self.loss_type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
        else:
            raise NotImplementedError(f"Unknown loss type {self.loss_type}")

        loss_dict[f"{self.loss_type}_loss"] = loss

        del target
        torch.cuda.empty_cache()

        # Lip loss
        z = model_output
        video = self.decode_first_stage(z, first_stage_model).clip(-1, 1)
        landmarks = rearrange(landmarks, "b t c d -> (b t) c d")
        lip_emb = rearrange(lip_emb, "b t c -> (b t) c")
        video_proccessed = self.video_process(video, landmarks, True)
        # video_proccessed = torch.from_numpy(video_proccessed)
        # video_proccessed = video_proccessed.permute((0, 3, 1, 2))
        video_proccessed, t = self.video_transform(video_proccessed)

        pred_lip_emb = self.lip_model(video_proccessed, extract_position="conformer").squeeze(0)
        lip_loss = torch.mean(
            (w.squeeze().unsqueeze(1) * (pred_lip_emb - lip_emb) ** 2).reshape(lip_emb.shape[0], -1), 1
        )

        loss_dict["lip_loss"] = lip_loss * self.lip_weight

        loss_dict["loss"] = loss_dict[f"{self.loss_type}_loss"] + loss_dict["lip_loss"]

        del video, video_proccessed, pred_lip_emb

        return loss_dict

    def diffusion_loss(self, model_output, target, w, mask=None):
        if mask is not None:
            print(w.shape, mask.shape)
            w = w * mask

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

        if self.lambda_upper != 1.0:
            weight_upper = torch.ones_like(model_output, device=w.device)
            weight_upper[:, :, : model_output.shape[2] // 2] *= self.lambda_upper
            w = weight_upper * w

        if self.loss_type == "l2":
            return torch.mean((w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1)
        elif self.loss_type == "l1":
            return torch.mean((w * (model_output - target).abs()).reshape(target.shape[0], -1), 1)
        elif self.loss_type == "lpips":
            return self.lpips(model_output, target).reshape(-1)
        else:
            raise NotImplementedError(f"Unknown loss type {self.loss_type}")

    def lip_loss(self, model_output, w, landmarks=None, lip_emb=None, first_stage_model=None):
        # Lip loss
        z = model_output
        video = self.decode_first_stage(z, first_stage_model).clip(-1, 1)
        landmarks = rearrange(landmarks, "b t c d -> (b t) c d")
        lip_emb = rearrange(lip_emb, "b t c -> (b t) c")
        video_proccessed = self.video_process(video, landmarks, True)
        # video_proccessed = torch.from_numpy(video_proccessed)
        # video_proccessed = video_proccessed.permute((0, 3, 1, 2))
        video_proccessed, t = self.video_transform(video_proccessed)

        pred_lip_emb = self.lip_model(video_proccessed, extract_position="conformer").squeeze(0)
        lip_loss = torch.mean(
            (w.squeeze().unsqueeze(1) * (pred_lip_emb - lip_emb) ** 2).reshape(lip_emb.shape[0], -1), 1
        )

        return lip_loss * self.lip_weight
