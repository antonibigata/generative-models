from typing import Dict, Union

import torch
import torch.nn as nn
from einops import repeat, rearrange
from ...util import append_dims, instantiate_from_config
from .denoiser_scaling import DenoiserScaling
from .discretizer import Discretization


def chunk_network(network, input, c_in, c_noise, cond, additional_model_inputs, chunk_size, num_frames=1):
    out = []

    for i in range(0, input.shape[0], chunk_size):
        start_idx = i
        end_idx = i + chunk_size

        input_chunk = input[start_idx:end_idx]
        c_in_chunk = (
            c_in[start_idx:end_idx]
            if c_in.shape[0] == input.shape[0]
            else c_in[start_idx // num_frames : end_idx // num_frames]
        )
        c_noise_chunk = (
            c_noise[start_idx:end_idx]
            if c_noise.shape[0] == input.shape[0]
            else c_noise[start_idx // num_frames : end_idx // num_frames]
        )

        cond_chunk = {}
        for k, v in cond.items():
            if isinstance(v, torch.Tensor) and v.shape[0] == input.shape[0]:
                cond_chunk[k] = v[start_idx:end_idx]
            elif isinstance(v, torch.Tensor):
                cond_chunk[k] = v[start_idx // num_frames : end_idx // num_frames]
            else:
                cond_chunk[k] = v

        additional_model_inputs_chunk = {}
        for k, v in additional_model_inputs.items():
            if isinstance(v, torch.Tensor) and v.shape[0] == input.shape[0]:
                additional_model_inputs_chunk[k] = v[start_idx:end_idx]
            elif isinstance(v, torch.Tensor):
                additional_model_inputs_chunk[k] = v[start_idx // num_frames : end_idx // num_frames]
            else:
                additional_model_inputs_chunk[k] = v

        out.append(network(input_chunk * c_in_chunk, c_noise_chunk, cond_chunk, **additional_model_inputs_chunk))

    return torch.cat(out, dim=0)


class Denoiser(nn.Module):
    def __init__(self, scaling_config: Dict):
        super().__init__()

        self.scaling: DenoiserScaling = instantiate_from_config(scaling_config)

    def possibly_quantize_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma

    def possibly_quantize_c_noise(self, c_noise: torch.Tensor) -> torch.Tensor:
        return c_noise

    def forward(
        self,
        network: nn.Module,
        input: torch.Tensor,
        sigma: torch.Tensor,
        cond: Dict,
        **additional_model_inputs,
    ) -> torch.Tensor:
        sigma = self.possibly_quantize_sigma(sigma)
        if input.ndim == 5:
            T = input.shape[2]
            input = rearrange(input, "b c t h w -> (b t) c h w")
            sigma = repeat(sigma, "b ... -> b t ...", t=T)
            sigma = rearrange(sigma, "b t ... -> (b t) ...")
        sigma_shape = sigma.shape
        sigma = append_dims(sigma, input.ndim)
        c_skip, c_out, c_in, c_noise = self.scaling(sigma)
        c_noise = self.possibly_quantize_c_noise(c_noise.reshape(sigma_shape))
        out = network(input * c_in, c_noise, cond, **additional_model_inputs)
        return out * c_out + input * c_skip


class DenoiserDub(nn.Module):
    def __init__(self, scaling_config: Dict):
        super().__init__()

        self.scaling: DenoiserScaling = instantiate_from_config(scaling_config)

    def possibly_quantize_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma

    def possibly_quantize_c_noise(self, c_noise: torch.Tensor) -> torch.Tensor:
        return c_noise

    def forward(
        self,
        network: nn.Module,
        input: torch.Tensor,
        sigma: torch.Tensor,
        cond: Dict,
        num_overlap_frames: int = 1,
        num_frames: int = 14,
        n_skips: int = 1,
        chunk_size: int = None,
        **additional_model_inputs,
    ) -> torch.Tensor:
        sigma = self.possibly_quantize_sigma(sigma)
        if input.ndim == 5:
            T = input.shape[2]
            input = rearrange(input, "b c t h w -> (b t) c h w")
            sigma = repeat(sigma, "b ... -> b t ...", t=T)
            sigma = rearrange(sigma, "b t ... -> (b t) ...")
        sigma_shape = sigma.shape
        sigma = append_dims(sigma, input.ndim)
        c_skip, c_out, c_in, c_noise = self.scaling(sigma)
        c_noise = self.possibly_quantize_c_noise(c_noise.reshape(sigma_shape))
        gt = cond.get("gt", torch.Tensor([]).type_as(input))
        gt = rearrange(gt, "b c t h w -> (b t) c h w")
        masks = cond.get("masks", None)
        masks = rearrange(masks, "b c t h w -> (b t) c h w")
        input = input * masks + gt * (1.0 - masks)
       
        if chunk_size is not None:
            assert chunk_size % num_frames == 0, "Chunk size should be multiple of num_frames"
            out = chunk_network(
                network, input, c_in, c_noise, cond, additional_model_inputs, chunk_size, num_frames=num_frames
            )
        else:
            out = network(input * c_in, c_noise, cond, **additional_model_inputs)
        out = out * c_out + input * c_skip
        out = out * masks + gt * (1.0 - masks)
        return out


class DenoiserTemporalMultiDiffusion(nn.Module):
    def __init__(self, scaling_config: Dict, is_dub: bool = False):
        super().__init__()

        self.scaling: DenoiserScaling = instantiate_from_config(scaling_config)
        self.is_dub = is_dub

    def possibly_quantize_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma

    def possibly_quantize_c_noise(self, c_noise: torch.Tensor) -> torch.Tensor:
        return c_noise

    def forward(
        self,
        network: nn.Module,
        input: torch.Tensor,
        sigma: torch.Tensor,
        cond: Dict,
        num_overlap_frames: int,
        num_frames: int,
        n_skips: int,
        chunk_size: int = None,
        **additional_model_inputs,
    ) -> torch.Tensor:
        """
        Args:
            network: Denoising network
            input: Noisy input
            sigma: Noise level
            cond: Dictionary containing additional information
            num_overlap_frames: Number of overlapping frames
            additional_model_inputs: Additional inputs for the denoising network
        Returns:
            out: Denoised output
        This function assumes the input is of shape (B, C, T, H, W) with the B dimension being the number of segments in video.
        The num_overlap_frames is the number of overlapping frames between the segments to be able to handle the temporal overlap.
        """
        sigma = self.possibly_quantize_sigma(sigma)
        T = num_frames
        if input.ndim == 5:
            T = input.shape[2]
            input = rearrange(input, "b c t h w -> (b t) c h w")
            sigma = repeat(sigma, "b ... -> b t ...", t=T)
            sigma = rearrange(sigma, "b t ... -> (b t) ...")
        sigma_shape = sigma.shape
        sigma = append_dims(sigma, input.ndim)
        c_skip, c_out, c_in, c_noise = self.scaling(sigma)
        c_noise = self.possibly_quantize_c_noise(c_noise.reshape(sigma_shape))
        if self.is_dub:
            gt = cond.get("gt", torch.Tensor([]).type_as(input))
            gt = rearrange(gt, "b c t h w -> (b t) c h w")
            masks = cond.get("masks", None)
            masks = rearrange(masks, "b c t h w -> (b t) c h w")
            input = input * masks + gt * (1.0 - masks)

        # Now we want to find the overlapping frames and average them
        input = rearrange(input, "(b t) c h w -> b c t h w", t=T)
        # Overlapping frames are at begining and end of each segment and given by num_overlap_frames
        for i in range(input.shape[0] - n_skips):
            average_frame = torch.stack(
                [input[i, :, -num_overlap_frames:], input[i + 1, :, :num_overlap_frames]]
            ).mean(0)
            input[i, :, -num_overlap_frames:] = average_frame
            input[i + n_skips, :, :num_overlap_frames] = average_frame

        input = rearrange(input, "b c t h w -> (b t) c h w")

        if chunk_size is not None:
            assert chunk_size % num_frames == 0, "Chunk size should be multiple of num_frames"
            out = chunk_network(
                network, input, c_in, c_noise, cond, additional_model_inputs, chunk_size, num_frames=num_frames
            )
        else:
            out = network(input * c_in, c_noise, cond, **additional_model_inputs)

        out = out * c_out + input * c_skip

        if self.is_dub:
            out = out * masks + gt * (1.0 - masks)
        return out


class DiscreteDenoiser(Denoiser):
    def __init__(
        self,
        scaling_config: Dict,
        num_idx: int,
        discretization_config: Dict,
        do_append_zero: bool = False,
        quantize_c_noise: bool = True,
        flip: bool = True,
    ):
        super().__init__(scaling_config)
        self.discretization: Discretization = instantiate_from_config(discretization_config)
        sigmas = self.discretization(num_idx, do_append_zero=do_append_zero, flip=flip)
        self.register_buffer("sigmas", sigmas)
        self.quantize_c_noise = quantize_c_noise
        self.num_idx = num_idx

    def sigma_to_idx(self, sigma: torch.Tensor) -> torch.Tensor:
        dists = sigma - self.sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape)

    def idx_to_sigma(self, idx: Union[torch.Tensor, int]) -> torch.Tensor:
        return self.sigmas[idx]

    def possibly_quantize_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        return self.idx_to_sigma(self.sigma_to_idx(sigma))

    def possibly_quantize_c_noise(self, c_noise: torch.Tensor) -> torch.Tensor:
        if self.quantize_c_noise:
            return self.sigma_to_idx(c_noise)
        else:
            return c_noise
