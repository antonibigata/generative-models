#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os
import random

import sentencepiece
import torch
import torchvision

import math
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


SP_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "spm",
    "unigram",
    "unigram5000.model",
)

DICT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "spm",
    "unigram",
    "unigram5000_units.txt",
)


####################################################################
# Imported from diffusion_foward.py
####################################################################
def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


def identity(x):
    return x


################New Diffusion Model#################################


class ForwardDiffusion(nn.Module):
    def __init__(
        self,
        timesteps=1000,
        max_noise_level=None,
        beta_schedule="cosine",
        schedule_fn_kwargs=dict(),
        auto_normalize=False,  # if True, normalize image from [0,1] to [-1,1]
    ) -> None:
        super().__init__()

        if beta_schedule == "linear":
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == "cosine":
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == "sigmoid":
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f"unknown beta schedule {beta_schedule}")

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.noise_level = default(max_noise_level, self.num_timesteps)

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))  # noqa

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def forward(self, img, time_step=None):
        """
        Takes images and returns noised image with timestep
        args: img (tensor): image tensor in range [0,1] if auto_normalize is True else [-1,1]
        """

        b = img.shape[0]
        device = img.device
        if time_step is None:
            t = torch.randint(0, self.noise_level, (b,), device=device).long()
        else:
            t = time_step

        img = self.normalize(img)
        return self.q_sample(img, t), t


####################################################################


class FunctionalModule(torch.nn.Module):
    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, input):
        return self.functional(input)


class AdaptiveTimeMask(torch.nn.Module):
    def __init__(self, window, stride):
        super().__init__()
        self.window = window
        self.stride = stride

    def forward(self, x):
        # x: [T, ...]
        cloned = x.clone()
        length = cloned.size(0)
        n_mask = int((length + self.stride - 0.1) // self.stride)
        ts = torch.randint(0, self.window, size=(n_mask, 2))
        for t, t_end in ts:
            if length - t <= 0:
                continue
            t_start = random.randrange(0, length - t)
            if t_start == t_start + t:
                continue
            t_end += t_start
            cloned[t_start:t_end] = 0
        return cloned


class VideoTransform:
    def __init__(self, subset, timesteps=1000, max_noise_level=None, center_crop_size=88):
        if subset == "train":
            self.video_pipeline = torch.nn.Sequential(
                # ----- diffusion noise ------ #
                FunctionalModule(lambda x: x.unsqueeze(0) / 255.0 * 2 - 1),
                ForwardDiffusion(timesteps=timesteps, max_noise_level=max_noise_level),
                FunctionalModule(lambda x: (((x[0] + 1) * 0.5).clip(0, 1) * 255.0, x[1])),
                FunctionalModule(lambda x: (x[0][0], x[1])),
                # ----- clean verion ------ #
                # FunctionalModule(lambda x: (x, 0)),
                # ------ after ------ #
                FunctionalModule(lambda x: (x[0] / 255.0, x[1])),
                FunctionalModule(lambda x: (torchvision.transforms.RandomCrop(center_crop_size)(x[0]), x[1])),
                FunctionalModule(lambda x: (torchvision.transforms.Grayscale()(x[0]), x[1])),
                FunctionalModule(lambda x: (AdaptiveTimeMask(10, 25)(x[0]), x[1])),
                # FunctionalModule(lambda x: x / 255.0),
                # torchvision.transforms.RandomCrop(88),
                # torchvision.transforms.Grayscale(),
                # AdaptiveTimeMask(10, 25),
                # torchvision.transforms.Normalize(0.421, 0.165),
            )
        elif subset == "val_w_noise":
            self.video_pipeline = torch.nn.Sequential(
                # ----- diffusion noise ------ #
                FunctionalModule(lambda x: x.unsqueeze(0) / 255.0 * 2 - 1),
                ForwardDiffusion(timesteps=timesteps, max_noise_level=max_noise_level),
                FunctionalModule(lambda x: (((x[0] + 1) * 0.5).clip(0, 1) * 255.0, x[1])),
                FunctionalModule(lambda x: (x[0][0], x[1])),
                # ----- clean verion ------ #
                # FunctionalModule(lambda x: (x, 0)),
                # ------ after ------ #
                FunctionalModule(lambda x: (x[0] / 255.0, x[1])),
                FunctionalModule(lambda x: (torchvision.transforms.CenterCrop(center_crop_size)(x[0]), x[1])),
                FunctionalModule(lambda x: (torchvision.transforms.Grayscale()(x[0]), x[1])),
                # FunctionalModule(lambda x: x / 255.0),
                # torchvision.transforms.RandomCrop(88),
                # torchvision.transforms.Grayscale(),
                # AdaptiveTimeMask(10, 25),
                # torchvision.transforms.Normalize(0.421, 0.165),
            )
        elif subset == "val" or subset == "test":
            self.video_pipeline = torch.nn.Sequential(
                FunctionalModule(lambda x: x / 255.0),
                torchvision.transforms.CenterCrop(center_crop_size),
                torchvision.transforms.Grayscale(),
                # torchvision.transforms.Normalize(0.421, 0.165),
            )
        elif subset == "loss":
            self.video_pipeline = torch.nn.Sequential(
                FunctionalModule(lambda x: (x + 1) / 2),
                torchvision.transforms.CenterCrop(center_crop_size),
                torchvision.transforms.Grayscale(),
                # torchvision.transforms.Normalize(0.421, 0.165),
            )

    def __call__(self, sample):
        # sample: T x C x H x W
        # rtype: T x 1 x H x W
        # print(f"[before transformation]: {sample.size()}")
        x = self.video_pipeline(sample)
        if isinstance(x, tuple):
            # print(f"x is a tuple")
            # print(f"before transformation: {sample.size()}")
            x, t = x
            # print(f"after transformation: {x.size()}")
            t = torch.tensor([t])
        else:
            t = torch.tensor([0])
        # print(f"[after transformation]: {x.size()}")
        assert x.size(0) == sample.size(
            0
        ), f"{sample.size()}/{x.size()} cannot get exactly same size before/after transformation."
        # print(f"{sample.size()}/{x.size()}")
        return x, t


class TextTransform:
    """Mapping Dictionary Class for SentencePiece tokenization."""

    def __init__(
        self,
        sp_model_path=SP_MODEL_PATH,
        dict_path=DICT_PATH,
    ):
        # Load SentencePiece model
        self.spm = sentencepiece.SentencePieceProcessor(model_file=sp_model_path)

        # Load units and create dictionary
        units = open(dict_path, encoding="utf8").read().splitlines()
        self.hashmap = {unit.split()[0]: unit.split()[-1] for unit in units}
        # 0 will be used for "blank" in CTC
        self.token_list = ["<blank>"] + list(self.hashmap.keys()) + ["<eos>"]
        self.ignore_id = -1

    def tokenize(self, text):
        tokens = self.spm.EncodeAsPieces(text)
        token_ids = [self.hashmap.get(token, self.hashmap["<unk>"]) for token in tokens]
        return torch.tensor(list(map(int, token_ids)))

    def post_process(self, token_ids):
        token_ids = token_ids[token_ids != -1]
        text = self._ids_to_str(token_ids, self.token_list)
        text = text.replace("\u2581", " ").strip()
        return text

    def _ids_to_str(self, token_ids, char_list):
        token_as_list = [char_list[idx] for idx in token_ids]
        return "".join(token_as_list).replace("<space>", " ")
