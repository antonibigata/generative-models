import torch
import torch.nn as nn
from packaging import version
from einops import repeat, rearrange

OPENAIUNETWRAPPER = "sgm.modules.diffusionmodules.wrappers.OpenAIWrapper"


class IdentityWrapper(nn.Module):
    def __init__(self, diffusion_model, compile_model: bool = False):
        super().__init__()
        compile = (
            torch.compile
            if (version.parse(torch.__version__) >= version.parse("2.0.0")) and compile_model
            else lambda x: x
        )
        self.diffusion_model = compile(diffusion_model)

    def forward(self, *args, **kwargs):
        return self.diffusion_model(*args, **kwargs)


class OpenAIWrapper(IdentityWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs) -> torch.Tensor:
        cond_cat = c.get("concat", torch.Tensor([]).type_as(x))
        if len(cond_cat.shape) and x.shape[0] != cond_cat.shape[0]:
            cond_cat = repeat(cond_cat, "b c h w -> b c t h w", t=x.shape[0] // cond_cat.shape[0])
            cond_cat = rearrange(cond_cat, "b c t h w -> (b t) c h w")
        x = torch.cat((x, cond_cat), dim=1)
        return self.diffusion_model(
            x,
            timesteps=t,
            context=c.get("crossattn", None),
            y=c.get("vector", None),
            audio_emb=c.get("audio_emb", None),
            **kwargs,
        )


class InterpolationWrapper(IdentityWrapper):
    def __init__(
        self,
        diffusion_model,
        compile_model: bool = False,
        im_size=[512, 512],
        n_channels=4,
        starting_mask_method="zeros",
    ):
        super().__init__(diffusion_model, compile_model)
        im_size = [x // 8 for x in im_size]  # 8 is the default downscaling factor in the vae model
        if starting_mask_method == "zeros":
            self.learned_mask = nn.Parameter(torch.zeros(n_channels, im_size[0], im_size[1]))
        elif starting_mask_method == "ones":
            self.learned_mask = nn.Parameter(torch.ones(n_channels, im_size[0], im_size[1]))
        elif starting_mask_method == "random":
            self.learned_mask = nn.Parameter(torch.randn(n_channels, im_size[0], im_size[1]))
        else:
            raise NotImplementedError(f"Unknown stating_mask_method: {starting_mask_method}")

        # self.zeros_mask = torch.zeros(n_channels, im_size[0], im_size[1])
        # self.ones_mask = torch.ones(n_channels, im_size[0], im_size[1])

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs) -> torch.Tensor:
        cond_cat = c.get("concat", torch.Tensor([]).type_as(x))
        cond_cat = rearrange(cond_cat, "b (t c) h w -> b c t h w", t=2)
        T = x.shape[0] // cond_cat.shape[0]
        start, end = cond_cat.chunk(2, dim=2)
        learned_mask = repeat(self.learned_mask, "c h w -> b c h w", b=cond_cat.shape[0])
        ones_mask = torch.ones_like(learned_mask)[:, 0].unsqueeze(1)
        zeros_mask = torch.zeros_like(learned_mask)[:, 0].unsqueeze(1)
        cond_seq = torch.stack([start.squeeze(2)] + [learned_mask] * (T - 2) + [end.squeeze(2)], dim=2)
        cond_seq = rearrange(cond_seq, "b c t h w -> (b t) c h w")
        mask_seq = torch.stack([ones_mask] + [zeros_mask] * (T - 2) + [ones_mask], dim=2)
        mask_seq = rearrange(mask_seq, "b c t h w -> (b t) c h w")
        x = torch.cat((x, cond_seq, mask_seq), dim=1)
        return self.diffusion_model(
            x,
            timesteps=t,
            context=c.get("crossattn", None),
            y=c.get("vector", None),
            audio_emb=c.get("audio_emb", None),
            **kwargs,
        )
