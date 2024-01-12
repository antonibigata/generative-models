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
