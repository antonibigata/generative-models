import torch
import torch.nn as nn
from packaging import version
from einops import repeat, rearrange
from diffusers.utils import _get_model_file
from diffusers.models.modeling_utils import load_state_dict
from ...modules.diffusionmodules.augment_pipeline import AugmentPipe

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
    def __init__(self, diffusion_model, compile_model: bool = False, ada_aug_percent=0.0):
        super().__init__(diffusion_model, compile_model)

        self.augment_pipe = None
        if ada_aug_percent > 0.0:
            augment_kwargs = dict(xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1)
            self.augment_pipe = AugmentPipe(ada_aug_percent, **augment_kwargs)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs) -> torch.Tensor:
        cond_cat = c.get("concat", torch.Tensor([]).type_as(x))
        if len(cond_cat.shape) and cond_cat.shape[0] and x.shape[0] != cond_cat.shape[0]:
            cond_cat = repeat(cond_cat, "b c h w -> b c t h w", t=x.shape[0] // cond_cat.shape[0])
            cond_cat = rearrange(cond_cat, "b c t h w -> (b t) c h w")
        x = torch.cat((x, cond_cat), dim=1)

        if self.augment_pipe is not None:
            x, labels = self.augment_pipe(x)
        else:
            labels = torch.zeros(x.shape[0], 9, device=x.device)

        # if c.get("reference", None) is not None:
        #     c["crossattn"] = c["reference"]

        return self.diffusion_model(
            x,
            timesteps=t,
            context=c.get("crossattn", None),
            reference_context=c.get("reference", None),
            y=c.get("vector", None),
            audio_emb=c.get("audio_emb", None),
            landmarks=c.get("landmarks", None),
            aug_labels=labels,
            **kwargs,
        )


class StabilityWrapper(IdentityWrapper):
    def __init__(
        self,
        diffusion_model,
        compile_model: bool = False,
        use_ipadapter: bool = False,
        ipadapter_model: str = "ip-adapter_sd15.bin",
        adapter_scale: float = 1.0,
        n_adapters: int = 1,
        skip_text_emb: bool = False,
        # pass_image_emb_to_hidden_states: bool = False,
    ):
        super().__init__(diffusion_model, compile_model)
        self.use_ipadapter = use_ipadapter
        # self.pass_image_emb_to_hidden_states = pass_image_emb_to_hidden_states

        if use_ipadapter:
            model_file = _get_model_file(
                "h94/IP-Adapter",
                weights_name=ipadapter_model,  # ip-adapter_sd15.bin
                # cache_dir="/vol/paramonos2/projects/antoni/.cache",
                subfolder="models",
            )
            state_dict = load_state_dict(model_file)
            state_dict = [load_state_dict(model_file)] * n_adapters
            print(f"Loading IP-Adapter weights from {model_file}")
            diffusion_model._load_ip_adapter_weights(state_dict)
            # diffusion_model.convert_ip_adapter_attn_to_diffusers_and_load(
            #     state_dict, skip_text_emb=skip_text_emb
            # )  # Custom method
            diffusion_model.set_ip_adapter_scale(adapter_scale)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs) -> torch.Tensor:
        added_cond_kwargs = None
        if self.use_ipadapter:
            added_cond_kwargs = {"image_embeds": c.get("image_embeds", None)}
            landmarks = c.get("landmarks", None)
            if landmarks is not None:
                added_cond_kwargs["image_embeds"] = [added_cond_kwargs["image_embeds"], landmarks]

        cond_cat = c.get("concat", torch.Tensor([]).type_as(x))
        if len(cond_cat.shape) and cond_cat.shape[0]:
            cond_cat = repeat(cond_cat, "b c h w -> b c t h w", t=x.shape[0] // cond_cat.shape[0])
            cond_cat = rearrange(cond_cat, "b c t h w -> (b t) c h w")
            x = torch.cat((x, cond_cat), dim=1)

        # if self.pass_image_emb_to_hidden_states:
        #     encoder_hidden_states = c.get("image_embeds", None)
        # else:
        # encoder_hidden_states = c.get("crossattn", None)

        return self.diffusion_model(
            x,
            t,
            encoder_hidden_states=c.get("crossattn", None),
            # y=c.get("vector", None),
            # audio_emb=c.get("audio_emb", None),
            # stability=self.stability,
            added_cond_kwargs=added_cond_kwargs,
            audio_emb=c.get("audio_emb", None),
            **kwargs,
        )[0]


class InterpolationWrapper(IdentityWrapper):
    def __init__(
        self,
        diffusion_model,
        compile_model: bool = False,
        im_size=[512, 512],
        n_channels=4,
        starting_mask_method="zeros",
        add_mask=True,
    ):
        super().__init__(diffusion_model, compile_model)
        im_size = [x // 8 for x in im_size]  # 8 is the default downscaling factor in the vae model
        if starting_mask_method == "zeros":
            self.learned_mask = nn.Parameter(torch.zeros(n_channels, im_size[0], im_size[1]))
        elif starting_mask_method == "ones":
            self.learned_mask = nn.Parameter(torch.ones(n_channels, im_size[0], im_size[1]))
        elif starting_mask_method == "random":
            self.learned_mask = nn.Parameter(torch.randn(n_channels, im_size[0], im_size[1]))
        elif starting_mask_method == "none":
            self.learned_mask = None
        elif starting_mask_method == "fixed_ones":
            self.learned_mask = torch.ones(n_channels, im_size[0], im_size[1])
        elif starting_mask_method == "fixed_zeros":
            self.learned_mask = torch.zeros(n_channels, im_size[0], im_size[1])
        else:
            raise NotImplementedError(f"Unknown stating_mask_method: {starting_mask_method}")

        self.add_mask = add_mask
        # self.zeros_mask = torch.zeros(n_channels, im_size[0], im_size[1])
        # self.ones_mask = torch.ones(n_channels, im_size[0], im_size[1])

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs) -> torch.Tensor:
        cond_cat = c.get("concat", torch.Tensor([]).type_as(x))
        cond_cat = rearrange(cond_cat, "b (t c) h w -> b c t h w", t=2)
        T = x.shape[0] // cond_cat.shape[0]
        start, end = cond_cat.chunk(2, dim=2)
        if self.learned_mask is None:
            learned_mask = torch.stack([start.squeeze(2)] * (T // 2 - 1) + [end.squeeze(2)] * (T // 2 - 1), dim=2)
        else:
            learned_mask = repeat(self.learned_mask.to(x.device), "c h w -> b c h w", b=cond_cat.shape[0])
        ones_mask = torch.ones_like(learned_mask)[:, 0].unsqueeze(1)
        zeros_mask = torch.zeros_like(learned_mask)[:, 0].unsqueeze(1)
        if self.learned_mask is None:
            cond_seq = torch.cat([start] + [learned_mask] + [end], dim=2)
        else:
            cond_seq = torch.stack([start.squeeze(2)] + [learned_mask] * (T - 2) + [end.squeeze(2)], dim=2)
        cond_seq = rearrange(cond_seq, "b c t h w -> (b t) c h w")
        x = torch.cat((x, cond_seq), dim=1)
        if self.add_mask:
            mask_seq = torch.stack([ones_mask] + [zeros_mask] * (T - 2) + [ones_mask], dim=2)
            mask_seq = rearrange(mask_seq, "b c t h w -> (b t) c h w")
            x = torch.cat((x, mask_seq), dim=1)

        return self.diffusion_model(
            x,
            timesteps=t,
            context=c.get("crossattn", None),
            y=c.get("vector", None),
            audio_emb=c.get("audio_emb", None),
            **kwargs,
        )
