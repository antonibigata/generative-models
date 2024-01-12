import torch
from torch import nn
import math
from einops import rearrange


def get_weight_value(weight_type, scaling, x):
    if weight_type in ["gate"]:
        scaling = torch.mean(torch.sigmoid(scaling(x)), dim=1).view(-1, 1, 1)
    elif weight_type in ["scale", "scale_channel"] or weight_type.startswith("scalar"):
        scaling = scaling
    else:
        scaling = None
    return scaling


def choose_weight_type(weight_type, dim):
    if weight_type == "gate":
        scaling = nn.Linear(dim, 1)
    elif weight_type == "scale":
        scaling = nn.Parameter(torch.Tensor(1))
        scaling.data.fill_(1)
    elif weight_type == "scale_channel":
        scaling = nn.Parameter(torch.Tensor(dim))
        scaling.data.fill_(1)
    elif weight_type and weight_type.startswith("scalar"):
        scaling = float(weight_type.split("_")[-1])
    else:
        scaling = None
    return scaling


class SCEAdapter(nn.Module):
    def __init__(
        self,
        dim,
        adapter_length,
        adapter_type=None,
        adapter_weight=None,
        act_layer=nn.GELU,
        zero_init_last=True,
        use_bias=True,
        adapt_on_time=False,
        condition_dim=None,
        condition_on=None,
    ):
        super(SCEAdapter, self).__init__()
        self.dim = dim
        self.adapter_length = adapter_length
        self.adapter_type = adapter_type
        self.adapter_weight = adapter_weight
        self.zero_init_last = zero_init_last
        self.adapt_on_time = adapt_on_time
        self.condition_dim = condition_dim
        self.condition_on = condition_on if condition_dim is not None else None

        if self.condition_on in ["both", "space"]:
            self.mlp_cond_space = nn.Sequential(nn.SiLU(), nn.Linear(condition_dim, adapter_length*2))
        if self.condition_on in ["both", "time"]:
            self.mlp_cond_time = nn.Sequential(nn.SiLU(), nn.Linear(condition_dim, adapter_length*2))

        self.ln1 = nn.Linear(dim, adapter_length, bias=use_bias)
        if isinstance(act_layer, str):
            act_layer = act_layer.lower()
            if act_layer == "gelu":
                act_layer = nn.GELU
            elif act_layer == "relu":
                act_layer = nn.ReLU
            elif act_layer == "leakyrelu":
                act_layer = nn.LeakyReLU
            elif act_layer == "silu":
                act_layer = nn.SiLU
            else:
                raise NotImplementedError(f"activation layer {act_layer} not implemented")

        self.activate = act_layer()
        self.ln2 = nn.Linear(adapter_length, dim, bias=use_bias)

        if adapt_on_time:
            self.time_ln1 = nn.Linear(dim, adapter_length, bias=use_bias)
            self.time_ln2 = nn.Linear(adapter_length, dim, bias=use_bias)

        self.init_weights()
        self.init_scaling()

    def _zero_init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def _kaiming_init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))

    def init_weights(self):
        self._kaiming_init_weights(self.ln1)
        if self.zero_init_last:
            self._zero_init_weights(self.ln2)
        else:
            self._kaiming_init_weights(self.ln2)

    def init_scaling(self):
        if self.adapter_weight:
            self.scaling = choose_weight_type(self.adapter_weight, self.dim)
        else:
            self.scaling = None

    def forward(self, x, x_shortcut=None, use_shortcut=True, n_frames=None, condition=None):
        # x shape is ((batch_size T), channels, height, width)
        # condition shape is (batch_size, T, condition_dim) if audio

        if x_shortcut is None:
            x_shortcut = x

        h, w = x.shape[-2:]
        out = rearrange(x, "b c h w -> b (h w) c")
        out = self.activate(self.ln1(out))
        if self.condition_on in ["both", "space"]:
            cond_space = self.mlp_cond_space(condition)
            cond_space = rearrange(cond_space, "b t c -> (b t) 1 c")
            scale, shift = cond_space.chunk(2, dim=-1)
            out = out * scale + shift
        out = self.ln2(out)
        out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w)

        if self.adapt_on_time:
            assert n_frames is not None, "n_frames must be provided if adapt_on_time is True"
            out = rearrange(out, "(b t) c h w -> (b h w) t c", t=n_frames)
            out = self.activate(self.time_ln1(out))
            if self.condition_on in ["both", "time"]:
                cond_time = self.mlp_cond_time(condition)
                cond_time = rearrange(cond_time, "b t c -> (b t) c 1 1")
                scale, shift = cond_time.chunk(2, dim=-1)
                out = rearrange(out, "(b h w) t c -> (b t) c h w", h=h, w=w)
                out = out * scale + shift
                out = rearrange(out, "(b t) c h w -> (b h w) t c", t=n_frames)
            out = self.time_ln2(out)
            out = rearrange(out, "(b h w) t c -> (b t) c h w", h=h, w=w)

        if self.adapter_weight:
            scaling = get_weight_value(self.adapter_weight, self.scaling, out)
            out = out * scaling if scaling is not None else out

        if use_shortcut:
            out = x_shortcut + out
        return out
