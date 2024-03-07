import torch
from torch import nn


class LoRA(nn.Module):
    def __init__(self, layer, name="weight", rank=16, alpha=1):
        super().__init__()
        weight = getattr(layer, name)
        # print(f"LoRA: {layer} {weight.size()}")
        self.lora_down = nn.Parameter(torch.zeros((rank, weight.size(1))))
        self.lora_up = nn.Parameter(torch.zeros((weight.size(0), rank)))
        nn.init.normal_(self.lora_up, mean=0, std=1)

        self.scale = alpha / rank
        self.enabled = True

    def forward(self, original_weights):
        if self.enabled:
            lora_shape = list(original_weights.shape[:2]) + [1] * (len(original_weights.shape) - 2)
            lora_weights = torch.matmul(self.lora_up.clone(), self.lora_down.clone()).view(*lora_shape) * self.scale
            return original_weights + lora_weights
        else:
            return original_weights


def get_module_names(model, filters=None, all_modules_in_filter=False):
    def check_parameter(module, name):
        return (
            hasattr(module, name)
            and not torch.nn.utils.parametrize.is_parametrized(module, name)
            and isinstance(getattr(module, name), nn.Parameter)
            # and not isinstance(getattr(module, name), nn.LayerNorm)
        )

    module_list = []
    prev_layer_num = ""
    for name, module in model.named_modules():
        layer_num = ".".join(name.split(".")[1:3])
        is_in_filter = any([f in name for f in filters])
        if is_in_filter:
            prev_layer_num = layer_num

        if all_modules_in_filter:
            is_in_filter = is_in_filter or prev_layer_num == layer_num
        # print(name, prev_layer_num, layer_num)
        if filters is None or is_in_filter:

            if check_parameter(module, "weight"):
                if isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
                    continue
                module_list.append(name)
            elif check_parameter(module, "in_proj_weight"):
                if isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
                    continue
                module_list.append(name)

    return module_list


def apply_lora(model, filters=None, rank=16, all_modules_in_filter=False):
    def check_parameter(module, name):
        return (
            hasattr(module, name)
            and not torch.nn.utils.parametrize.is_parametrized(module, name)
            and isinstance(getattr(module, name), nn.Parameter)
            # and not isinstance(getattr(module, name), nn.LayerNorm)
        )

    prev_layer_num = ""
    for name, module in model.named_modules():
        layer_num = ".".join(name.split(".")[1:3])
        is_in_filter = any([f in name for f in filters])
        if is_in_filter:
            prev_layer_num = layer_num

        if all_modules_in_filter:
            is_in_filter = is_in_filter or prev_layer_num == layer_num
        # print(name, prev_layer_num, layer_num)
        if filters is None or is_in_filter:

            if check_parameter(module, "weight"):
                if isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
                    continue
                device, dtype = module.weight.device, module.weight.dtype
                torch.nn.utils.parametrize.register_parametrization(
                    module, "weight", LoRA(module, "weight", rank=rank).to(dtype).to(device)
                )
            elif check_parameter(module, "in_proj_weight"):
                if isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
                    continue
                device, dtype = module.in_proj_weight.device, module.in_proj_weight.dtype
                torch.nn.utils.parametrize.register_parametrization(
                    module, "in_proj_weight", LoRA(module, "in_proj_weight", rank=rank).to(dtype).to(device)
                )


class ReToken(nn.Module):
    def __init__(self, indices=None):
        super().__init__()
        assert indices is not None
        self.embeddings = nn.Parameter(torch.zeros(len(indices), 1280))
        self.register_buffer("indices", torch.tensor(indices))
        self.enabled = True

    def forward(self, embeddings):
        if self.enabled:
            embeddings = embeddings.clone()
            for i, idx in enumerate(self.indices):
                embeddings[idx] += self.embeddings[i]
        return embeddings


def apply_retoken(module, indices=None):
    def check_parameter(module, name):
        return (
            hasattr(module, name)
            and not torch.nn.utils.parametrize.is_parametrized(module, name)
            and isinstance(getattr(module, name), nn.Parameter)
        )

    if check_parameter(module, "weight"):
        device, dtype = module.weight.device, module.weight.dtype
        torch.nn.utils.parametrize.register_parametrization(
            module, "weight", ReToken(indices=indices).to(dtype).to(device)
        )


def remove_lora(model, leave_parametrized=True):
    for module in model.modules():
        if torch.nn.utils.parametrize.is_parametrized(module, "weight"):
            nn.utils.parametrize.remove_parametrizations(module, "weight", leave_parametrized=leave_parametrized)
        elif torch.nn.utils.parametrize.is_parametrized(module, "in_proj_weight"):
            nn.utils.parametrize.remove_parametrizations(
                module, "in_proj_weight", leave_parametrized=leave_parametrized
            )
