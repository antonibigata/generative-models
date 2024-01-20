import torch
from einops import rearrange
from collections import defaultdict


# Merge B and T dimensions of specified tensor
def collate_video(data, merge_keys=[]):
    data_list = defaultdict(list)
    for item in data:
        for k, v in item.items():
            data_list[k].append(v)

    for k in data_list.keys():
        if k in merge_keys:
            data_list[k] = rearrange(torch.stack(data_list[k]), "b c t ... -> (b t) c ...")
        elif isinstance(data_list[k][0], torch.Tensor):
            data_list[k] = torch.stack(data_list[k])
        else:
            if k == "__key__":
                continue
            data_list[k] = torch.as_tensor(data_list[k])
    return data_list
