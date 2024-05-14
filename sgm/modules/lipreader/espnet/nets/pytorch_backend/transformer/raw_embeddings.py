import logging

import torch
from espnet.nets.pytorch_backend.backbones.conv1d_extractor import Conv1dResNet

from espnet.nets.pytorch_backend.backbones.conv3d_extractor import Conv3dResNet
from espnet.nets.pytorch_backend.backbones.conv3d_extractor_timeconditioned import Conv3dResNetTimeConditioned
from espnet.nets.pytorch_backend.transformer.time_embedding import SinusoidalPosEmb


class VideoEmbedding(torch.nn.Module):
    """Video Embedding

    :param int idim: input dim
    :param int odim: output dim
    :param flaot dropout_rate: dropout rate
    """

    def __init__(
        self,
        idim,
        odim,
        dropout_rate,
        pos_enc_class,
        time_embedding=False,
        backbone_type="resnet",
        relu_type="prelu",
    ):
        super(VideoEmbedding, self).__init__()

        self.time_embedding = time_embedding

        if time_embedding:
            dim = 64
            self.to_timestep_cond = torch.nn.Sequential(SinusoidalPosEmb(dim), torch.nn.Linear(dim, 4 * dim), torch.nn.SiLU())
            self.trunk = Conv3dResNetTimeConditioned(backbone_type=backbone_type, relu_type=relu_type)
        else:
            self.trunk = Conv3dResNet(backbone_type=backbone_type, relu_type=relu_type)

        self.out = torch.nn.Sequential(
            torch.nn.Linear(idim, odim),
            pos_enc_class,
        )

    def forward(self, x, x_mask, t=None, extract_feats=None):
        """video embedding for x

        :param torch.Tensor x: input tensor
        :param torch.Tensor x_mask: input mask
        :param str extract_features: the position for feature extraction
        :return: subsampled x and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]
        """
        if self.time_embedding:
            t_emb = self.to_timestep_cond(t)
            x_resnet, x_mask = self.trunk(x, timestep_emb=t_emb)
        else:
            x_resnet, x_mask = self.trunk(x, x_mask)
        x = self.out(x_resnet)
        if extract_feats:
            return x, x_mask, x_resnet
        else:
            return x, x_mask


class AudioEmbedding(torch.nn.Module):
    """Audio Embedding

    :param int idim: input dim
    :param int odim: output dim
    :param flaot dropout_rate: dropout rate
    """

    def __init__(
        self,
        idim,
        odim,
        dropout_rate,
        pos_enc_class,
        relu_type="prelu",
        a_upsample_ratio=1,
    ):
        super(AudioEmbedding, self).__init__()
        self.trunk = Conv1dResNet(
            relu_type=relu_type,
            a_upsample_ratio=a_upsample_ratio,
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(idim, odim),
            pos_enc_class,
        )

    def forward(self, x, x_mask, extract_feats=None):
        """audio embedding for x

        :param torch.Tensor x: input tensor
        :param torch.Tensor x_mask: input mask
        :param str extract_features: the position for feature extraction
        :return: subsampled x and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]
        """
        x_resnet, x_mask = self.trunk(x, x_mask)
        x = self.out(x_resnet)
        if extract_feats:
            return x, x_mask, x_resnet
        else:
            return x, x_mask
