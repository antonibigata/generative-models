#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch
import torch.nn as nn
from sgm.modules.lipreader.espnet.nets.pytorch_backend.backbones.modules.resnet import BasicBlock, ResNet
from sgm.modules.lipreader.espnet.nets.pytorch_backend.transformer.convolution import Swish


def threeD_to_2D_tensor(x):
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch * s_time, n_channels, sx, sy)


class Conv3dResNet(nn.Module):
    """Conv3dResNet module"""

    def __init__(self, backbone_type="resnet", relu_type="swish"):
        """__init__.
        :param backbone_type: str, the type of a visual front-end.
        :param relu_type: str, activation function used in an audio front-end.
        """
        super(Conv3dResNet, self).__init__()

        self.backbone_type = backbone_type

        self.frontend_nout = 64
        self.trunk = ResNet(
            BasicBlock,
            [2, 2, 2, 2],
            relu_type=relu_type,
        )

        # -- frontend3D
        if relu_type == "relu":
            frontend_relu = nn.ReLU(True)
        elif relu_type == "prelu":
            frontend_relu = nn.PReLU(self.frontend_nout)
        elif relu_type == "swish":
            frontend_relu = nn.SiLU(inplace=True)

        self.frontend3D = nn.Sequential(
            nn.Conv3d(
                in_channels=1,
                out_channels=self.frontend_nout,
                kernel_size=(5, 7, 7),
                stride=(1, 2, 2),
                padding=(2, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(self.frontend_nout),
            frontend_relu,
            nn.MaxPool3d(
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
            ),
        )

    def forward(self, xs_pad, timestep_emb=None):
        """forward.
        :param xs_pad: torch.Tensor, batch of padded input sequences.
        """
        # -- include Channel dimension
        xs_pad = xs_pad.transpose(2, 1)  # [B, T, C, H, W] -> [B, C, T, H, W]
        B, C, T, H, W = xs_pad.size()
        xs_pad = self.frontend3D(xs_pad)
        Tnew = xs_pad.shape[2]  # outpu should be B x C2 x Tnew x H x W
        xs_pad = threeD_to_2D_tensor(xs_pad)
        xs_pad = self.trunk(xs_pad, timestep_emb=timestep_emb)
        xs_pad = xs_pad.view(B, Tnew, xs_pad.size(1))
        return xs_pad
