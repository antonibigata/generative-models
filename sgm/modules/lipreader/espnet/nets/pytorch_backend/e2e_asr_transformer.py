# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer speech recognition model (pytorch)."""

import logging
import math
from argparse import Namespace
from distutils.util import strtobool

import numpy
import torch

from sgm.modules.lipreader.espnet.nets.ctc_prefix_score import CTCPrefixScore
from sgm.modules.lipreader.espnet.nets.e2e_asr_common import end_detect, ErrorCalculator
from sgm.modules.lipreader.espnet.nets.pytorch_backend.ctc import CTC
from sgm.modules.lipreader.espnet.nets.pytorch_backend.nets_utils import (
    get_subsample,
    make_non_pad_mask,
    th_accuracy,
)
from sgm.modules.lipreader.espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from sgm.modules.lipreader.espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
)
from sgm.modules.lipreader.espnet.nets.pytorch_backend.transformer.decoder import Decoder
from sgm.modules.lipreader.espnet.nets.pytorch_backend.transformer.encoder import Encoder
from sgm.modules.lipreader.espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from sgm.modules.lipreader.espnet.nets.pytorch_backend.transformer.mask import target_mask
from sgm.modules.lipreader.espnet.nets.scorers.ctc import CTCPrefixScorer


class E2E(torch.nn.Module):
    @property
    def attention_plot_class(self):
        """Return PlotAttentionReport."""
        return PlotAttentionReport

    def __init__(self, odim, ignore_id=-1):
        """Construct an E2E object.
        :param int odim: dimension of outputs
        """
        torch.nn.Module.__init__(self)
        idim = 80

        self.encoder = Encoder(
            idim=idim,
            attention_dim=768,
            attention_heads=12,
            linear_units=3072,
            num_blocks=12,
            input_layer="conv3d",
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            attention_dropout_rate=0.1,
            encoder_attn_layer_type="rel_mha",
            macaron_style=True,
            use_cnn_module=True,
            cnn_module_kernel=31,
            zero_triu=False,
            a_upsample_ratio=1,
            relu_type="swish",
            time_embedding=True,
        )

        self.time_embedding = True

        self.decoder = Decoder(
            odim=odim,
            attention_dim=768,
            attention_heads=12,
            linear_units=3072,
            num_blocks=6,
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            self_attention_dropout_rate=0.1,
            src_attention_dropout_rate=0.1,
        )
        self.blank = 0
        self.sos = odim - 1
        self.eos = odim - 1
        self.odim = odim
        self.ignore_id = ignore_id

        # self.lsm_weight = a
        self.criterion = LabelSmoothingLoss(self.odim, self.ignore_id, 0.1, False)

        self.ctc = CTC(odim, 768, 0.1, ctc_type="builtin", reduce=True)

    def scorers(self):
        """Scorers."""
        return dict(decoder=self.decoder, ctc=CTCPrefixScorer(self.ctc, self.eos))

    def forward(self, x, lengths, label, timesteps=None):
        if self.transformer_input_layer == "conv1d":
            lengths = torch.div(lengths, 640, rounding_mode="trunc")
        padding_mask = make_non_pad_mask(lengths).to(x.device).unsqueeze(-2)
        if self.time_embedding:
            x, _ = self.encoder(x, padding_mask, t=timesteps)
        else:
            x, _ = self.encoder(x, padding_mask)

        # ctc loss
        loss_ctc, ys_hat = self.ctc(x, lengths, label)

        if self.proj_decoder:
            x = self.proj_decoder(x)

        # decoder loss
        ys_in_pad, ys_out_pad = add_sos_eos(label, self.sos, self.eos, self.ignore_id)
        ys_mask = target_mask(ys_in_pad, self.ignore_id)
        pred_pad, _ = self.decoder(ys_in_pad, ys_mask, x, padding_mask)
        loss_att = self.criterion(pred_pad, ys_out_pad)
        loss = self.mtlalpha * loss_ctc + (1 - self.mtlalpha) * loss_att

        acc = th_accuracy(pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id)

        return loss, loss_ctc, loss_att, acc
