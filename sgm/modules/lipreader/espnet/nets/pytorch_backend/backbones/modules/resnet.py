import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    """conv3x3.
    :param in_planes: int, number of channels in the input sequence.
    :param out_planes: int,  number of channels produced by the convolution.
    :param stride: int, size of the convolving kernel.
    """
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def downsample_basic_block(inplanes, outplanes, stride):
    """downsample_basic_block.
    :param inplanes: int, number of channels in the input sequence.
    :param outplanes: int, number of channels produced by the convolution.
    :param stride: int, size of the convolving kernel.
    """
    return  nn.Sequential(
        nn.Conv2d(
            inplanes,
            outplanes,
            kernel_size=1,
            stride=stride,
            bias=False,
            ),
        nn.BatchNorm2d(outplanes),
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        timestep_cond_dim=None,
        relu_type="swish",
    ):
        """__init__.
        :param inplanes: int, number of channels in the input sequence.
        :param planes: int,  number of channels produced by the convolution.
        :param stride: int, size of the convolving kernel.
        :param downsample: boolean, if True, the temporal resolution is downsampled.
        :param relu_type: str, type of activation function.
        """
        super(BasicBlock, self).__init__()

        assert relu_type in ["relu", "prelu", "swish"]

        self.timestep_mlp = None
        if timestep_cond_dim is not None:
          self.timestep_mlp = nn.Sequential(nn.SiLU(), nn.Linear(timestep_cond_dim, planes * 2))

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)

        if relu_type == "relu":
            self.relu1 = nn.ReLU(inplace=True)
            self.relu2 = nn.ReLU(inplace=True)
        elif relu_type == "prelu":
            self.relu1 = nn.PReLU(num_parameters=planes)
            self.relu2 = nn.PReLU(num_parameters=planes)
        elif relu_type == "swish":
            self.relu1 = nn.SiLU(inplace=True)
            self.relu2 = nn.SiLU(inplace=True)
        else:
            raise NotImplementedError
        # --------

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x, timestep_emb=None):
        """forward.
        :param x: torch.Tensor, input tensor with input size (B, C, T, H, W).
        """
        scale, shift = None, None
        if self.timestep_mlp is not None and exists(timestep_emb):
            time_emb = self.timestep_mlp(timestep_emb)
            to_einsum_eq = "b c " + "1 "*(x.ndim-2)
            time_emb = rearrange(time_emb, f"b c -> {to_einsum_eq}")
            scale, shift = time_emb.chunk(2, dim=1)

        residual = x
        out = self.conv1(x)
        out = self.bn1(out)

        if scale is not None:
          out = out * (scale + 1) + shift

        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)

        return out

class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

class ResNet(nn.Module):

    def __init__(
        self,
        block,
        layers,
        relu_type="swish",
    ):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.relu_type = relu_type
        self.downsample_block = downsample_basic_block

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)


    def _make_layer(self, block, planes, blocks, stride=1):
        """_make_layer.
        :param block: torch.nn.Module, class of blocks.
        :param planes: int,  number of channels produced by the convolution.
        :param blocks: int, number of layers in a block.
        :param stride: int, size of the convolving kernel.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = self.downsample_block(
                inplanes=self.inplanes,
                outplanes=planes*block.expansion,
                stride=stride,
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                relu_type=self.relu_type,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    relu_type=self.relu_type,
                )
            )

        return mySequential(*layers)

    def forward(self, x, timestep_emb=None):
        """forward.
        :param x: torch.Tensor, input tensor with input size (B, C, T, H, W).
        """
        x = self.layer1(x, timestep_emb)
        x = self.layer2(x, timestep_emb)
        x = self.layer3(x, timestep_emb)
        x = self.layer4(x, timestep_emb)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
