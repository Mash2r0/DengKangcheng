# Add your custom network here
from .default import DRNet, R3LiteNet, GatedR3LiteNet
import torch.nn as nn


def basenet(in_channels, out_channels, **kwargs):
    return DRNet(in_channels, out_channels, 256, 13, norm=None, res_scale=0.1, bottom_kernel_size=1, **kwargs)


def errnet(in_channels, out_channels, **kwargs):
    return DRNet(in_channels, out_channels, 256, 13, norm=None, res_scale=0.1, se_reduction=8, bottom_kernel_size=1, pyramid=True, **kwargs)


def errnet_r3lite(in_channels, out_channels, **kwargs):
    return R3LiteNet(in_channels, 256, 13, norm=None, res_scale=0.1, se_reduction=8, bottom_kernel_size=1, pyramid=True, **kwargs)


def errnet_r3lite_gated(in_channels, out_channels, **kwargs):
    return GatedR3LiteNet(in_channels, 256, 13, norm=None, res_scale=0.1, se_reduction=8, bottom_kernel_size=1, pyramid=True, **kwargs)
