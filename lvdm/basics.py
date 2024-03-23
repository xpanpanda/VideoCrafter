# adopted from
# https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
# and
# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
# and
# https://github.com/openai/guided-diffusion/blob/0ba878e517b276c45d1195eb29f6f5f72659a05b/guided_diffusion/nn.py
#
# thanks!

import torch.nn as nn
from utils.utils import instantiate_from_config


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    """这块代码的目标是为了强制关闭 PyTorch 模型的 .train() 方法，这样模型无论何时都会处于评估（evaluation）模式。
    也就是说，model.train() 不会切换模型到训练模式，会保持它在评估模式"""
    return self

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    """
     PyTorch 模型（或者模型的一个模块）的所有参数设为零
    """
    for p in module.parameters():
        # 忽略前面计算结果，原地替换所有参数值为0
        p.detach().zero_()
    return module

def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    """
     PyTorch 模型（或者模型的一个模块）的所有参数都乘以一个新的倍数
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def nonlinearity(type='silu'):
    # 两种激活函数可以选择
    if type == 'silu':
        return nn.SiLU()
    elif type == 'leaky_relu':
        return nn.LeakyReLU()


class GroupNormSpecific(nn.GroupNorm):
    # 组归一化->通道层面的归一化
    def forward(self, x):
        # 在forward前将数据类型转化为float
        return super().forward(x.float()).type(x.dtype)


def normalization(channels, num_groups=32):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    # 组归一化的normalization
    return GroupNormSpecific(num_groups, channels)


class HybridConditioner(nn.Module):

    def __init__(self, c_concat_config, c_crossattn_config):
        super().__init__()
        self.concat_conditioner = instantiate_from_config(c_concat_config)
        self.crossattn_conditioner = instantiate_from_config(c_crossattn_config)

    def forward(self, c_concat, c_crossattn):
        c_concat = self.concat_conditioner(c_concat)
        c_crossattn = self.crossattn_conditioner(c_crossattn)
        return {'c_concat': [c_concat], 'c_crossattn': [c_crossattn]}
