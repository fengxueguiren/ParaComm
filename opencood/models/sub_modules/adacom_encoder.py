import torch
import torch.nn as nn
from timm.layers import DropPath, LayerNorm
import torch.nn.functional as F

"""
ConvNeXt 
"""
class ConvNeXtBlock(nn.Module):
    r""" 
    https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py

    ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, kernel_size=7, deform=False):
        super().__init__()
        self.deform = deform
        if self.deform:
            from mmcv.ops import DeformConv2dPack as dconv2d
            self.dfconv = dconv2d(dim, dim, kernel_size=3, padding=1)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        if self.deform:
            x = self.dfconv(x)
            x = self.act(x)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    def __init__(self, args):
        super().__init__()
        dim = args['dim']
        kernel_size = args.get("kernel_size", 7)
        num_of_blocks = args['num_of_blocks']
        deform = args.get('deform', False)
        model_list = nn.ModuleList()
        for _ in range(num_of_blocks):
            model_list.append(ConvNeXtBlock(dim, kernel_size=kernel_size, deform=deform))
 
        self.model = nn.Sequential(*model_list)

    def forward(self, x):
        return self.model(x)
