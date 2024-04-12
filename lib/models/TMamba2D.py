"""Module with DenseVNet"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.cuda.amp import autocast
import sys
try:
    from mamba_ssm import Mamba_FFT, Mamba
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
    print('succesfully import mamba_ssm')
except:
    pass

from timm.models.layers import DropPath
from torch import Tensor
from typing import Optional
from functools import partial
import math
from einops import rearrange, repeat
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# an alternative for mamba_ssm
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass

from .TMamba3D import AbsolutePositionalEncoder

class Tim_Block(nn.Module):
    def __init__(
        self, dim, fused_add_norm=False, residual_in_fp32=False,drop_path=0.,bimamba_type='v2',device=None, dtype=None, high_freq=0.9, low_freq=0.1
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.dim = dim
        factory_kwargs = {"device": device, "dtype": dtype}
        # ssm_cfg = {'d_state': 16,  # SSM state expansion factor
        #         'd_conv': 4,    # Local convolution width
        #         'expand': 2,    # Block expansion factor
        #     }   
        ssm_cfg = {}
        mixer_cls = partial(Mamba_FFT, layer_idx=None, bimamba_type=bimamba_type, high_freq=high_freq, low_freq=low_freq, **ssm_cfg, **factory_kwargs)
        norm_cls = partial(
            RMSNorm, eps=float(1e-5), **factory_kwargs
        )

        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, x: Tensor, learnable_positional_embed: nn.Parameter=None, num_block:int=-1, SpectralGatingBlocks: nn.Parameter=None, GateModules: list=None, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """

        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        hidden_states = x.reshape(B, C, n_tokens).transpose(-1, -2)
        
        # add resdual
        hidden_states_resdual = x.reshape(B, C, n_tokens).transpose(-1, -2) # (b, n_tokens, c)
        # add learnable_positional_embed
        device = hidden_states.device
        hidden_states = hidden_states + learnable_positional_embed.to(device)
        
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )    
        hidden_states = self.mixer(hidden_states, inference_params=inference_params, No_block=num_block, SpectralGatingBlocks=SpectralGatingBlocks, GateModules=GateModules)

        # add pos twice 
        hidden_states = hidden_states + learnable_positional_embed.to(device)
        
        # add residual
        hidden_states = hidden_states + hidden_states_resdual

        hidden_states = hidden_states.transpose(-1, -2).reshape(B, C, *img_dims)
        return hidden_states #, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

# U-Mamba
class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
    
    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)

        return out

class TMamba2D(nn.Module):
    def __init__(self, in_channels: int = 1, classes: int = 1, scaling_version="TINY", input_size: tuple = (640, 1280), high_freq: float = 0.9, low_freq: float = 0.1):
        super().__init__()

        self.model_name = "TMamba2D"
        self.classes = classes
        if scaling_version == "TINY":
            kernel_size = [5, 3, 3]
            num_downsample_channels = [24, 24, 24]
            num_skip_channels = [12, 24, 24]
            units = [5, 10, 10]
            growth_rate = [4, 8, 16]
        elif scaling_version == "SMALL":
            kernel_size = [5, 3, 3]
            num_downsample_channels = [48, 48, 48]
            num_skip_channels = [24, 48, 48]
            units = [5, 10, 10]
            growth_rate = [4, 8, 16]
        elif scaling_version == "WIDER":
            kernel_size = [5, 3, 3]
            num_downsample_channels = [36, 36, 36]
            num_skip_channels = [12, 24, 24]
            units = [5, 10, 10]
            growth_rate = [6, 12, 24]
        elif scaling_version == "BASE":
            kernel_size = [5, 3, 3]
            num_downsample_channels = [36, 36, 36]
            num_skip_channels = [12, 24, 24]
            units = [8, 16, 16]
            growth_rate = [6, 12, 24]
        elif scaling_version == "LARGE":
            kernel_size = [5, 3, 3]
            num_downsample_channels = [48, 48, 48]
            num_skip_channels = [24, 48, 48]
            units = [12, 24, 36]
            growth_rate = [8, 16, 32]
            
        else:
            raise RuntimeError(f"{scaling_version} scaling version is not available")
        W, H = input_size
        self.high_freq = high_freq
        self.low_freq = low_freq
        self.dfs_blocks = torch.nn.ModuleList()
        for i in range(3):
            self.dfs_blocks.append(
                DownsampleWithDfs2D(
                    in_channels=in_channels,
                    downsample_channels=num_downsample_channels[i],
                    skip_channels=num_skip_channels[i],
                    kernel_size=kernel_size[i],
                    units=units[i],
                    growth_rate=growth_rate[i],
                    high_freq=self.high_freq, 
                    low_freq=self.low_freq
                )
            )
            in_channels = num_downsample_channels[i] + units[i] * growth_rate[i]

        self.upsample_1 = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample_2 = torch.nn.Upsample(scale_factor=4, mode='bilinear')
        self.out_conv = ConvBlock(
            in_channels=sum(num_skip_channels),
            out_channels=self.classes,
            kernel_size=3,
            batch_norm=True,
            preactivation=True,
        )
        self.upsample_out = torch.nn.Upsample(scale_factor=2, mode='bilinear')

        self.learnable_positional_embed = [nn.Parameter(AbsolutePositionalEncoder(growth_rate[0], int(H*W/4))),
                                        nn.Parameter(AbsolutePositionalEncoder(growth_rate[1], int(H*W/16))),
                                        nn.Parameter(AbsolutePositionalEncoder(growth_rate[2], int(H*W/64)))]

        self.SpectralGatingBlocks = [nn.Parameter(torch.randn(growth_rate[0]*4, int(H*W/4), dtype=torch.float32) * 0.02),
                                    nn.Parameter(torch.randn(growth_rate[1]*4, int(H*W/16), dtype=torch.float32) * 0.02),
                                    nn.Parameter(torch.randn(growth_rate[2]*4, int(H*W/64), dtype=torch.float32) * 0.02)
                                ]
        self.GateModules = [[nn.Linear(2048, 512), nn.Linear(512, 256), nn.Linear(256, 128)], 
                        [nn.Linear(2048, 512), nn.Linear(512, 256), nn.Linear(256, 128)], 
                        [nn.Linear(2048, 512), nn.Linear(512, 256), nn.Linear(256, 128)]
                                ]

    def forward(self, x):
        x, skip_1 = self.dfs_blocks[0](x, self.learnable_positional_embed[0], 0, self.SpectralGatingBlocks[0], self.GateModules[0])
        x, skip_2 = self.dfs_blocks[1](x, self.learnable_positional_embed[1], 0, self.SpectralGatingBlocks[1], self.GateModules[1])
        _, skip_3 = self.dfs_blocks[2](x, self.learnable_positional_embed[2], 0, self.SpectralGatingBlocks[2], self.GateModules[2])
        
        skip_2 = self.upsample_1(skip_2)
        skip_3 = self.upsample_2(skip_3)

        out = self.out_conv(torch.cat([skip_1, skip_2, skip_3], 1))
        out = self.upsample_out(out)

        return out


class ConvBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        stride=1,
        batch_norm=True,
        preactivation=False,
    ):
        super().__init__()

        if dilation != 1:
            raise NotImplementedError()

        padding = kernel_size - stride
        if padding % 2 != 0:
            pad = torch.nn.ConstantPad2d(
                tuple([padding % 2, padding - padding % 2] * 2), 0
            )
        else:
            pad = torch.nn.ConstantPad2d(padding // 2, 0)

        if preactivation:
            layers = [
                torch.nn.ReLU(),
                pad,
                torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                ),
            ]
            if batch_norm:
                layers = [torch.nn.BatchNorm2d(in_channels)] + layers
        else:
            layers = [
                pad,
                torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                ),
            ]
            if batch_norm:
                layers.append(torch.nn.BatchNorm2d(out_channels))
            layers.append(torch.nn.ReLU())

        self.conv = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class DenseFeatureStack(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        units,
        growth_rate,
        kernel_size,
        dilation=1,
        batch_norm=True,
        batchwise_spatial_dropout=False,
        high_freq=0.9,
        low_freq=0.1
    ):
        super().__init__()

        self.units = torch.nn.ModuleList()
        self.Tim_Block = torch.nn.ModuleList() # add by hj
        for _ in range(units):
            if batchwise_spatial_dropout:
                raise NotImplementedError
            self.units.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=growth_rate,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    stride=1,
                    batch_norm=batch_norm,
                    preactivation=True,
                )
            )
            # self.mamba.append(MambaLayer(growth_rate)) # add by hj
            self.Tim_Block.append(Tim_Block(growth_rate, fused_add_norm=True, residual_in_fp32=True,\
                drop_path=0.,device='cuda', dtype=None, high_freq=high_freq, low_freq=low_freq
            ))
            # self.VMamba.append(SS2D_VMamba(growth_rate, d_conv=3))
            in_channels += growth_rate
            # self.mamba.append(MambaLayer(in_channels)) # add by hj

    def forward(self, x, learnable_positional_embed=None, no_block=-1, SpectralGatingBlocks=None, GateModules=None):
        feature_stack = [x]

        for i, unit in enumerate(self.units):
            inputs = torch.cat(feature_stack, 1)
            out = unit(inputs) # (b, c, w, h)
            out = self.Tim_Block[i](out, learnable_positional_embed, no_block, SpectralGatingBlocks, GateModules)
            feature_stack.append(out)

        return torch.cat(feature_stack, 1)


class DownsampleWithDfs2D(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        downsample_channels,
        skip_channels,
        kernel_size,
        units,
        growth_rate,
        high_freq,
        low_freq
    ):
        super().__init__()

        self.downsample = ConvBlock(
            in_channels=in_channels,
            out_channels=downsample_channels,
            kernel_size=kernel_size,
            stride=2,
            batch_norm=True,
            preactivation=True,
        )
        self.dfs = DenseFeatureStack(
            downsample_channels, units, growth_rate, 3, batch_norm=True, high_freq=high_freq, low_freq=low_freq
        )
        self.skip = ConvBlock(
            in_channels=downsample_channels + units * growth_rate,
            out_channels=skip_channels,
            kernel_size=3,
            batch_norm=True,
            preactivation=True,
        )
        # self.mamba = MambaLayer(downsample_channels + units * growth_rate) # add by hj

    def forward(self, x, learnable_positional_embed=None, no_block=-1, SpectralGatingBlocks=None, GateModules=None):
        x = self.downsample(x) # a 2d conv
        x = self.dfs(x, learnable_positional_embed, no_block, SpectralGatingBlocks, GateModules)
        # x = self.mamba(x) # add by hj
        x_skip = self.skip(x)

        return x, x_skip

def main():
    input_value = np.random.randn(1, 3, 640, 1280)
    input_value = torch.from_numpy(input_value).float().cuda()
    print(input_value.dtype)
    model = TMamba2D(3, 2).cuda()
    model.train()
    out = model(input_value)
    print(out.shape)

if __name__ == '__main__':
    main()
