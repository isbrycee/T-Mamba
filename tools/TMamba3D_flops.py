"""Module with DenseVNet"""
import os
import torch
import torch.nn as nn
import numpy as np
import copy
from torch.cuda.amp import autocast
try:
    from mamba_ssm import Mamba
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
    print('succesfully import mamba_ssm')
except:
    pass
from mamba_ssm import Mamba_FFT
from timm.models.layers import DropPath
from torch import Tensor, device
from typing import Optional
from functools import partial

def AbsolutePositionalEncoder(emb_dim, max_position=512):
    position = torch.arange(max_position).unsqueeze(1)

    positional_encoding = torch.zeros(1, max_position, emb_dim)

    _2i = torch.arange(0, emb_dim, step=2).float()

    # PE(pos, 2i) = sin(pos/10000^(2i/d_model))
    positional_encoding[0, :, 0::2] = torch.sin(position / (10000 ** (_2i / emb_dim)))

    # PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
    positional_encoding[0, :, 1::2] = torch.cos(position / (10000 ** (_2i / emb_dim)))
    return positional_encoding

class Vim_Block(nn.Module):
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
        ssm_cfg = {'d_state': 16,  # SSM state expansion factor
                'd_conv': 4,    # Local convolution width
                'expand': 2,    # Block expansion factor
            }   
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
        self, x: Tensor, learnable_positional_embed: nn.Parameter=None, num_block:int=-1, SpectralGatingBlocks: nn.Parameter=None, GateModules: list=None, residual: Optional[Tensor] = None, inference_params=None,
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2] # x(b, c, w ,h depth)
        
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        hidden_states = x.reshape(B, C, n_tokens).transpose(-1, -2) # (b, n_tokens, c)
        
        # add resdual
        hidden_states_resdual = x.reshape(B, C, n_tokens).transpose(-1, -2) # (b, n_tokens, c)
        # add first learnable_positional_embed
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
        return hidden_states # , residual

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

class TMamba3D(nn.Module):
    def __init__(self, in_channels: int = 1, classes: int = 1, input_size: tuple = (160, 160, 96), high_freq: float = 0.9, low_freq: float = 0.1):
        super().__init__()
        self.model_name = "TMamba3D"
        self.classes = classes
        kernel_size = [5, 3, 3]
        num_downsample_channels = [24, 24, 24]
        num_skip_channels = [12, 24, 24]
        units = [5, 10, 10]
        growth_rate = [4, 8, 16]
        W, H, D = input_size
        self.high_freq = high_freq
        self.low_freq = low_freq
        self.dfs_blocks = torch.nn.ModuleList()
        self.learnable_positional_embed = [nn.Parameter(AbsolutePositionalEncoder(growth_rate[0], int(H*W*D/8))),
                                            nn.Parameter(AbsolutePositionalEncoder(growth_rate[1], int(H*W*D/64))),
                                            nn.Parameter(AbsolutePositionalEncoder(growth_rate[2], int(H*W*D/512)))]
        self.SpectralGatingBlocks = [nn.Parameter(torch.randn(16, int(H*W*D/8), dtype=torch.float32) * 0.02),
                                     nn.Parameter(torch.randn(32, int(H*W*D/64), dtype=torch.float32) * 0.02),
                                     nn.Parameter(torch.randn(64, int(H*W*D/512), dtype=torch.float32) * 0.02)
                                    ]
        self.GateModules = [[nn.Linear(2048, 512), nn.Linear(512, 256), nn.Linear(256, 128)], 
                            [nn.Linear(2048, 512), nn.Linear(512, 256), nn.Linear(256, 128)], 
                            [nn.Linear(2048, 512), nn.Linear(512, 256), nn.Linear(256, 128)]
                                    ]

        for i in range(3): # 3 scales in total
            self.dfs_blocks.append(
                DownsampleWithDfs(
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

        self.upsample_1 = torch.nn.Upsample(scale_factor=2, mode='trilinear')
        self.upsample_2 = torch.nn.Upsample(scale_factor=4, mode='trilinear')
        # self.fuse_mamba = MambaLayer(sum(num_skip_channels))
        self.out_conv = ConvBlock(
            in_channels=sum(num_skip_channels),
            out_channels=self.classes,
            kernel_size=3,
            batch_norm=True,
            preactivation=True,
        )
        self.upsample_out = torch.nn.Upsample(scale_factor=2, mode='trilinear')


    def forward(self, x):
        x, skip_1 = self.dfs_blocks[0](x, self.learnable_positional_embed[0], 0, self.SpectralGatingBlocks[0], self.GateModules[0])
        x, skip_2 = self.dfs_blocks[1](x, self.learnable_positional_embed[1], 1, self.SpectralGatingBlocks[1], self.GateModules[1])
        _, skip_3 = self.dfs_blocks[2](x, self.learnable_positional_embed[2], 2, self.SpectralGatingBlocks[2], self.GateModules[2])

        skip_2 = self.upsample_1(skip_2)
        skip_3 = self.upsample_2(skip_3)

        # print(skip_1.size(), skip_2.size(), skip_3.size())
        # out = self.out_conv(self.fuse_mamba(torch.cat([skip_1, skip_2, skip_3], 1))) # add by hj
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
            pad = torch.nn.ConstantPad3d(
                tuple([padding % 2, padding - padding % 2] * 3), 0
            )
        else:
            pad = torch.nn.ConstantPad3d(padding // 2, 0)

        if preactivation:
            layers = [
                torch.nn.ReLU(),
                pad,
                torch.nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                ),
            ]
            if batch_norm:
                layers = [torch.nn.BatchNorm3d(in_channels)] + layers
        else:
            layers = [
                pad,
                torch.nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                ),
            ]
            if batch_norm:
                layers.append(torch.nn.BatchNorm3d(out_channels))
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
        # self.mamba = torch.nn.ModuleList() # add by hj
        self.Vim_Block = torch.nn.ModuleList() # add by hj
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
            self.Vim_Block.append(Vim_Block(growth_rate, fused_add_norm=True, residual_in_fp32=True,\
                drop_path=0.,device='cuda', dtype=None, high_freq=high_freq, low_freq=low_freq
            ))
            in_channels += growth_rate
            # self.mamba.append(MambaLayer(in_channels)) # add by hj
            

    def forward(self, x, learnable_positional_embed=None, no_block=-1, SpectralGatingBlocks=None, GateModules=None):
        feature_stack = [x]

        for i, unit in enumerate(self.units):
            inputs = torch.cat(feature_stack, 1)
            # print(inputs.shape)
            out = unit(inputs) # (b, 4, 80, 80, 48)
            out = self.Vim_Block[i](out, learnable_positional_embed, no_block, SpectralGatingBlocks, GateModules) # add by hj
            # print(residual.shape)
            # out = self.mamba[i](out)
            feature_stack.append(out)

        return torch.cat(feature_stack, 1)


class DownsampleWithDfs(torch.nn.Module):
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
            downsample_channels, units, growth_rate, 3, batch_norm=True, high_freq=high_freq, low_freq=low_freq, 
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
        x = self.downsample(x) # a 3d conv
        x = self.dfs(x, learnable_positional_embed, no_block, SpectralGatingBlocks, GateModules)
        # x = self.mamba(x) # add by hj
        x_skip = self.skip(x)

        return x, x_skip



def count_parameters(model):
    # 获取模型中所有参数
    params = list(model.parameters())
    # 计算所有参数的数量
    num_params = sum(p.numel() for p in params)
    return num_params

class FlopCounter:
    def __init__(self):
        self.flops = 0

    def count_conv3d(self, in_channels, out_channels, kernel_size, input_shape, stride=1, groups=1):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        
        output_height = input_shape[0] // stride[0]
        output_width = input_shape[1] // stride[1]
        output_depth = input_shape[2] // stride[2]
        
        flops = 2 * output_height * output_width * output_depth * (in_channels // groups) * out_channels * kernel_size[0] * kernel_size[1] * kernel_size[2]
        self.flops += flops
        print(f"Conv3D: {flops:,} FLOPs")

    def count_bn(self, num_features, input_shape):
        flops = 2 * num_features * input_shape[0] * input_shape[1] * input_shape[2]
        self.flops += flops
        print(f"BatchNorm3D: {flops:,} FLOPs")

    def count_relu(self, input_shape):
        flops = input_shape[0] * input_shape[1] * input_shape[2]
        self.flops += flops
        print(f"ReLU: {flops:,} FLOPs")

    def count_mamba(self, dim, d_state, d_conv, expand, seq_len):
        expanded_dim = dim * expand
        flops = seq_len * (
            2 * dim * expanded_dim +  # Input projection
            2 * expanded_dim * d_state +  # SSM computation
            2 * expanded_dim * d_conv +  # Convolution
            2 * expanded_dim * dim  # Output projection
        )
        self.flops += flops
        print(f"Mamba: {flops:,} FLOPs")

    def count_upsample(self, scale_factor, input_shape):
        if isinstance(scale_factor, (int, float)):
            scale_factor = (scale_factor, scale_factor, scale_factor)
        output_elements = input_shape[0] * input_shape[1] * input_shape[2] * scale_factor[0] * scale_factor[1] * scale_factor[2]
        flops = output_elements  # Assuming 1 operation per output element for trilinear interpolation
        self.flops += flops
        print(f"Upsample: {flops:,} FLOPs")
def count_flops_tmamba3d(model, input_size):
    counter = FlopCounter()
    B, C, H, W, D = input_size

    for i, block in enumerate(model.dfs_blocks):
        print(f"\nProcessing DownsampleWithDfs block {i}")
        
        # Downsample
        conv_layer = next(layer for layer in block.downsample.conv if isinstance(layer, nn.Conv3d))
        counter.count_conv3d(conv_layer.in_channels, 
                             conv_layer.out_channels, 
                             conv_layer.kernel_size, 
                             (H // (2**i), W // (2**i), D // (2**i)),
                             stride=conv_layer.stride)
        
        bn_layer = next(layer for layer in block.downsample.conv if isinstance(layer, nn.BatchNorm3d))
        counter.count_bn(bn_layer.num_features, (H // (2**(i+1)), W // (2**(i+1)), D // (2**(i+1))))
        
        counter.count_relu((H // (2**(i+1)), W // (2**(i+1)), D // (2**(i+1))))
        
        # DenseFeatureStack
        for j, unit in enumerate(block.dfs.units):
            print(f"  Processing DFS unit {j}")
            bn_layer = next(layer for layer in unit.conv if isinstance(layer, nn.BatchNorm3d))
            counter.count_bn(bn_layer.num_features, (H // (2**(i+1)), W // (2**(i+1)), D // (2**(i+1))))
            counter.count_relu((H // (2**(i+1)), W // (2**(i+1)), D // (2**(i+1))))
            
            conv_layer = next(layer for layer in unit.conv if isinstance(layer, nn.Conv3d))
            counter.count_conv3d(conv_layer.in_channels, 
                                 conv_layer.out_channels, 
                                 conv_layer.kernel_size, 
                                 (H // (2**(i+1)), W // (2**(i+1)), D // (2**(i+1))))

        # Vim_Block (Mamba)
        for j, vim_block in enumerate(block.dfs.Vim_Block):
            print(f"  Processing Vim_Block {j}")
            counter.count_mamba(vim_block.dim, 16, 4, 2, (H * W * D) // (8 * (2**i)))
        
        # Skip connection
        conv_layer = next(layer for layer in block.skip.conv if isinstance(layer, nn.Conv3d))
        counter.count_conv3d(conv_layer.in_channels, 
                             conv_layer.out_channels, 
                             conv_layer.kernel_size, 
                             (H // (2**(i+1)), W // (2**(i+1)), D // (2**(i+1))))
        
        bn_layer = next(layer for layer in block.skip.conv if isinstance(layer, nn.BatchNorm3d))
        counter.count_bn(bn_layer.num_features, (H // (2**(i+1)), W // (2**(i+1)), D // (2**(i+1))))
        
        counter.count_relu((H // (2**(i+1)), W // (2**(i+1)), D // (2**(i+1))))


    # Upsample operations
    print("\nProcessing Upsample operations")
    if hasattr(model.upsample_1, 'scale_factor'):
        counter.count_upsample(model.upsample_1.scale_factor, (H//4, W//4, D//4))
    else:
        # If scale_factor is not available, we can infer it from the input and output sizes
        counter.count_upsample(2, (H//4, W//4, D//4))  # Assuming scale factor of 2
    
    if hasattr(model.upsample_2, 'scale_factor'):
        counter.count_upsample(model.upsample_2.scale_factor, (H//8, W//8, D//8))
    else:
        counter.count_upsample(4, (H//8, W//8, D//8))  # 

    # Final convolution
    print("\nProcessing Final Convolution")
    conv_layer = next(layer for layer in model.out_conv.conv if isinstance(layer, nn.Conv3d))
    counter.count_conv3d(conv_layer.in_channels, 
                         conv_layer.out_channels, 
                         conv_layer.kernel_size, 
                         (H//2, W//2, D//2))
    
    bn_layer = next(layer for layer in model.out_conv.conv if isinstance(layer, nn.BatchNorm3d))
    counter.count_bn(bn_layer.num_features, (H//2, W//2, D//2))
    
    counter.count_relu((H//2, W//2, D//2))


     # Final upsample
    print("\nProcessing Final Upsample")
    if hasattr(model.upsample_out, 'scale_factor'):
        counter.count_upsample(model.upsample_out.scale_factor, (H//2, W//2, D//2))
    else:
        counter.count_upsample(2, (H//2, W//2, D//2))  # Assuming scale factor of 2

    return counter.flops

def main():
    input_value = np.random.randn(1, 1, 160, 160, 96)
    input_value = torch.from_numpy(input_value).float().cuda()
    print(input_value.dtype)
    model = TMamba3D(1).cuda()
    model.train()

    out = model(input_value)
    print(out.shape)

    # 计算模型参数量
    num_params = count_parameters(model)
    print("模型参数量:", num_params)
    
    # 计算模型的浮点运算量 (FLOPs)
    input_size = (1, 1, 160, 160, 96)  # 示例输入大小为 (batch_size, channels, height, width)

    
    flops = count_flops_tmamba3d(model, input_size)
    print(f"\nTotal Estimated FLOPs: {flops:,}")
    print(f"Total Estimated GFLOPs: {flops / 1e9:.2f}")

if __name__ == '__main__':
    main()