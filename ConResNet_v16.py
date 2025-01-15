
import torch.nn as nn
from torch.nn import functional as F
import torch
import numpy as np
from torchsummary import summary
########################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_,to_3tuple, DropPath
from functools import partial

class IRB(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, ksize=3, act_layer=nn.Hardswish, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv3d(in_features, hidden_features, 1, 1, 0)
        self.act = act_layer()
        self.conv = nn.Conv3d(hidden_features, hidden_features, kernel_size=ksize, padding=ksize//2, stride=1, groups=hidden_features)
        self.fc2 = nn.Conv3d(hidden_features, out_features, 1, 1, 0)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x, D, H, W):
        # B, C, D_, H_, W_ = x.shape
        B, N, C = x.shape
        # assert D == D_, f"Input depth {D_} should match expected depth {D}"
        x = x.permute(0,2,1).reshape(B, C, D, H, W)
        x = self.fc1(x)
        x = self.act(x)
        x = self.conv(x)
        x = self.act(x)
        x = self.fc2(x)
        return x.reshape(B, C, -1).permute(0,2,1)

class PoolingAttention(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, attn_drop=0., proj_drop=0., 
        pool_ratios=[1,2,3,6]):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.num_elements = np.array([t*t*t for t in pool_ratios]).sum()
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias))
        self.kv = nn.Sequential(nn.Linear(dim, dim * 2, bias=qkv_bias))
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pool_ratios = pool_ratios
        self.pools = nn.ModuleList()
        
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, D, H, W, d_convs=None):
        # B, C, D_, H_, W_ = x.shape       
        # assert D == D_, f"Input depth {D_} should match expected depth {D}"
        B, N, C = x.shape   #1,75,256
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        pools = []
        x_ = x.permute(0, 2, 1).reshape(B, C, D, H, W)
        for (pool_ratio, l) in zip(self.pool_ratios, d_convs):
            pool = F.adaptive_avg_pool3d(x_, (round(D/pool_ratio), round(H/pool_ratio), round(W/pool_ratio)))
            pool = pool + l(pool)
            pools.append(pool.view(B, C, -1))        
        pools = torch.cat(pools, dim=2)
        pools = self.norm(pools.permute(0,2,1))
        
        kv = self.kv(pools).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v)   
        x = x.transpose(1,2).contiguous().reshape(B, N, C)
        
        x = self.proj(x)
        return x#.reshape(B, C, D_, H_, W_)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_ratios=[12,16,20,24]):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PoolingAttention(
                    dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, pool_ratios=pool_ratios)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = IRB(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.Hardswish, drop=drop, ksize=3)

    def forward(self, x, D, H, W, d_convs=None):
        x = x + self.drop_path(self.attn(self.norm1(x), D, H, W, d_convs=d_convs))
        x = x + self.drop_path(self.mlp(self.norm2(x), D, H, W))
        return x

class PatchEmbed3D(nn.Module):
    """ Volume to Patch Embedding
    """
    def __init__(self, vol_size=(32, 32, 32), patch_size=(2, 2, 2), kernel_size=3, in_chans=1, embed_dim=768, overlap=True):
        super().__init__()
        vol_size = to_3tuple(vol_size)
        patch_size = to_3tuple(patch_size)

        self.vol_size = vol_size
        self.patch_size = patch_size
        assert vol_size[0] % patch_size[0] == 0 and vol_size[1] % patch_size[1] == 0 and vol_size[2] % patch_size[2] == 0, \
            f"vol_size {vol_size} should be divisible by patch_size {patch_size}."
        self.D, self.H, self.W = vol_size[0] // patch_size[0], vol_size[1] // patch_size[1], vol_size[2] // patch_size[2]
        self.num_patches = self.D * self.H * self.W
        if not overlap:
            self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        else:
            self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size, padding=kernel_size//2)
        
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        D, H, W = D // self.patch_size[0], H // self.patch_size[1], W // self.patch_size[2]
        return x, (D, H, W)

class PyramidPoolingTransformer3D(nn.Module):
    # def __init__(self, vol_size=(80,160,160), patch_size=2, in_chans=1, num_classes=1000, embed_dims=[32, 64, 128, 256],
    #              num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
    #              attn_drop_rate=0., drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 9, 3], **kwargs):
    def __init__(self, vol_size=(80,160,160), patch_size=2, in_chans=1, num_classes=1000, embed_dims=[64, 128, 256],
                 num_heads=[1, 2, 4], mlp_ratios=[8, 8, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 9], **kwargs):
        super().__init__()
        print("loading p2t")
        self.num_classes = num_classes
        self.depths = depths

        self.embed_dims = embed_dims

        # pyramid pooling ratios for each stage
        pool_ratios = [[12,16,20,24], [6,8,10,12], [3,4,5,6], [1,2,3,4]]

        self.patch_embed1 = PatchEmbed3D(vol_size=vol_size, patch_size=patch_size, kernel_size=7, in_chans=in_chans,
                                         embed_dim=embed_dims[0], overlap=True)

        self.patch_embed2 = PatchEmbed3D(vol_size=tuple(np.array(vol_size) // 2), patch_size=2, in_chans=embed_dims[0],
                                         embed_dim=embed_dims[1], overlap=True)
        self.patch_embed3 = PatchEmbed3D(vol_size=tuple(np.array(vol_size) // 4), patch_size=2, in_chans=embed_dims[1],
                                         embed_dim=embed_dims[2], overlap=True)
        # self.patch_embed4 = PatchEmbed3D(vol_size=tuple(np.array(vol_size) // 8), patch_size=2, in_chans=embed_dims[2],
        #                                  embed_dim=embed_dims[3], overlap=True)

        self.d_convs1 = nn.ModuleList([nn.Conv3d(embed_dims[0], embed_dims[0], kernel_size=3, stride=1, padding=1, groups=embed_dims[0]) for temp in pool_ratios[0]])
        self.d_convs2 = nn.ModuleList([nn.Conv3d(embed_dims[1], embed_dims[1], kernel_size=3, stride=1, padding=1, groups=embed_dims[1]) for temp in pool_ratios[1]])
        self.d_convs3 = nn.ModuleList([nn.Conv3d(embed_dims[2], embed_dims[2], kernel_size=3, stride=1, padding=1, groups=embed_dims[2]) for temp in pool_ratios[2]])
        # self.d_convs4 = nn.ModuleList([nn.Conv3d(embed_dims[3], embed_dims[3], kernel_size=3, stride=1, padding=1, groups=embed_dims[3]) for temp in pool_ratios[3]])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        ksize = 3

        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, pool_ratios=pool_ratios[0])
            for i in range(depths[0])])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, pool_ratios=pool_ratios[1])
            for i in range(depths[1])])
        cur += depths[1]

        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate,drop_path=dpr[cur + i], norm_layer=norm_layer, pool_ratios=pool_ratios[2])
            for i in range(depths[2])])
        cur += depths[2]

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        outs = []
        B = x.shape[0]

        # stage 1
        x, (D, H, W) = self.patch_embed1(x)
        for idx, blk in enumerate(self.block1):
            x = blk(x, D, H, W, self.d_convs1)
        x = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3)
        outs.append(x)

        # stage 2
        x, (D, H, W) = self.patch_embed2(x)
        for idx, blk in enumerate(self.block2):
            x = blk(x, D, H, W, self.d_convs2)
        x = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3)
        outs.append(x)

        x, (D, H, W) = self.patch_embed3(x)
        for idx, blk in enumerate(self.block3):
            x = blk(x, D, H, W, self.d_convs3)
        x = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3)
        outs.append(x)

        # stage 4
        # x, (D, H, W) = self.patch_embed4(x)
        # for idx, blk in enumerate(self.block4):
        #     x = blk(x, D, H, W, self.d_convs4)
        # x = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3)
        # outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        # classification head, usually not used in dense prediction tasks
        # x = self.gap(x[-1]).reshape(x[0].shape[0], -1)
        # x = self.head(x)
        return x
########################################################################################
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            # print(self.weight.size())
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]

            return x


class ux_block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, norm_layer=nn.BatchNorm3d):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=5, padding=2, groups=dim) #depthwise
        # self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim) #depthwise
        self.norm = nn.GroupNorm(8, dim)
        self.pwconv1 = nn.Conv3d(dim, 4 * dim, kernel_size=1, groups=dim) #pointwise
        self.act = nn.Hardswish(inplace=True)
        self.pwconv2 = nn.Conv3d(4 * dim, dim, kernel_size=1, groups=dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((1, dim,1,1,1)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        # x = x.permute(0, 2, 3, 4, 1) # (N, C, H, W, D) -> (N, H, W, D, C)
        x = self.norm(x)

        # x = x .permute(0, 4, 1, 2, 3)# (N, H, W, D, C) -> (N, C, H, W, D)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        # x = x.permute(0, 2, 3, 4, 1)    # (N, C, H, W, D) -> (N, H, W, D, C)
        if self.gamma is not None:
            x = self.gamma * x

        # x = x.permute(0, 4, 1, 2, 3)    # (N, H, W, D, C) -> (N, C, H, W, D)
        x = input + self.drop_path(x)
        return x



class uxnet_conv(nn.Module):
    """
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    # def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
    #              drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
    def __init__(self, in_chans=32, depths=[2, 2, 2], dims=[64, 128, 256],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2]):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        # stem = nn.Sequential(
        #     nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
        #     LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        # )
        stem = nn.Sequential(
              nn.Conv3d(in_chans, dims[0], kernel_size=5, stride=2, padding=2), #ori's stride=2
              LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
              )
        self.downsample_layers.append(stem)
        for i in range(2):  #3
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv3d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(3):  #4
            stage = nn.Sequential(
                *[ux_block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.out_indices = out_indices

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(3):    #4
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        # self.apply(self._init_weights)

    def forward_features(self, x):
        outs = []
        for i in range(3):  #4
            # print(i)
            # print(x.size())
            x = self.downsample_layers[i](x)
            # print(x.size())
            x = self.stages[i](x)
            # print(x.size())
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x

########################################################################################
class Conv3d(nn.Conv3d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1), groups=1, bias=False):
        super(Conv3d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True)
        weight = weight - weight_mean
        std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12).view(-1, 1, 1, 1, 1)
        weight = weight / std.expand_as(weight)
        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

def conv3x3x3(in_planes, out_planes, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1,1,1), dilation=(1,1,1), bias=False,
              weight_std=False):
    "3x3x3 convolution with padding"
    if weight_std:
        return Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      bias=bias)
    else:
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                         dilation=dilation, bias=bias)
    
# Replace conv3x3x3 with 3D UX-Net Block


class ConResAtt(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1),
                 dilation=(1, 1, 1), bias=False, weight_std=False, first_layer=False):
        super(ConResAtt, self).__init__()
        self.weight_std = weight_std
        self.stride = stride
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.first_layer = first_layer

        self.hardswish = nn.Hardswish(inplace=True)
        self.gn_seg = nn.GroupNorm(8, in_planes)
        self.conv_seg = conv3x3x3(in_planes, out_planes, kernel_size=(kernel_size[0], kernel_size[1], kernel_size[2]),
                               stride=(stride[0], stride[1], stride[2]), padding=(padding[0], padding[1], padding[2]),
                               dilation=(dilation[0], dilation[1], dilation[2]), bias=bias, weight_std=self.weight_std)

        self.gn_res = nn.GroupNorm(8, out_planes)
        self.conv_res = conv3x3x3(out_planes, out_planes, kernel_size=(1,1,1),
                               stride=(1, 1, 1), padding=(0,0,0),
                               dilation=(dilation[0], dilation[1], dilation[2]), bias=bias, weight_std=self.weight_std)

        self.gn_res1 = nn.GroupNorm(8, out_planes)
        self.conv_res1 = conv3x3x3(out_planes, out_planes, kernel_size=(kernel_size[0], kernel_size[1], kernel_size[2]),
                                stride=(1, 1, 1), padding=(padding[0], padding[1], padding[2]),
                                dilation=(dilation[0], dilation[1], dilation[2]), bias=bias, weight_std=self.weight_std)
        self.gn_res2 = nn.GroupNorm(8, out_planes)
        self.conv_res2 = conv3x3x3(out_planes, out_planes, kernel_size=(kernel_size[0], kernel_size[1], kernel_size[2]),
                                stride=(1, 1, 1), padding=(padding[0], padding[1], padding[2]),
                                dilation=(dilation[0], dilation[1], dilation[2]), bias=bias, weight_std=self.weight_std)

        self.gn_mp = nn.GroupNorm(8, in_planes)
        self.conv_mp_first = conv3x3x3(1, out_planes, kernel_size=(kernel_size[0], kernel_size[1], kernel_size[2]),
                              stride=(stride[0], stride[1], stride[2]), padding=(padding[0], padding[1], padding[2]),
                              dilation=(dilation[0], dilation[1], dilation[2]), bias=bias, weight_std=self.weight_std)
        self.conv_mp = conv3x3x3(in_planes, out_planes, kernel_size=(kernel_size[0], kernel_size[1], kernel_size[2]),
                               stride=(stride[0], stride[1], stride[2]), padding=(padding[0], padding[1], padding[2]),
                               dilation=(dilation[0], dilation[1], dilation[2]), bias=bias, weight_std=self.weight_std)

    # def _res(self, x):  # bs, channel, D, W, H

    #     bs, channel, depth, heigt, width = x.shape
    #     x_copy = torch.zeros_like(x).cuda()
    #     x_copy[:, :, 1:, :, :] = x[:, :, 0: depth - 1, :, :]
    #     res = x - x_copy
    #     res[:, :, 0, :, :] = 0
    #     res = torch.abs(res)
    #     return res
    def _res(self, x):
        '''
        Improved considering residual between current slice and adjacent slices
        '''
        bs, channel, depth, height, width = x.shape
        res = torch.zeros_like(x).cuda()
        
        # 计算当前切片与前一个切片的残差
        res[:, :, 1:, :, :] += torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :])
        # 计算当前切片与后一个切片的残差
        res[:, :, :-1, :, :] += torch.abs(x[:, :, :-1, :, :] - x[:, :, 1:, :, :])
        return res


    def forward(self, input):
        x1, x2 = input
        x1 = self.gn_seg(x1)
        # x1 = self.relu(x1)
        x1 = self.hardswish(x1)
        x1 = self.conv_seg(x1)

        res = torch.sigmoid(x1)
        res = self._res(res)
        # print(res.size())
        res = self.conv_res(res)
        if self.first_layer:    
            x2 = self.conv_mp_first(x2) #for 1st layer
        else:
            if self.in_planes != self.out_planes:   #for seg_x2, seg_x4
                x2 = self.gn_mp(x2)
                # x2 = self.relu(x2)
                x2 = self.hardswish(x2)
                x2 = self.conv_mp(x2)
        # print(x2.size())
        x2 = x2 + res

        x2 = self.gn_res1(x2)
        # x2 = self.relu(x2)
        x2 = self.hardswish(x2)
        x2 = self.conv_res1(x2)

        x1 = x1*(1 + torch.sigmoid(x2))

        return [x1, x2]


class NoBottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=(1, 1, 1), dilation=(1, 1, 1), downsample=None, fist_dilation=1,
                 multi_grid=1, weight_std=False):
        super(NoBottleneck, self).__init__()
        self.weight_std = weight_std
        self.hardswish = nn.Hardswish(inplace=True)
        self.gn1 = nn.GroupNorm(8, inplanes)
        self.conv1 = conv3x3x3(inplanes, planes, kernel_size=(3, 3, 3), stride=stride, padding=dilation * multi_grid,
                                 dilation=dilation * multi_grid, bias=False, weight_std=self.weight_std)

        self.gn2 = nn.GroupNorm(8, planes)
        self.conv2 = conv3x3x3(planes, planes, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=dilation * multi_grid,
                                 dilation=dilation * multi_grid, bias=False, weight_std=self.weight_std)

        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        skip = x

        seg = self.gn1(x)
        # seg = self.relu(seg)
        seg = self.hardswish(seg)
        seg = self.conv1(seg)

        seg = self.gn2(seg)
        # seg = self.relu(seg)
        seg = self.hardswish(seg)
        seg = self.conv2(seg)

        if self.downsample is not None:
            skip = self.downsample(x)

        seg = seg + skip
        return seg


class conresnet(nn.Module):
    def __init__(self, shape, block, layers, num_classes=3, weight_std=False, do_ds=True):
        self.shape = shape
        self.weight_std = weight_std
        super(conresnet, self).__init__()

        self.conv_4_32 = nn.Sequential(
            conv3x3x3(1, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), weight_std=self.weight_std))

        self.conv_32_64 = nn.Sequential(
            nn.GroupNorm(8, 32),
            nn.Hardswish(inplace=True),
            conv3x3x3(32, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), weight_std=self.weight_std))

        self.conv_64_128 = nn.Sequential(
            nn.GroupNorm(8, 64),
            nn.Hardswish(inplace=True),
            conv3x3x3(64, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), weight_std=self.weight_std))

        self.conv_128_256 = nn.Sequential(
            nn.GroupNorm(8, 128),
            nn.Hardswish(inplace=True),
            conv3x3x3(128, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2), weight_std=self.weight_std))

        self.layer0 = self._make_layer(block, 32, 32, layers[0], stride=(1, 1, 1))
        self.layer1 = self._make_layer(block, 64, 64, layers[1], stride=(1, 1, 1))
        self.layer2 = self._make_layer(block, 128, 128, layers[2], stride=(1, 1, 1))
        self.layer3 = self._make_layer(block, 256, 256, layers[3], stride=(1, 1, 1))
        self.layer4 = self._make_layer(block, 256, 256, layers[4], stride=(1, 1, 1), dilation=(2,2,2))

        self.fusionConv = nn.Sequential(
            nn.GroupNorm(8, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout3d(0.1),
            conv3x3x3(256, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), weight_std=self.weight_std)
        )

        self.seg_x4 = nn.Sequential(
            ConResAtt(128, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1), weight_std=self.weight_std, first_layer=True))
        self.seg_x2 = nn.Sequential(
            ConResAtt(64, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1), weight_std=self.weight_std))
        self.seg_x1 = nn.Sequential(
            ConResAtt(32, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1), weight_std=self.weight_std))

        self.seg_cls = nn.Sequential(
            nn.Conv3d(32, num_classes, kernel_size=1)
        )
        self.res_cls = nn.Sequential(
            nn.Conv3d(32, num_classes, kernel_size=1)
        )
        self.resx2_cls = nn.Sequential(
            nn.Conv3d(32, num_classes, kernel_size=1)
        )
        self.resx4_cls = nn.Sequential(
            nn.Conv3d(64, num_classes, kernel_size=1)
        )
        self.use_3DUXNet = True
        self.uxnet_3d = uxnet_conv(
            in_chans= 32,
            depths=[2, 2, 2],
            dims=[64, 128, 256],
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            out_indices=[0,1,2]
        )
        self.use_p2t3d = True
        self.p2t3d = PyramidPoolingTransformer3D(vol_size=(80,160,160))
        self.do_ds = do_ds
    def _make_layer(self, block, inplanes, outplanes, blocks, stride=(1, 1, 1), dilation=(1, 1, 1), multi_grid=1):
        downsample = None
        if stride[0] != 1 or stride[1] != 1 or stride[2] != 1 or inplanes != outplanes:
            downsample = nn.Sequential(
                nn.GroupNorm(8, inplanes),
                nn.Hardswish(inplace=True),
                conv3x3x3(inplanes, outplanes, kernel_size=(1, 1, 1), stride=stride, padding=(0, 0, 0),
                            weight_std=self.weight_std)
            )

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(inplanes, outplanes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid), weight_std=self.weight_std))
        for i in range(1, blocks):
            layers.append(
                block(inplanes, outplanes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid),
                      weight_std=self.weight_std))
        return nn.Sequential(*layers)


    def forward(self, x_list):
        x, x_res = x_list

        x = self.conv_4_32(x)   #[1,32,80,160,160]
        x = self.layer0(x)
        skip1 = x

        if self.use_3DUXNet:
            outsuxn = self.uxnet_3d(skip1)
            # print(outsuxn[0].size())   #torch.Size([1, 64, 40, 80, 80])
            # print(outsuxn[1].size())   #torch.Size([1, 128, 20, 40, 40])
            # print(outsuxn[2].size())   #torch.Size([1, 256, 10, 20, 20])
        if self.use_p2t3d:
            outp2d = self.p2t3d(x_list[0])
            # print(outp2d[0].size())
            # print(outp2d[1].size())
            # print(outp2d[2].size())

        if self.use_3DUXNet and self.use_p2t3d:
            x = outsuxn[0] + outp2d[0]
        else:
            x = self.conv_32_64(x)
        x = self.layer1(x)
        skip2 = x

        if self.use_3DUXNet and self.use_p2t3d:
            x = outsuxn[1] + outp2d[1]
        else:
            x = self.conv_64_128(x)
        x = self.layer2(x)
        skip3 = x

        if self.use_3DUXNet and self.use_p2t3d:
            x = outsuxn[2] + outp2d[2] 
        else:
            x = self.conv_128_256(x)
        del outsuxn, outp2d
        x = self.layer3(x)  #[1,256,10,20,20]

        x = self.layer4(x)

        x = self.fusionConv(x)  #[1,256,10,20,20]

        ## decoder  
        # 1.adjust size from conved x and original x_res(with more info)  
        # 2.skip connection for seg_x
        # 3.ConResAtt
        res_x4 = F.interpolate(x_res, size=(int(self.shape[0] / 4), int(self.shape[1] / 4), int(self.shape[2] / 4)), mode='trilinear', align_corners=True)
        seg_x4 = F.interpolate(x, size=(int(self.shape[0] / 4), int(self.shape[1] / 4), int(self.shape[2] / 4)), mode='trilinear', align_corners=True)
        seg_x4 = seg_x4 + skip3
        del x_res, skip3
        seg_x4, res_x4 = self.seg_x4([seg_x4, res_x4])

        res_x2 = F.interpolate(res_x4, size=(int(self.shape[0] / 2), int(self.shape[1] / 2), int(self.shape[2] / 2)), mode='trilinear', align_corners=True)
        seg_x2 = F.interpolate(seg_x4, size=(int(self.shape[0] / 2), int(self.shape[1] / 2), int(self.shape[2] / 2)), mode='trilinear', align_corners=True)
        seg_x2 = seg_x2 + skip2
        del seg_x4, skip2
        seg_x2, res_x2 = self.seg_x2([seg_x2, res_x2])

        res_x1 = F.interpolate(res_x2, size=(int(self.shape[0] / 1), int(self.shape[1] / 1), int(self.shape[2] / 1)), mode='trilinear', align_corners=True)
        seg_x1 = F.interpolate(seg_x2, size=(int(self.shape[0] / 1), int(self.shape[1] / 1), int(self.shape[2] / 1)), mode='trilinear', align_corners=True)
        seg_x1 = seg_x1 + skip1
        del seg_x2, skip1
        seg_x1, res_x1 = self.seg_x1([seg_x1, res_x1])

        seg = self.seg_cls(seg_x1)
        res = self.res_cls(res_x1)
        resx2 = self.resx2_cls(res_x2)
        resx4 = self.resx4_cls(res_x4)

        resx2 = F.interpolate(resx2, size=(int(self.shape[0] / 1), int(self.shape[1] / 1), int(self.shape[2] / 1)),
                      mode='trilinear', align_corners=True)
        resx4 = F.interpolate(resx4, size=(int(self.shape[0] / 1), int(self.shape[1] / 1), int(self.shape[2] / 1)),
                      mode='trilinear', align_corners=True)
        if self.do_ds:
            out = [seg, res, resx2, resx4]
        else:
            out = seg
        return out


def ConResNet(shape, num_classes=3, weight_std=True, do_ds=True):

    model = conresnet(shape, NoBottleneck, [1, 2, 2, 2, 2], num_classes, weight_std, do_ds)

    return model

if __name__ == '__main__':
    # 
    input_size = (80,160,160)
    num_classes = 8
    model = ConResNet(input_size, num_classes, weight_std=True)
    model.cuda()
    input = torch.randn(1,1, 80,160,160).cuda()
    # summary(model([input,input]))
    preds= model([input, input])
    print(preds[0].shape)
