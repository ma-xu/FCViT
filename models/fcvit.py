import os
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.layers.helpers import to_2tuple
from einops.layers.torch import Rearrange
from einops import rearrange

try:
    from mmseg.models.builder import BACKBONES as seg_BACKBONES
    from mmseg.utils import get_root_logger
    from mmcv.runner import _load_checkpoint
    has_mmseg = True
except ImportError:
    # print("If for semantic segmentation, please install mmsegmentation first")
    has_mmseg = False

try:
    from mmdet.models.builder import BACKBONES as det_BACKBONES
    from mmdet.utils import get_root_logger
    from mmcv.runner import _load_checkpoint
    has_mmdet = True
except ImportError:
    # print("If for detection, please install mmdetection first")
    has_mmdet = False

try:
    # install depthwise_conv2d_implicit_gemm for fast large kernel training/inference.
    # install see: https://github.com/MegEngine/cutlass/tree/master/examples/19_large_depthwise_conv2d_torch_extension
    from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
    print("Using DepthWiseConv2dImplicitGEMM for DW-Conv")
    class DWConv2D(DepthWiseConv2dImplicitGEMM):
        def __init__(self, in_channels, kernel_size, bias=True):
            super().__init__( in_channels, kernel_size, bias)

except:
    print("Using Pytorch  for DW-Conv")
    class DWConv2D(nn.Conv2d):
        def __init__(self, in_channels, kernel_size, bias=True):
            super().__init__(in_channels, in_channels, kernel_size, stride=1,
                             padding= kernel_size//2,  groups=in_channels, bias=bias)

params= {
    "global_context":{
        "weighted_gc": True,
        "gc_reduction": 8,
        "compete": False,
        "head": 8,
    },
    "spatial_mixer":{
        "use_globalcontext":True,
        "useSecondTokenMix": True,
        "mix_size_1": 5,
        "mix_size_2": 7,
        "fc_factor": 8,
        "fc_min_value": 16,
        "useSpatialAtt": True
    },
    "channel_mixer":{
        "useChannelAtt": True,
        "useDWconv":True,
        "DWconv_size":3
    },
    "spatial_att":{
        "kernel_size": 3,
        "dim_reduction":8
    },
    "channel_att":{
        "size_1": 3,
        "size_2": 5,
    }
}

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .95, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    's': _cfg(crop_pct=0.9),
    'm': _cfg(crop_pct=0.95),
}


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.BatchNorm2d(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm,nn.GroupNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class SpatialAtt(nn.Module):
    def __init__(self, dim, act_layer=nn.GELU, params = params):
        super().__init__()
        self.spatial_att = nn.Sequential(
            DWConv2D(dim, kernel_size=params["spatial_att"]["kernel_size"]),
            nn.Conv2d(dim, dim//params["spatial_att"]["dim_reduction"], 1),
            act_layer(),
            nn.Conv2d(dim//params["spatial_att"]["dim_reduction"], dim, 1),
            nn.Sigmoid()
        )

    def forward(self,x):
        return x * self.spatial_att(x)


class ChannelAtt(nn.Module):
    def __init__(self, act_layer=nn.GELU, params=params):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        size_1 =params["channel_att"]["size_1"]
        size_2 =params["channel_att"]["size_2"]
        self.channelConv1 = nn.Conv1d(1, 1, size_1, padding=size_1//2)
        self.channelConv2 = nn.Conv1d(1, 1, kernel_size=size_2, padding=size_2//2)
        self.act = act_layer()

    def forward(self, x):
        res = x.clone()
        x = self.avg_pool(self.act(x))
        x = self.channelConv1(x.squeeze(-1).transpose(-1, -2))
        x = self.act(x)
        x = self.channelConv2(x)
        x = x.transpose(-1, -2).unsqueeze(-1)

        return res + x


class GlobalContext(nn.Module):
    def __init__(self, dim, act_layer=nn.GELU, params=params):
        super().__init__()
        # bottleneck information
        # "Compete to compute." NeurIPS 2013
        self.compete = params["global_context"]["compete"]
        if self.compete:
            self.fc1 = nn.Linear(dim, 2*dim//params["global_context"]["gc_reduction"])
            self.fc2 = nn.Linear(dim//params["global_context"]["gc_reduction"], dim)
        else:
            self.fc = nn.Sequential(
                nn.Linear(dim, dim//params["global_context"]["gc_reduction"]),
                act_layer(),
                nn.Linear(dim//params["global_context"]["gc_reduction"], dim)
            )
        self.weight_gc = params["global_context"]["weighted_gc"]
        if self.weight_gc:
            self.head = params["global_context"]["head"]
            self.scale = (dim//self.head) ** -0.5
            self.rescale_weight = nn.Parameter(torch.ones(self.head))
            self.rescale_bias = nn.Parameter(torch.zeros(self.head))
            self.epsilon = 1e-5

    def _get_gc(self, gap): # gap [b,c]
        if self.compete:
            b,c = gap.size()
            gc = self.fc1(gap).reshape([b,2,-1])
            gc, _ = gc.max(dim=1)
            gc = self.fc2(gc)
            return gc
        else:
            return self.fc(gap)


    def forward(self,x):
        if self.weight_gc:
            b,c,w,h = x.size()
            x = rearrange(x,"b c x y -> b c (x y)")
            gap = x.mean(dim=-1, keepdim=True)
            q, g = map(lambda t: rearrange(t, 'b (h d) n -> b h d n', h = self.head), [x,gap])  #[b,head, hdim, n]
            sim = einsum('bhdi,bhjd->bhij', q, g.transpose(-1, -2)).squeeze(dim=-1) * self.scale  #[b,head, w*h]
            std, mean = torch.std_mean(sim, dim=[1,2], keepdim=True)
            sim = (sim-mean)/(std+self.epsilon)
            sim = sim * self.rescale_weight.unsqueeze(dim=0).unsqueeze(dim=-1) + self.rescale_bias.unsqueeze(dim=0).unsqueeze(dim=-1)
            sim = sim.reshape(b,self.head,1, w, h) # [b, head, 1, w, h]
            gc = self._get_gc(gap.squeeze(dim=-1)).reshape(b,self.head,-1).unsqueeze(dim=-1).unsqueeze(dim=-1)  # [b, head, hdim, 1, 1]
            gc = rearrange(sim*gc, "b h d x y -> b (h d) x y")  # [b, head, hdim, w, h] - > [b,c,w,h]
        else:
            gc = self._get_gc(x.mean(dim=-1).mean(dim=-1)).unsqueeze(dim=-1).unsqueeze(dim=-1)
        return gc  # [b,c,w,h] for weighted or [b,c,1,1]


class TokenMixer(nn.Module):
    def __init__(self, dim, act_layer=nn.GELU, params=params):
        super().__init__()
        self.act = act_layer()
        self.useSpatialAtt = params["spatial_mixer"]["useSpatialAtt"]
        if params["spatial_mixer"]["use_globalcontext"]:
            self.gc1 = GlobalContext(dim, act_layer=act_layer, params=params)
        # self.dw1 = nn.Conv2d(dim, dim, kernel_size=params["spatial_mixer"]["mix_size_1"],
        #                      padding=params["spatial_mixer"]["mix_size_1"]//2, stride=1, groups=dim)
        self.dw1 = DWConv2D(dim, params["spatial_mixer"]["mix_size_1"])
        if params["spatial_mixer"]["fc_factor"]>1:
            self.fc1 = nn.Sequential(
                nn.Conv2d(dim, max(dim//params["spatial_mixer"]["fc_factor"], params["spatial_mixer"]["fc_min_value"]), 1),
                self.act,
                nn.Conv2d(max(dim//params["spatial_mixer"]["fc_factor"], params["spatial_mixer"]["fc_min_value"]), dim, 1)
            )
        else:
            self.fc1 = nn.Conv2d(dim,dim,1)

        if params["spatial_mixer"]["useSecondTokenMix"]:
            if params["spatial_mixer"]["use_globalcontext"]:
                self.gc2 = GlobalContext(dim, act_layer=act_layer, params=params)
            self.dw2 = DWConv2D(dim, params["spatial_mixer"]["mix_size_2"])
            if params["spatial_mixer"]["fc_factor"]>1:
                self.fc2 = nn.Sequential(
                    nn.Conv2d(dim, max(dim//params["spatial_mixer"]["fc_factor"], params["spatial_mixer"]["fc_min_value"]), 1),
                    self.act,
                    nn.Conv2d(max(dim//params["spatial_mixer"]["fc_factor"], params["spatial_mixer"]["fc_min_value"]), dim, 1)
                )
            else:
                self.fc2 = nn.Conv2d(dim,dim,1)

        if params["spatial_mixer"]["useSpatialAtt"]:
            self.spatial_att = SpatialAtt(dim=dim, act_layer=act_layer, params=params)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm,nn.GroupNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        if hasattr(self,"gc1"):
            gc1 = self.gc1(x)
            x = x + gc1
        x = self.act(self.fc1(self.dw1(x)))

        if hasattr(self, "fc2"):
            if hasattr(self, "gc2"):
                gc2 = self.gc2(x)
                x = x + gc2
            x = self.act(self.fc2(self.dw2(x)))
        if self.useSpatialAtt:
            x = self.spatial_att(x)
        return x


class ChannelMixer(nn.Module):
    def __init__(self, dim, hidden_dim=None, act_layer=nn.GELU, drop=0., params=params):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.useChannelAtt = params["channel_mixer"]["useChannelAtt"]
        self.act = act_layer()
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1)
        if params["channel_mixer"]["useDWconv"]:
            ks=params["channel_mixer"]["DWconv_size"]
            self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, ks, padding=ks//2, groups=hidden_dim)

        self.fc2 =nn.Conv2d(hidden_dim, dim, 1)
        self.drop = nn.Dropout(drop)
        if self.useChannelAtt:
            self.channel_att = ChannelAtt(act_layer=act_layer, params=params)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm,nn.GroupNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        if hasattr(self, "dwconv"):
            x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        if self.useChannelAtt:
            x = self.channel_att(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, dim,  mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-4,
                 params = params):

        super().__init__()
        self.norm1 = norm_layer(dim)
        self.token_mixer = TokenMixer(dim=dim, act_layer=act_layer, params=params)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.channel_mixer = ChannelMixer(dim=dim, hidden_dim=mlp_hidden_dim, act_layer=act_layer, drop=drop, params=params)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.channel_mixer(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.channel_mixer(self.norm2(x)))
        return x


def basic_blocks(dim, index, layers,
                 mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop_rate=.0, drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 params = params):
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * ( block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(BasicBlock(
            dim, mlp_ratio=mlp_ratio,
            act_layer=act_layer, norm_layer=norm_layer,
            drop=drop_rate, drop_path=block_dpr,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            params=params
            )
        )
    blocks = nn.Sequential(*blocks)

    return blocks


class BaseFormer(nn.Module):
    def __init__(self, layers, embed_dims=None,
                 mlp_ratios=None, downsamples=None,
                 norm_layer=GroupNorm, act_layer=nn.GELU,
                 num_classes=1000,
                 drop_rate=0., drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 fork_feat=False,
                 init_cfg=None,
                 pretrained=None,
                 params = params,
                 **kwargs):

        super().__init__()

        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat

        self.patch_embed = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=3, embed_dim=embed_dims[0])

        # set the main block in network
        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], i, layers,
                                 mlp_ratio=mlp_ratios[i],
                                 act_layer=act_layer, norm_layer=norm_layer,
                                 drop_rate=drop_rate,
                                 drop_path_rate=drop_path_rate,
                                 use_layer_scale=use_layer_scale,
                                 layer_scale_init_value=layer_scale_init_value,
                                 params=params)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i+1]:
                # downsampling between two stages
                network.append(
                    OverlapPatchEmbed(patch_size= 3, stride= 2, in_chans=embed_dims[i], embed_dim=embed_dims[i+1])
                )

        self.network = nn.ModuleList(network)

        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    # TODO: more elegant way
                    """For RetinaNet, `start_level=1`. The first norm layer will not used.
                    cmd: `FORK_LAST3=1 python -m torch.distributed.launch ...`
                    """
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            # Classifier head
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Linear(
                embed_dims[-1], num_classes) if num_classes > 0 \
                else nn.Identity()

        self.apply(self.cls_init_weights)

        self.init_cfg = copy.deepcopy(init_cfg)
        # load pre-trained model
        if self.fork_feat and (
                self.init_cfg is not None or pretrained is not None):
            self.init_weights()

    # init for classification
    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # init for mmdetection or mmsegmentation by loading
    # imagenet pre-trained weights
    def init_weights(self, pretrained=None):
        logger = get_root_logger()
        if self.init_cfg is None and pretrained is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            pass
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg['checkpoint']
            elif pretrained is not None:
                ckpt_path = pretrained

            ckpt = _load_checkpoint(
                ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = _state_dict
            missing_keys, unexpected_keys = \
                self.load_state_dict(state_dict, False)

            # show for debug
            print('missing_keys: ', missing_keys)
            print('unexpected_keys: ', unexpected_keys)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        if self.fork_feat:
            # output the features of four stages for dense prediction
            return outs
        # output only the features of last layer for image classification
        return x

    def forward(self, x):
        # input embedding
        x = self.forward_embeddings(x)
        # through backbone
        x = self.forward_tokens(x)
        if self.fork_feat:
            # otuput features of four stages for dense prediction
            return x
        x = self.norm(x)
        cls_out = self.head(x.mean([-2, -1]))
        # for image classification
        return cls_out


@register_model
def fcvit_tiny(pretrained=False, **kwargs):

    fcvt_params = params.copy()

    fcvt_params["spatial_mixer"]["useSecondTokenMix"] = True
    fcvt_params["spatial_mixer"]["use_globalcontext"]=True
    fcvt_params["spatial_mixer"]["mix_size_1"] = 11
    fcvt_params["spatial_mixer"]["mix_size_2"]=11

    fcvt_params["global_context"]["weighted_gc"] = True
    fcvt_params["global_context"]["head"] = 8
    fcvt_params["global_context"]["compete"] = True

    fcvt_params["channel_mixer"]["useDWconv"] = True

    fcvt_params["spatial_mixer"]["useSpatialAtt"] = False
    fcvt_params["channel_mixer"]["useChannelAtt"] = False

    layers = [3, 3, 5, 2]
    embed_dims = [32, 64, 160, 320]
    mlp_ratios = [8, 8, 4, 4]
    downsamples = [True, True, True, True]

    model = BaseFormer(
        layers, embed_dims=embed_dims,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        params = fcvt_params,
        **kwargs)
    model.default_cfg = default_cfgs['s']
    return model


@register_model
def fcvit_b12(pretrained=False, **kwargs):
    fcvt_params = params.copy()

    fcvt_params["spatial_mixer"]["useSecondTokenMix"] = True
    fcvt_params["spatial_mixer"]["use_globalcontext"]=True
    fcvt_params["spatial_mixer"]["mix_size_1"] = 11
    fcvt_params["spatial_mixer"]["mix_size_2"]=11

    fcvt_params["global_context"]["weighted_gc"] = True
    fcvt_params["global_context"]["head"] = 8
    fcvt_params["global_context"]["compete"] = True

    fcvt_params["channel_mixer"]["useDWconv"] = True

    fcvt_params["spatial_mixer"]["useSpatialAtt"] = False
    fcvt_params["channel_mixer"]["useChannelAtt"] = False

    layers = [2, 2, 6, 2]
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [8, 8, 4, 4]
    downsamples = [True, True, True, True]

    model = BaseFormer(
        layers, embed_dims=embed_dims,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        params = fcvt_params,
        **kwargs)
    model.default_cfg = default_cfgs['s']
    return model


@register_model
def fcvit_b24(pretrained=False, **kwargs):
    fcvt_params = params.copy()

    fcvt_params["spatial_mixer"]["useSecondTokenMix"] = True
    fcvt_params["spatial_mixer"]["use_globalcontext"]=True
    fcvt_params["spatial_mixer"]["mix_size_1"] = 11
    fcvt_params["spatial_mixer"]["mix_size_2"]=11

    fcvt_params["global_context"]["weighted_gc"] = True
    fcvt_params["global_context"]["head"] = 8
    fcvt_params["global_context"]["compete"] = True

    fcvt_params["channel_mixer"]["useDWconv"] = True

    fcvt_params["spatial_mixer"]["useSpatialAtt"] = False
    fcvt_params["channel_mixer"]["useChannelAtt"] = False

    layers = [4, 4, 12, 4]
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [8, 8, 4, 4]
    downsamples = [True, True, True, True]

    model = BaseFormer(
        layers, embed_dims=embed_dims,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        params = fcvt_params,
        **kwargs)
    model.default_cfg = default_cfgs['m']
    return model


@register_model
def fcvit_b48(pretrained=False, **kwargs):
    fcvt_params = params.copy()

    fcvt_params["spatial_mixer"]["useSecondTokenMix"] = True
    fcvt_params["spatial_mixer"]["use_globalcontext"]=True
    fcvt_params["spatial_mixer"]["mix_size_1"] = 11
    fcvt_params["spatial_mixer"]["mix_size_2"]=11

    fcvt_params["global_context"]["weighted_gc"] = True
    fcvt_params["global_context"]["head"] = 8
    fcvt_params["global_context"]["compete"] = True

    fcvt_params["channel_mixer"]["useDWconv"] = True

    fcvt_params["spatial_mixer"]["useSpatialAtt"] = False
    fcvt_params["channel_mixer"]["useChannelAtt"] = False

    layers = [8, 8, 24, 8]
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [8, 8, 4, 4]
    downsamples = [True, True, True, True]

    model = BaseFormer(
        layers, embed_dims=embed_dims,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        params = fcvt_params,
        **kwargs)
    model.default_cfg = default_cfgs['m']
    return model


if has_mmdet & has_mmseg:
    @det_BACKBONES.register_module()
    @seg_BACKBONES.register_module()
    class fcvit_b24_feat(BaseFormer):
        def __init__(self, **kwargs):
                fcvt_params = params.copy()
                fcvt_params["spatial_mixer"]["useSecondTokenMix"] = True
                fcvt_params["spatial_mixer"]["use_globalcontext"]=True
                fcvt_params["spatial_mixer"]["mix_size_1"] = 11
                fcvt_params["spatial_mixer"]["mix_size_2"]=11
                fcvt_params["global_context"]["weighted_gc"] = True
                fcvt_params["global_context"]["head"] = 8
                fcvt_params["global_context"]["compete"] = True
                fcvt_params["channel_mixer"]["useDWconv"] = True
                fcvt_params["spatial_mixer"]["useSpatialAtt"] = False
                fcvt_params["channel_mixer"]["useChannelAtt"] = False

                layers = [4, 4, 12, 4]
                embed_dims = [64, 128, 320, 512]
                mlp_ratios = [8, 8, 4, 4]
                downsamples = [True, True, True, True]
                super().__init__(
                    layers, embed_dims=embed_dims,
                    mlp_ratios=mlp_ratios, downsamples=downsamples,
                    params = fcvt_params,
                    layer_scale_init_value=1e-6,
                    fork_feat=True,
                    **kwargs)


    @det_BACKBONES.register_module()
    @seg_BACKBONES.register_module()
    class fcvit_b12_feat(BaseFormer):
        def __init__(self, **kwargs):
                fcvt_params = params.copy()
                fcvt_params["spatial_mixer"]["useSecondTokenMix"] = True
                fcvt_params["spatial_mixer"]["use_globalcontext"]=True
                fcvt_params["spatial_mixer"]["mix_size_1"] = 11
                fcvt_params["spatial_mixer"]["mix_size_2"]=11
                fcvt_params["global_context"]["weighted_gc"] = True
                fcvt_params["global_context"]["head"] = 8
                fcvt_params["global_context"]["compete"] = True
                fcvt_params["channel_mixer"]["useDWconv"] = True
                fcvt_params["spatial_mixer"]["useSpatialAtt"] = False
                fcvt_params["channel_mixer"]["useChannelAtt"] = False

                layers = [2, 2, 6, 2]
                embed_dims = [64, 128, 320, 512]
                mlp_ratios = [8, 8, 4, 4]
                downsamples = [True, True, True, True]
                super().__init__(
                    layers, embed_dims=embed_dims,
                    mlp_ratios=mlp_ratios, downsamples=downsamples,
                    params = fcvt_params,
                    layer_scale_init_value=1e-6,
                    fork_feat=True,
                    **kwargs)

if __name__ == '__main__':
    input = torch.rand(2, 3, 224, 224)
    model = fcvit_tiny()
    out = model(input)
    print(model)
    print(out.shape)

