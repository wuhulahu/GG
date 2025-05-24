# Copyright IBM All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


"""
Modifed from Timm. https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.hub
from functools import partial


from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg, Mlp
from itertools import combinations
from random import shuffle, choice
from collections import OrderedDict
from torch.jit import Final
from timm.layers import  Mlp, DropPath, trunc_normal_, lecun_normal_, resample_patch_embed, \
    resample_abs_pos_embed, RmsNorm, PatchDropout, use_fused_attn, SwiGLUPacked
from timm.models.vision_transformer import LayerScale

_model_urls = {
    'crossvit_15_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_15_224.pth',
    'crossvit_15_dagger_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_15_dagger_224.pth',
    'crossvit_15_dagger_384': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_15_dagger_384.pth',
    'crossvit_18_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_18_224.pth',
    'crossvit_18_dagger_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_18_dagger_224.pth',
    'crossvit_18_dagger_384': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_18_dagger_384.pth',
    'crossvit_9_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_9_224.pth',
    'crossvit_9_dagger_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_9_dagger_224.pth',
    'crossvit_base_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_base_224.pth',
    'crossvit_small_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_small_224.pth',
    'crossvit_tiny_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_tiny_224.pth',
}



class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, multi_conv=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        if multi_conv:
            if patch_size[0] == 12:
                self.proj = nn.Sequential(
                    nn.Conv2d(in_chans, embed_dim // 4, kernel_size=7, stride=4, padding=3),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=3, padding=0),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=1, padding=1),
                )
            elif patch_size[0] == 16:
                self.proj = nn.Sequential(
                    nn.Conv2d(in_chans, embed_dim // 4, kernel_size=7, stride=4, padding=3),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
                )
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    # def softmax_with_mask(self, attn, policy, eps=1e-6):
    #
    #     # 将输入的policy转化为对应的mask，保证mask中值为0的token不参与attention的计算
    #     # B, N, _ = policy.size()
    #     B, H, N, N = attn.size()
    #     attn_mask = policy.reshape(B, 1, 1, N)  # * policy.reshape(B, 1, N, 1)
    #     eye = torch.eye(N, dtype=attn_mask.dtype, device=attn_mask.device).view(1, 1, N, N)
    #     attn_policy = attn_mask + (1.0 - attn_mask) * eye
    #     max_att = torch.max(attn, dim=-1, keepdim=True)[0]
    #     attn = attn - max_att
    #     # attn = attn.exp_() * attn_policy
    #     # return attn / attn.sum(dim=-1, keepdim=True)
    #
    #     # for stable training
    #     attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
    #     attn = (attn + eps / N) / (attn.sum(dim=-1, keepdim=True) + eps)
    #     return attn.type_as(max_att)

    def softmax_with_mask(self, attn, mask):
        # todo 这里的policy就是mask，乘以-10000加上去就好了可能。

        # 将输入的policy转化为对应的mask，保证mask中值为0的token不参与attention的计算
        # B, N, _ = policy.size()
        B, H, N, N = attn.size()
        expanded_mask = mask.reshape(B, 1, 1, N)  # * policy.reshape(B, 1, N, 1)
        expanded_mask = expanded_mask.expand_as(attn)

        # 在需要进行 mask 的位置设置一个很大的负数，使得在 softmax 时趋近于零
        attn = attn.masked_fill(expanded_mask == 0, float('-inf'))
        softmax_output = torch.nn.functional.softmax(attn, dim=-1)
        return softmax_output

    def forward(self, x, mask):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # 原来的 为了避免fused_attn
        # if self.fused_attn:
        #     x = F.scaled_dot_product_attention(
        #         q, k, v,
        #         dropout_p=self.attn_drop.p,
        #     )
        # else:
        #     q = q * self.scale
        #     attn = q @ k.transpose(-2, -1)
        #     attn = attn.softmax(dim=-1)
        #     attn = self.attn_drop(attn)
        #     x = attn @ v

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = self.softmax_with_mask(attn, mask)
        attn = self.attn_drop(attn)
        x = attn @ v


        # extended_s2_mask = s2_mask.unsqueeze(1).unsqueeze(2)
        # extended_s2_mask = extended_s2_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        # extended_s2_mask = (1.0 - extended_s2_mask) * -10000.0  # s2 mask


        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x_):
        x = x_[0]
        mask = x_[1]
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return [x, mask]


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    # def softmax_with_mask(self, attn, mask, eps=1e-6):
    #     # 将输入的policy转化为对应的mask，保证mask中值为0的token不参与attention的计算
    #     # B, N, _ = policy.size()
    #     B, H, _, N = attn.size()
    #     attn_mask = mask.reshape(B, 1, 1, N)  # * policy.reshape(B, 1, N, 1)
    #     # eye = torch.eye(N, dtype=attn_mask.dtype, device=attn_mask.device).view(1, 1, N, N)
    #     attn_policy = attn_mask + (1.0 - attn_mask) * attn_mask
    #     max_att = torch.max(attn, dim=-1, keepdim=True)[0]
    #     attn = attn - max_att
    #     # attn = attn.exp_() * attn_policy
    #     # return attn / attn.sum(dim=-1, keepdim=True)
    #
    #     # for stable training
    #     attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
    #     attn = (attn + eps / N) / (attn.sum(dim=-1, keepdim=True) + eps)
    #     return attn.type_as(max_att)

    # def softmax_with_mask(self, attn, mask, eps=1e-6):
    #
    #     B, H, _, N = attn.size()
    #     attn_mask = mask.reshape(B, 1, 1, N)  # * policy.reshape(B, 1, N, 1)
    #     # eye = torch.eye(N, dtype=attn_mask.dtype, device=attn_mask.device).view(1, 1, N, N)
    #     attn_policy = attn_mask + (1.0 - attn_mask) * attn_mask
    #     max_att = torch.max(attn, dim=-1, keepdim=True)[0]
    #     attn = attn - max_att
    #     # attn = attn.exp_() * attn_policy
    #     # return attn / attn.sum(dim=-1, keepdim=True)
    #
    #     # for stable training
    #     attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
    #     attn = (attn + eps / N) / (attn.sum(dim=-1, keepdim=True) + eps)
    #     return attn.type_as(max_att)

    def softmax_with_mask(self, attn, mask):
        # todo 这里的policy就是mask，乘以-10000加上去就好了可能。

        # 将输入的policy转化为对应的mask，保证mask中值为0的token不参与attention的计算
        # B, N, _ = policy.size()
        B, H, N, N = attn.size()
        expanded_mask = mask.reshape(B, 1, 1, N)  # * policy.reshape(B, 1, N, 1)
        expanded_mask = expanded_mask.expand_as(attn)

        # 在需要进行 mask 的位置设置一个很大的负数，使得在 softmax 时趋近于零
        attn = attn.masked_fill(expanded_mask == 0, float('-inf'))
        softmax_output = torch.nn.functional.softmax(attn, dim=-1)
        return softmax_output

    def forward(self, x, mask):
        B, N, C = x.shape
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        # attn = attn.softmax(dim=-1)
        attn = self.softmax_with_mask(attn, mask)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)   # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x_):
        x = x_[0]
        mask = x_[1]
        x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x), mask))
        if self.has_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return [x, mask]


class MultiScaleBlock(nn.Module):

    def __init__(self, dim, patches, depth, num_heads, mlp_ratio, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        num_branches = len(dim)
        self.scale = qk_scale or 1000 ** -0.5
        self.num_branches = num_branches
        # different branch could have different embedding size, the first one is the base
        self.blocks = nn.ModuleList()
        for d in range(num_branches):
            tmp = []
            for i in range(depth[d]):
                tmp.append(
                    Block(dim=dim[d], num_heads=num_heads[d], mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias,
                          proj_drop=drop, attn_drop=attn_drop, drop_path=drop_path[i], norm_layer=norm_layer))
            if len(tmp) != 0:
                self.blocks.append(nn.Sequential(*tmp))

        if len(self.blocks) == 0:
            self.blocks = None

        self.projs = nn.ModuleList()
        for d in range(num_branches):
            if dim[d] == dim[(d+1) % num_branches] and False:
                tmp = [nn.Identity()]
            else:
                tmp = [norm_layer(dim[d]), act_layer(), nn.Linear(dim[d], dim[(d+1) % num_branches])]
            self.projs.append(nn.Sequential(*tmp))

        self.fusion = nn.ModuleList()
        for d in range(num_branches):
            d_ = (d+1) % num_branches
            nh = num_heads[d_]
            if depth[-1] == 0:  # backward capability:
                self.fusion.append(CrossAttentionBlock(dim=dim[d_], num_heads=nh, mlp_ratio=mlp_ratio[d],
                                                       qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                       drop=drop, attn_drop=attn_drop, drop_path=drop_path[-1],
                                                       norm_layer=norm_layer,
                                                       has_mlp=False))
            else:
                tmp = []
                for _ in range(depth[-1]):
                    tmp.append(CrossAttentionBlock(dim=dim[d_], num_heads=nh, mlp_ratio=mlp_ratio[d],
                                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                   drop=drop, attn_drop=attn_drop, drop_path=drop_path[-1],
                                                   norm_layer=norm_layer,
                                                   has_mlp=False))
                self.fusion.append(nn.Sequential(*tmp))

        self.revert_projs = nn.ModuleList()
        for d in range(num_branches):
            if dim[(d+1) % num_branches] == dim[d] and False:
                tmp = [nn.Identity()]
            else:
                tmp = [norm_layer(dim[(d+1) % num_branches]), act_layer(), nn.Linear(dim[(d+1) % num_branches], dim[d])]
            self.revert_projs.append(nn.Sequential(*tmp))

    def forward(self, x, mask):
        # outs_b = [block(x_, policy) for x_, block in zip(x, self.blocks)]
        outs_b = []
        for x_, block, mask_ in zip(x, self.blocks, mask):
            t = block([x_, mask_])  # 输入block之前，加上mask
            t = t[0]    # 去掉mask
            outs_b.append(t)
        # 在这里只有cls_token，投影至另一个通道的维度。
        proj_cls_token = [proj(x[:, 0:1]) for x, proj in zip(outs_b, self.projs)]
        # cross attention
        outs = []
        for i in range(self.num_branches):
            tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
            tmp = [tmp, mask[(i + 1) % self.num_branches]]  # 加上mask
            tmp = self.fusion[i](tmp)
            tmp = tmp[0]    # 去掉mask
            # 将cls_token投影回去
            reverted_proj_cls_token = self.revert_projs[i](tmp[:, 0:1, ...])

            tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
            outs.append(tmp)
        # todo 在这里删除与cls的token
        return outs


def _compute_num_patches(img_size, patches):
    return [i // p * i // p for i, p in zip(img_size,patches)]


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=(224, 224), patch_size=(8, 16), in_chans=3, num_classes=1000, embed_dim=(192, 384),
                 depth=([1, 3, 1], [1, 3, 1], [1, 3, 1]),
                 num_heads=(6, 12), mlp_ratio=(2., 2., 4.), qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., code_length=16, synchronized=False, weight_remain=0.75,soft_per_example=True, layer_cut=-1, sigma=3, limit=0.5, hybrid_backbone=None, norm_layer=nn.LayerNorm, multi_conv=False, **kwargs):
        super().__init__()

        self.layer_cut = layer_cut
        self.synchronized = synchronized    # 控制两个通道的mask比例是否相似
        self.weight_remain = weight_remain  # 保留的比例
        self.soft_per_example = soft_per_example
        self.sigma = sigma
        self.limit = limit
        self.num_classes = num_classes
        if not isinstance(img_size, list):
            img_size = to_2tuple(img_size)
        self.img_size = img_size

        num_patches = _compute_num_patches(img_size, patch_size)

        # 多尺度通道的个数
        self.num_branches = len(patch_size)

        # patch_emb模型和pos_emb方式
        self.patch_embed = nn.ModuleList()
        if hybrid_backbone is None:
            self.pos_embed = nn.ParameterList([nn.Parameter(torch.zeros(1, 1 + num_patches[i], embed_dim[i])) for i in range(self.num_branches)])
            for im_s, p, d in zip(img_size, patch_size, embed_dim):
                self.patch_embed.append(PatchEmbed(img_size=im_s, patch_size=p, in_chans=in_chans, embed_dim=d, multi_conv=multi_conv))
        else:
            self.pos_embed = nn.ParameterList()
            from .t2t import T2T, get_sinusoid_encoding
            tokens_type = 'transformer' if hybrid_backbone == 't2t' else 'performer'
            for idx, (im_s, p, d) in enumerate(zip(img_size, patch_size, embed_dim)):
                self.patch_embed.append(T2T(im_s, tokens_type=tokens_type, patch_size=p, embed_dim=d))
                self.pos_embed.append(nn.Parameter(data=get_sinusoid_encoding(n_position=1 + num_patches[idx], d_hid=embed_dim[idx]), requires_grad=False))

            del self.pos_embed
            self.pos_embed = nn.ParameterList([nn.Parameter(torch.zeros(1, 1 + num_patches[i], embed_dim[i])) for i in range(self.num_branches)])

        # 根据不同的尺度来初始化cls_token
        self.cls_token = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, embed_dim[i])) for i in range(self.num_branches)])
        self.pos_drop = nn.Dropout(p=drop_rate)

        total_depth = sum([sum(x[-2:]) for x in depth])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]  # stochastic depth decay rule
        dpr_ptr = 0

        # 模型的主干部分
        self.blocks = nn.ModuleList()
        for idx, block_cfg in enumerate(depth):
            curr_depth = max(block_cfg[:-1]) + block_cfg[-1]
            dpr_ = dpr[dpr_ptr:dpr_ptr + curr_depth]
            blk = MultiScaleBlock(embed_dim, num_patches, block_cfg, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr_,
                                  norm_layer=norm_layer)
            dpr_ptr += curr_depth
            self.blocks.append(blk)

        self.norm = nn.ModuleList([norm_layer(embed_dim[i]) for i in range(self.num_branches)])
        self.head = nn.ModuleList([nn.Linear(embed_dim[i], num_classes) if num_classes > 0 else nn.Identity() for i in range(self.num_branches)])

        self.hash_layer = nn.Sequential(
            nn.Linear(num_classes, code_length, bias=True),
            nn.Tanh()
        )

        self.DistanceEstimator = nn.Sequential(
            nn.Linear(code_length*2, 2, bias=True),
            nn.Softmax(dim=-1)
        )

        for i in range(self.num_branches):
            if self.pos_embed[i].requires_grad:
                trunc_normal_(self.pos_embed[i], std=.02)
            trunc_normal_(self.cls_token[i], std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        out = {'cls_token'}
        if self.pos_embed[0].requires_grad:
            out.add('pos_embed')
        return out

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def num_sample(self, sigma=50, limit=180, n=100):
        # draw an integer from gaussian within +/- limit
        for d in range(n):
            x = torch.randn((1)) * sigma
            if abs(x) <= limit:
                return int(abs(x))
        return int(0)

    def generate_row_column(self, bs, num_remain_tokens, soft_per_example, num_samples=None):
        if soft_per_example:
            row = []
            column = []
            label_adjust = []
            for i in range(bs):
                if num_samples == None:
                    # sigma是 剩余的75%的图片token数量 /3，limit是 75%的图片token数量 的一半
                    this_num_samples = self.num_sample(sigma=num_remain_tokens / self.sigma, limit=num_remain_tokens * self.limit)
                else:
                    this_num_samples = num_samples[i]
                # 对剩下的，截取了75%的idx_range进行采样，这里采出来的是对idx_range的索引，所以最大为num_remain_tokens，
                # 即idx_range的shape[-1]
                sampled_indices_idx_range = torch.randint(low=0, high=num_remain_tokens, size=(this_num_samples,))

                # 生成row和colum，进行采样, 修改tensor元素时， a[row, column]是可以的，a[idx]不行，这里的idx是指索引矩阵
                original_tensor = torch.tensor(i)  # 用来生成row
                this_row = original_tensor.repeat_interleave(this_num_samples)
                this_column = sampled_indices_idx_range.view(-1)
                this_label_adjust = 1 - (this_num_samples / num_remain_tokens)

                row.append(this_row)
                column.append(this_column)
                label_adjust.append(this_label_adjust)

            row = torch.hstack(row)
            column = torch.hstack(column)
        else:
            if num_samples == None:
                num_samples = self.num_sample(sigma=num_remain_tokens / self.sigma, limit=num_remain_tokens * self.limit)

            # 对剩下的，截取了75%的idx_range进行采样，这里采出来的是对idx_range的索引，所以最大为num_remain_tokens，
            # 即idx_range的shape[-1]
            sampled_indices_idx_range = torch.randint(low=0, high=num_remain_tokens, size=(bs, num_samples))

            # 生成row和colum，进行采样, 修改tensor元素时， a[row, column]是可以的，a[idx]不行，这里的idx是指索引矩阵
            original_tensor = torch.arange(0, bs)  # 用来生成row
            row = original_tensor.repeat_interleave(num_samples)
            column = sampled_indices_idx_range.view(-1)
            # 标签soft权重
            label_adjust = 1 - (num_samples / num_remain_tokens)
            label_adjust = [label_adjust]

        return row, column, label_adjust

    def sample_mask_token_debug(self, idx_range, row=None, column=None, sampe_ratio=None):
        num_remain_tokens = idx_range.shape[-1]
        bs = idx_range.shape[0]
        if row == None or column == None:
            row, column, label_adjust = self.generate_row_column(bs, num_remain_tokens, soft_per_example=self.soft_per_example)

        # 取出要mask的原始token的idx，注意要+1，因为mask矩阵包含了额外的一个cls
        sampled_indices = idx_range[row, column] + 1

        if sampe_ratio != None:
            label_adjust = sampe_ratio

        return row, sampled_indices, label_adjust

    def sample_mask_token(self, idx_range):
        # flag whole_soft
        num_remain_tokens = idx_range.shape[-1]
        bs = idx_range.shape[0]
        # sigma是 剩余的75%的图片token数量 /3，limit是 75%的图片token数量 的一半
        num_samples = self.num_sample(sigma=num_remain_tokens / self.sigma, limit=num_remain_tokens * self.limit)

        # 对剩下的，截取了75%的idx_range进行采样，这里采出来的是对idx_range的索引，所以最大为num_remain_tokens，
        # 即idx_range的shape[-1]
        sampled_indices_idx_range = torch.randint(low=0, high=num_remain_tokens, size=(bs, num_samples))

        # 生成row和colum，进行采样, 修改tensor元素时， a[row, column]是可以的，a[idx]不行，这里的idx是指索引矩阵
        original_tensor = torch.arange(0, bs)  # 用来生成row
        row = original_tensor.repeat_interleave(num_samples)
        column = sampled_indices_idx_range.view(-1)

        # 取出要mask的原始token的idx，注意要+1，因为mask矩阵包含了额外的一个cls
        sampled_indices = idx_range[row, column] + 1

        # 标签soft权重
        label_adjust = 1 - (num_samples / num_remain_tokens)

        return row, sampled_indices, [label_adjust]

    def sample_mask_token_(self, idx_range, row=None, column=None):
        # sample_soft
        # todo 在这里可以传入row和column
        num_remain_tokens = idx_range.shape[-1]
        bs = idx_range.shape[0]

        # 取出要mask的原始token的idx，注意要+1，因为mask矩阵包含了额外的一个cls
        if row == None or column == None:
            row, column, label_adjust = self.generate_row_column(bs, num_remain_tokens)
        sampled_indices = idx_range[row, column] + 1

        return row, sampled_indices, label_adjust

    def forward_features(self, x):
        B, C, H, W = x.shape
        xs = []
        for i in range(self.num_branches):
            x_ = torch.nn.functional.interpolate(x, size=(self.img_size[i], self.img_size[i]), mode='bicubic') if H != self.img_size[i] else x
            tmp = self.patch_embed[i](x_)
            cls_tokens = self.cls_token[i].expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            tmp = torch.cat((cls_tokens, tmp), dim=1)
            tmp = tmp + self.pos_embed[i]
            tmp = self.pos_drop(tmp)
            xs.append(tmp)

        mask = [torch.ones((B, xs[0].shape[1]), requires_grad=False).cuda(),
                  torch.ones((B, xs[1].shape[1]), requires_grad=False).cuda()]  # 两个通道有两个policy
        label_soft_weight = []
        sampe_ratio = None
        if self.synchronized:
            row_list = []
            column_list = []
            sampe_ratio = []
            for i in range(len(mask)):
                # bs, num_remain_tokens, soft_per_example, num_samples = None
                num_remain_tokens = int((mask[i].shape[1] - 1) * self.weight_remain)
                if self.soft_per_example:   # 这里 batch_soft和sample_soft要分开
                    this_num_samples_per = []

                    for i in range(B):
                        if len(sampe_ratio) == 0:
                            this_num_samples = self.num_sample(sigma=num_remain_tokens / self.sigma,
                                                               limit=num_remain_tokens * self.limit)
                            this_num_samples_per.append(this_num_samples)
                        else:
                            cut_ratio = 1 - sampe_ratio[0][i]
                            this_num_samples = int(cut_ratio * num_remain_tokens)
                            this_num_samples_per.append(this_num_samples)

                    row, column, this_sampe_ratio = self.generate_row_column(B, num_remain_tokens,
                                                                             self.soft_per_example, this_num_samples_per)
                else:
                    if len(sampe_ratio) == 0:
                        this_num_samples = self.num_sample(sigma=num_remain_tokens / self.sigma,
                                                           limit=num_remain_tokens * self.limit)
                    else:
                        if len(sampe_ratio[0]) == 1:   # 说明是batch_soft, 一个batch只有一个soft
                            cut_ratio = 1 - sampe_ratio[0][0]
                            this_num_samples = int(cut_ratio * num_remain_tokens)

                    row, column, this_sampe_ratio = self.generate_row_column(B, num_remain_tokens, self.soft_per_example, this_num_samples)

                row_list.append(row)
                column_list.append(column)
                sampe_ratio.append(this_sampe_ratio)
        else:
            row_list = [None, None]
            column_list = [None, None]
            sampe_ratio = [None, None]

        for idx_blk, blk in enumerate(self.blocks):
            if self.training:
                if idx_blk == self.layer_cut:
                    # 因为是在第layer_cut层的输出来计算att的，并且enumerate是从0开始的，所以att和生成policy操作放到当前层的前面。
                    with torch.no_grad():
                        att_score_list = [this_tokens.detach()[:, 0:1] @ this_tokens.detach()[:, 1:].permute(0,2,1) for this_tokens in xs]     # cls与token_patch计算分数
                        att_score_list = [F.softmax(i, dim=-1).squeeze(1) for i in att_score_list]

                        percentile = 1 - self.weight_remain
                        for i, att_scores in enumerate(att_score_list):
                            # 将att_scores按行排序
                            _, idx = torch.sort(att_scores, dim=-1)

                            # 计算前percentile分位数所在的位置na
                            k = int(idx.size(-1) * percentile)

                            # 将样本数量限制在最小值和最大值之间
                            idx_range = idx[:, k:]

                            # flag
                            row, sampled_indices, label_adjust = self.sample_mask_token_debug(idx_range, row_list[i], column_list[i], sampe_ratio[i])

                            mask[i][row, sampled_indices] = 0.

                            label_soft_weight.append(label_adjust)

            xs = blk(xs, mask)


        # NOTE: was before branch token section, move to here to assure all branch token are before layer norm
        xs = [self.norm[i](x) for i, x in enumerate(xs)]
        out = [x[:, 0] for x in xs]
        # 已经检查过，label_soft_weight.mean(dim=0)，无论是whole_batch一个soft还是每个样例一个soft，都可以适配
        # 当每个样例一个soft，append后的label_soft_weight.shape=[2,bs]
        return out, torch.tensor(label_soft_weight).mean(dim=0)

    def generate_mask(self, x):
        # 注意，只有self.training的时候才可以使用generat_mask
        with torch.no_grad():
            self.eval()
            B, C, H, W = x.shape
            xs = []
            for i in range(self.num_branches):
                x_ = torch.nn.functional.interpolate(x, size=(self.img_size[i], self.img_size[i]), mode='bicubic') if H != self.img_size[i] else x
                tmp = self.patch_embed[i](x_)
                cls_tokens = self.cls_token[i].expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
                tmp = torch.cat((cls_tokens, tmp), dim=1)
                tmp = tmp + self.pos_embed[i]
                tmp = self.pos_drop(tmp)
                xs.append(tmp)

            mask = [torch.ones((B, xs[0].shape[1]), requires_grad=False).cuda(),
                      torch.ones((B, xs[1].shape[1]), requires_grad=False).cuda()]  # 两个通道有两个policy
            label_soft_weight = []
            sampe_ratio = None
            # 在这里生成传入row和column，不生成就是两个通道分开
            if self.synchronized:
                row_list = []
                column_list = []
                sampe_ratio = []
                for i in range(len(mask)):
                    # bs, num_remain_tokens, soft_per_example, num_samples = None
                    num_remain_tokens = int((mask[i].shape[1] - 1) * self.weight_remain)
                    if self.soft_per_example:   # 这里 batch_soft和sample_soft要分开
                        this_num_samples_per = []

                        for i in range(B):
                            if len(sampe_ratio) == 0:
                                this_num_samples = self.num_sample(sigma=num_remain_tokens / self.sigma,
                                                                   limit=num_remain_tokens * self.limit)
                                this_num_samples_per.append(this_num_samples)
                            else:
                                cut_ratio = 1 - sampe_ratio[0][i]
                                this_num_samples = int(cut_ratio * num_remain_tokens)
                                this_num_samples_per.append(this_num_samples)

                        row, column, this_sampe_ratio = self.generate_row_column(B, num_remain_tokens,
                                                                                 self.soft_per_example, this_num_samples_per)
                    else:
                        if len(sampe_ratio) == 0:
                            this_num_samples = self.num_sample(sigma=num_remain_tokens / self.sigma,
                                                               limit=num_remain_tokens * self.limit)
                        else:
                            if len(sampe_ratio[0]) == 1:   # 说明是batch_soft, 一个batch只有一个soft
                                cut_ratio = 1 - sampe_ratio[0][0]
                                this_num_samples = int(cut_ratio * num_remain_tokens)

                        row, column, this_sampe_ratio = self.generate_row_column(B, num_remain_tokens, self.soft_per_example, this_num_samples)

                    row_list.append(row)
                    column_list.append(column)
                    sampe_ratio.append(this_sampe_ratio)
            else:
                row_list = [None, None]
                column_list = [None, None]
                sampe_ratio = [None, None]

            # get_feature
            for blk in self.blocks:
                xs = blk(xs)

            # generate_mask
            att_score_list = [this_tokens.detach()[:, 0:1] @ this_tokens.detach()[:, 1:].permute(0,2,1) for this_tokens in xs]     # cls与token_patch计算分数
            att_score_list = [F.softmax(i, dim=-1).squeeze(1) for i in att_score_list]

            percentile = 1 - self.weight_remain
            for i, att_scores in enumerate(att_score_list):
                # 将att_scores按行排序
                _, idx = torch.sort(att_scores, dim=-1)

                # 计算前percentile分位数所在的位置
                k = int(idx.size(-1) * percentile)

                # 将样本数量限制在最小值和最大值之间
                idx_range = idx[:, k:]

                # flag
                row, sampled_indices, label_adjust = self.sample_mask_token_debug(idx_range, row_list[i], column_list[i], sampe_ratio[i])

                mask[i][row, sampled_indices] = 0.

                label_soft_weight.append(label_adjust)

            self.train()

        xs = [self.norm[i](x) for i, x in enumerate(xs)]
        out = [x[:, 0] for x in xs]
        return out, torch.tensor(label_soft_weight).mean(dim=0)

    def forward(self, x, targets=None, phase='train', similarity=True):
        xs, label_soft_weight = self.forward_features(x)   # xs是各尺寸的cls_token
        if self.layer_cut == -1:
            label_soft_weight = None

        # 分类头得到logits
        ce_logits = [self.head[i](x) for i, x in enumerate(xs)]

        # 这里是将两个通道的结果平均
        ce_logits = torch.mean(torch.stack(ce_logits, dim=0), dim=0)

        # hash_layer
        out = self.hash_layer(ce_logits)

        return out, label_soft_weight


def load_partial_wegiths(pretrained_dict, new_model):
    model_dict = new_model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    new_model.load_state_dict(model_dict)
    return new_model


@register_model
def crossvit_small_224_cut(pretrained=False, **kwargs):
    model = VisionTransformer(img_size=[240, 224],
                              patch_size=[12, 16], embed_dim=[192, 384], depth=[[1, 4, 0], [1, 4, 0], [1, 4, 0]],
                              num_heads=[6, 6], mlp_ratio=[4, 4, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_small_224'], map_location='cpu')
        model = load_partial_wegiths(state_dict, model)
    return model
