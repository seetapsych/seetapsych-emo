# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed
from timm.layers import DropPath

from .util.pos_embed import get_2d_sincos_pos_embed
from .util.layers import Conv1DBlock, ResidualBlock1D, MultiScaleFeatureFusion


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attention=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        return x
  
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_linear = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key, value, return_attention=False):
        B, N, C = query.shape
        B_kv, N_kv, C_kv = key.shape
        
        # Check if key and value have the same shape
        assert key.shape == value.shape, "Key and value must have the same shape"

        q = self.q_linear(query).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv_linear(key).reshape(B_kv, N_kv, 2, self.num_heads, C_kv // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        if return_attention:
            attn_out, attn = self.attn(self.norm1(x), return_attention=True)
            x = x + self.drop_path(attn_out)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, attn
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, query, key, value, return_attention=False):
        normed_query = self.norm1(query)
        normed_key_value = self.norm1(key)  # Assuming key and value use the same normalization

        if return_attention:
            attn_out, attn = self.attn(normed_query, normed_key_value, value, return_attention=True)
            query = query + self.drop_path(attn_out)
            query = query + self.drop_path(self.mlp(self.norm2(query)))
            return query, attn
        else:
            query = query + self.drop_path(self.attn(normed_query, normed_key_value, value))
            query = query + self.drop_path(self.mlp(self.norm2(query)))
            return query

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, drop_path_rate=0.):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Add pose token
        self.pos_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Add id token
        self.id_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Modify pos_embed dimension
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 3, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, drop_path = drop_path_rate, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.depth = depth
        # Emotion head in pretraining
        # self.exp_head = nn.Linear(embed_dim, 768)

        # Add projection layers: output of encoder emotion head
        # self.class_proj = nn.Linear(768, 256)
        # self.au_proj = nn.Linear(768, 256)
        # self.valence_proj = nn.Linear(768, 256)
        # self.arousal_proj = nn.Linear(768, 256)

        # Add 1DCNN layers: extract cls and au features
        self.class_layers = nn.ModuleList()
        self.au_cross_attention = CrossAttentionBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=True,
                                                      qk_scale=None, drop_path = drop_path_rate, norm_layer=norm_layer)
        self.class_layers.append(Conv1DBlock(768, 512, kernel_size=3, padding=1))
        # self.au_layers.append(Conv1DBlock(768, 512, kernel_size=3, padding=1))
        for _ in range(3):
            self.class_layers.append(ResidualBlock1D(512, kernel_size=3))
            # self.au_layers.append(ResidualBlock1D(512, kernel_size=3))
        self.class_layers.append(Conv1DBlock(512, 256, kernel_size=3, padding=1))
        # self.au_layers.append(Conv1DBlock(512, 256, kernel_size=3, padding=1))

        # multiscale feature fusion
        self.class_fusion_block = MultiScaleFeatureFusion(embed_dim=768, num_layers_to_fuse=4)
        self.au_patch_fusion_block = MultiScaleFeatureFusion(embed_dim=768, num_layers_to_fuse=4)
        # self.au_token_fusion_block = MultiScaleFeatureFusion(embed_dim=768, num_layers_to_fuse=4)
        # Pooling layer: reduce N dimension of multiscale feature to 1
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # Add cls and au classification heads
        # Classifier BN layer expects (N,C,L) input, so transpose is needed, handled separately
        self.class_head = nn.Sequential()
        self.class_head.add_module('linear_trans_hidden',
                                   nn.Linear(256, 256))
        self.class_head.add_module('linear_trans_activation', nn.LeakyReLU(0.1))
        self.class_head.add_module('linear_trans_drop', nn.Dropout(0.1))
        self.class_head.add_module('linear_trans_final', nn.Linear(256, 7))

        self.au_head = nn.Sequential()
        self.au_head.add_module('linear_trans_hidden',
                                   nn.Linear(768, 256))
        self.au_head.add_module('linear_trans_activation', nn.LeakyReLU(0.1))
        self.au_head.add_module('linear_trans_drop', nn.Dropout(0.1))
        self.au_head.add_module('linear_trans_final', nn.Linear(256, 16))

        # project layer for valence and arousal
        self.proj4va = nn.Linear(768, 256)

        self.valence_head = nn.Sequential()
        self.valence_head.add_module('linear_trans_hidden',
                                     nn.Linear(512, 256))
        self.valence_head.add_module('linear_trans_activation', nn.LeakyReLU(0.1))
        self.valence_head.add_module('linear_trans_drop', nn.Dropout(0.1))
        self.valence_head.add_module('linear_trans_final', nn.Linear(256, 1))

        self.arousal_head = nn.Sequential()
        self.arousal_head.add_module('linear_trans_hidden',
                                     nn.Linear(512, 256))
        self.arousal_head.add_module('linear_trans_activation', nn.LeakyReLU(0.1))
        self.arousal_head.add_module('linear_trans_drop', nn.Dropout(0.1))
        self.arousal_head.add_module('linear_trans_final', nn.Linear(256, 1))

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), add_token_num=3)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), add_token_num=3)
        # self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        # torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs
    
    def forward(self, x):
        # embed patches
        x = self.patch_embed(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 3:, :]

        # append id, pose and cls token
        cls_token = self.cls_token + self.pos_embed[:, 2:3, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        pos_token = self.pos_token + self.pos_embed[:, 1:2, :]
        pos_tokens = pos_token.expand(x.shape[0], -1, -1)
        x = torch.cat((pos_tokens, x), dim=1)
        id_token = self.id_token + self.pos_embed[:, 0:1, :]
        id_tokens = id_token.expand(x.shape[0], -1, -1)
        x = torch.cat((id_tokens, x), dim=1)

        # Define which layer features to extract
        cls_extract_layers = [0, 3, 7, 11]
        cls_multi_scale_features = []
        au_extract_layers = [2, 5, 8, 11]
        au_multi_scale_patchs = []
        au_multi_scale_exp_tokens = []

        # Add Patch embedding
        # patch_fea = x
        # cls_multi_scale_features.append(patch_fea)
        # apply Transformer blocks and extract multiscale feature
        for i, blk in enumerate(self.blocks):
            x = blk(x, x[:,3:,:], x[:,3:,:])
            au_multi_scale_exp_tokens.append(x[:, 2, :].unsqueeze(1))

            if i in cls_extract_layers:
                cls_multi_scale_features.append(x)
            if i in au_extract_layers:
                au_multi_scale_patchs.append(x[:, 3:, :])
        # ------------Extract emotion head features---------------
        # x = self.norm(x)
        # exp_fea = self.exp_head(x[:,2])

        # # Per-task projection:
        # local_cls_fea = self.class_proj(exp_fea)
        # local_valence_fea = self.valence_proj(exp_fea)
        # local_arousal_fea = self.arousal_proj(exp_fea)
        # # --------------------------------------

        # -----------------Expression Classification & AU Branch--------------------------
        # Perform fusion
        # multi_scale_features is a list containing 4 tensors of shape (B, N, D)
        class_scale_fusion = self.class_fusion_block(cls_multi_scale_features)

        # 1DCNN feature extraction
        # Conv1d expects (B, C, L), so transpose (B, N, D) -> (B, D, N)
        class_scale_fusion = class_scale_fusion.permute(0, 2, 1)
        for layer in self.class_layers:
            class_scale_fusion = layer(class_scale_fusion)
        # Pooling
        global_class_fea = self.pool(class_scale_fusion)
        global_class_fea = global_class_fea.permute(0, 2, 1).squeeze(1)

        # au_scale_fusion = au_scale_fusion.permute(0, 2, 1)
        # for layer in self.au_layers:
        #     au_scale_fusion = layer(au_scale_fusion)
        # # Fuse local and global features
        au_scale_fusion = self.au_patch_fusion_block(au_multi_scale_patchs)
        local_au_fea = torch.cat(au_multi_scale_exp_tokens, dim=1)
        global_au_fea = self.au_cross_attention(au_scale_fusion, local_au_fea, local_au_fea)

        # Classification
        cls_preds = self.class_head(global_class_fea)
        au_preds_raw = self.au_head(global_au_fea)
        # Pool the Length dimension of au results to get final output
        au_preds_tmp = au_preds_raw.permute(0, 2, 1)
        au_preds_tmp = self.max_pool(au_preds_tmp)
        au_preds = au_preds_tmp.permute(0, 2, 1).squeeze(1)
        # ---------------------------------------------------------

        # ---------------------------Valence & Arousal Branch-----------
        frozen_class_fea = global_class_fea.detach().clone()

        # Pool combined_au_fea to get 2D feature map
        frozen_au_fea_tmp = global_au_fea.detach().clone()
        frozen_au_fea_mid = frozen_au_fea_tmp.permute(0, 2, 1)
        frozen_au_fea_mid = self.max_pool(frozen_au_fea_mid)
        frozen_au_fea = frozen_au_fea_mid.permute(0, 2, 1).squeeze(1)
        frozen_au_fea = self.proj4va(frozen_au_fea)

        combined_valence_fea = torch.cat([frozen_class_fea, frozen_au_fea], dim=-1)
        combined_arousal_fea = torch.cat([frozen_class_fea, frozen_au_fea], dim=-1)
        valence_preds = self.valence_head(combined_valence_fea)
        arousal_preds = self.arousal_head(combined_arousal_fea)
        # -----------------------------------------------------------

        return cls_preds, au_preds, valence_preds, arousal_preds
    
    def forward_attentions(self, x):
        # embed patches
        x = self.patch_embed(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 3:, :]
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, 2:3, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # apply Transformer blocks
        attentions = []
        # apply Transformer blocks
        for blk in self.blocks:
            x, attn = blk(x, return_attention=True)
            attentions.append(attn)
        return attentions


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
