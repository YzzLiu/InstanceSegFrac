from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Type, List, Tuple, Literal, Optional

from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from nnunetv2.nets.ContrastiveLearning.contrastiveNets.ContrastiveAttetion import \
    CFA, SSA, MultiScaleSkipEnhancer, SPMB_Full, VoxelProjectionHead, GlobalProjectionHead

import torch
import torch.nn as nn


class ContrastivePairAttetionMergedTraining(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False,
                 projection_dim: int = 32,
                 multi_scale_contrastive: bool = False, # True if using multi-scale contrastive learning, False for single scale
                 K_memory: int = 100,  # Memory bank size
                 use_momentum_update: bool = True,
                 batch_size = 4
                 ):
        super().__init__()

        self.deep_supervision = deep_supervision
        self.multi_scale_contrastive = multi_scale_contrastive
        self.use_momentum_update = use_momentum_update

        self.encoder = PlainConvEncoder(
            input_channels, n_stages, features_per_stage,
            conv_op, kernel_sizes, strides, n_conv_per_stage,
            conv_bias, norm_op, norm_op_kwargs,
            dropout_op, dropout_op_kwargs,
            nonlin, nonlin_kwargs,
            return_skips=True, nonlin_first=nonlin_first
        )

        self.decoder = UNetDecoder(
            self.encoder, num_classes, n_conv_per_stage_decoder,
            deep_supervision, nonlin_first=nonlin_first
        )

        target_channels = self.encoder.output_channels if multi_scale_contrastive else [self.encoder.output_channels[-1]]

        self.voxel_proj_heads = nn.ModuleList([
            VoxelProjectionHead(ch, projection_dim, norm_op_kwargs) for ch in target_channels
        ])

        self.global_proj_heads = nn.ModuleList([
            GlobalProjectionHead(projection_dim) for _ in target_channels
        ])

        # ---------- Memory Attention ----------
        self.S = len(target_channels)

        self.prior_memory_bank = nn.Parameter(
            torch.randn(K_memory * self.S * batch_size, projection_dim) * 0.02, requires_grad=True)
        # self.register_buffer(
        #     "prior_memory_bank",
        #     torch.randn(K_memory * self.S * batch_size, projection_dim) * 0.02
        # )
        # self.spmb = SPMB_Full(K_memory, self.S, projection_dim, momentum=0.99, use_momentum_update=self.use_momentum_update)
        self.spmb = SPMB_Full(K_memory, self.S, projection_dim,verbose=False)

        self.global_dim = [projection_dim]*self.S 
        self.skip_enhancer = MultiScaleSkipEnhancer(features_per_stage, self.global_dim, 
                                                    cfa_embed=64, ssa_embed=32, 
                                                    num_heads=4, dropout=0.1)

    def forward(self, x):
        skips = self.encoder(x)

        voxel_feats = []
        global_feats = []

        feats_to_project = skips if self.multi_scale_contrastive else [skips[-1]]

        for i, feat in enumerate(feats_to_project):
            voxel_feat = self.voxel_proj_heads[i](feat)  # (B, F, D, H, W)
            voxel_feat_norm = F.normalize(voxel_feat,dim=1)  # (B, F, D, H, W)
            # pooled_feat = F.adaptive_avg_pool3d(voxel_feat, 1).flatten(1)  # (B, F)
            global_feat = self.global_proj_heads[i](voxel_feat_norm)  # (B, F)
            # print(f"[Scale {i}] voxel_feat shape = {voxel_feat.shape}, global_feat shape = {global_feat.shape}")

            voxel_feats.append(voxel_feat)
            global_feats.append(global_feat)

        # ---------- Memory ↔ Skip 交互 -----------------------
        ##### >>> MOD 3
        if self.training:
            global_enh, mem_enh = self.spmb(global_feats, self.prior_memory_bank)
            with torch.no_grad():
                self.prior_memory_bank.data.copy_(mem_enh.detach()) 
            global_enh, mem_enh = self.spmb(global_feats, self.prior_memory_bank)

        enhanced_skips = self.skip_enhancer(skips, global_enh) 
        seg_merged = self.decoder(enhanced_skips)
        if self.deep_supervision:
            seg_merged = seg_merged[0]
        
        # print("self.prior_memory_bank = ", self.prior_memory_bank)
        
        return seg_merged, voxel_feats, global_feats