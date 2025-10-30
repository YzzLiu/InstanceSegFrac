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
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd


class CFA(nn.Module):
    def __init__(self, channel_dims, global_dims, embed_dim=128, num_heads=4, dropout=0.1):

        super(CFA, self).__init__()
        assert len(channel_dims) == len(global_dims)
        self.num_scales = len(channel_dims)
        self.embed_dim = embed_dim

        self.channel_proj = nn.ModuleList([
            nn.Linear(1, embed_dim) for _ in channel_dims  
        ])
        self.global_proj = nn.ModuleList([
            nn.Linear(global_dim, embed_dim) for global_dim in global_dims
        ])

        self.weight_proj = nn.ModuleList([
            nn.Linear(embed_dim, 1) for _ in channel_dims
        ])

        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, skip_features, global_features):

        enhanced_skips = []
        B = skip_features[0].shape[0]
        for i in range(self.num_scales):
            X = skip_features[i]  # (B, C_i, D_i, H_i, W_i)
            # print("X.shape = ",X.shape)
            C = X.shape[1]
            # Flatten spatial dimensions and compute mean: (B, C)
            channel_desc = X.view(B, C, -1).mean(dim=2).unsqueeze(-1)  # (B, C, 1)

            global_token = self.global_proj[i](global_features[i])    # (B, E)
            global_token = global_token.unsqueeze(1)                 # (B, 1, E)

            channel_tokens = self.channel_proj[i](channel_desc)      # (B, C, E)

            tokens = torch.cat([global_token, channel_tokens], dim=1)

            query = tokens[:, 1:, :]   # (B, C, E)
            key = tokens              # (B, C+1, E)
            value = tokens            # (B, C+1, E)

            attn_output, _ = self.attention(query, key, value)

            out1 = self.norm1(query + self.dropout(attn_output))     # (B, C, E)

            ff_out = self.ffn(out1)                                 # (B, C, E)
            out2 = self.norm2(out1 + ff_out)                        # (B, C, E)
            channel_weights = self.weight_proj[i](out2).squeeze(-1)  # (B, C)
            channel_weights = channel_weights.sigmoid()              # (B, C)
            weight = channel_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1, 1)
            # print(weight.shape)
            X_enhanced = X + X * weight                                 # (B, C, D_i, H_i, W_i)
            enhanced_skips.append(X_enhanced)
        return enhanced_skips


class SSA(nn.Module):

    def __init__(
        self,
        channel_dims: List[int],
        embed_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
        window_size: int | Tuple[int, int, int] = 4,
    ):
        super().__init__()
        self.num_scales = len(channel_dims)
        self.embed_dim = embed_dim

        if isinstance(window_size, int):
            self.window = (window_size,) * 3
        else:
            assert len(window_size) == 3
            self.window = tuple(window_size)

        self.patch_embeds = nn.ModuleList()
        self.patch_restores = nn.ModuleList()
        for C in channel_dims:
            pe = nn.Conv3d(C, embed_dim, kernel_size=1, stride=1, bias=False)
            pr = nn.ConvTranspose3d(embed_dim, C, kernel_size=1, stride=1, bias=False)
            self.patch_embeds.append(pe)
            self.patch_restores.append(pr)

        # ③ Attention + FFN
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    # --------------------------- forward --------------------------- #
    def forward(self, skips: List[torch.Tensor]) -> List[torch.Tensor]:
        B = skips[0].shape[0]
        all_tokens, shapes = [], []

        # ------- (a) patch embed -------
        for i, x in enumerate(skips):
            _, C, D, H, W = x.shape  
            pz = max(D // self.window[0], 1)
            py = max(H // self.window[1], 1)
            px = max(W // self.window[2], 1)
            patch_size = (pz, py, px)

            pe, pr = self.patch_embeds[i], self.patch_restores[i]
            if pe.kernel_size != patch_size:
                pe.kernel_size = pr.kernel_size = patch_size
                pe.stride = pr.stride = patch_size
                nn.init.kaiming_normal_(pe.weight, mode='fan_out')
                nn.init.kaiming_normal_(pr.weight, mode='fan_out')

            x_tok = pe(x)                              # (B, E, d, h, w)
            shapes.append(x_tok.shape[2:])             # (d, h, w)
            all_tokens.append(x_tok.flatten(2)         # (B, N, E)
                                 .transpose(1, 2))

        # ------- (b) Attention -------
        tokens = torch.cat(all_tokens, dim=1)          # (B, ΣN, E)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        out1 = self.norm1(tokens + attn_out)
        out2 = self.norm2(out1 + self.ffn(out1))       # (B, ΣN, E)

        # ------- (c) restore -------
        outs, offset = [], 0
        for i, t in enumerate(skips):
            _, C, D, H, W = t.shape
            d, h, w = shapes[i]
            N = d * h * w
            patch_tok = out2[:, offset:offset + N, :] \
                            .transpose(1, 2) \
                            .reshape(B, self.embed_dim, d, h, w)
            offset += N

            x_rec = self.patch_restores[i](patch_tok)  # (B, C, D', H', W')
            if x_rec.shape[2:] != (D, H, W):
                x_rec = F.interpolate(x_rec,
                                      size=(D, H, W),
                                      mode='trilinear',
                                      align_corners=False)
            outs.append(x_rec + t)                     # residual
        return outs


class MultiScaleSkipEnhancer(nn.Module):
    def __init__(self, channel_dims, global_dims, cfa_embed=128, ssa_embed=64, num_heads=4, dropout=0.1):

        super(MultiScaleSkipEnhancer, self).__init__()
        self.cfa = CFA(channel_dims, global_dims, embed_dim=cfa_embed, num_heads=num_heads, dropout=dropout)
        # self.ssa = SSA(channel_dims, embed_dim=ssa_embed, num_heads=num_heads, dropout=dropout)
        self.ssa = SSA(channel_dims, embed_dim=ssa_embed, num_heads=num_heads, dropout=dropout)

    def forward(self, skip_features, global_features):

        skip_cfa = self.cfa(skip_features, global_features)
        skip_ssa = self.ssa(skip_cfa)
        return skip_ssa


class SPMB_Full(nn.Module):
    def __init__(self, K: int, S: int, F: int,
                 heads: int = 4,
                 verbose: bool = False):
        super().__init__()
        self.KS, self.F, self.S = K * S, F, S
        self.verbose = verbose

        # Self-attention for memory bank
        self.self_attn = nn.MultiheadAttention(F, heads, batch_first=True)
        # Cross-attention: skips query, memory bank as key/value
        self.cross_attn = nn.MultiheadAttention(F, heads, batch_first=True)
        # MLP for enhanced skip and memory
        hidden = F * 4
        self.skip_mlp = nn.Sequential(
            nn.Linear(F, hidden),
            nn.GELU(),
            nn.Linear(hidden, F)
        )
        self.mem_mlp = nn.Sequential(
            nn.Linear(F, hidden),
            nn.GELU(),
            nn.Linear(hidden, F)
        )
        self.ln_skip = nn.LayerNorm(F)
        self.ln_mem = nn.LayerNorm(F)

    def forward(self, skips_global: list, memory_bank: torch.Tensor):

        # 1. Memory self-attention & MLP & Residual
        mem = memory_bank.unsqueeze(0)  # [1, M, F]
        mem_enh, _ = self.self_attn(mem, mem, mem)  # [1, M, F]
        mem_enh = mem_enh.squeeze(0)
        mem_enh = mem_enh + self.mem_mlp(self.ln_mem(mem_enh))  # Residual MLP

        # 2. Cross attention: skips as Q, mem_enh as K/V
        enhanced_skips = []
        for g in skips_global:
            q = g.unsqueeze(0)       # [1, B, F]
            k = mem_enh.unsqueeze(0) # [1, M, F]
            v = mem_enh.unsqueeze(0)
            out, _ = self.cross_attn(q, k, v) # [1, B, F]
            out = out.squeeze(0)
            out = out + self.skip_mlp(self.ln_skip(out))  # Residual MLP
            out = out + g                                 # Residual to input
            enhanced_skips.append(out)

        enhanced_skips_tensor = torch.stack(enhanced_skips, dim=0)   # [S, B, F]
        enhanced_skips_tensor = enhanced_skips_tensor.permute(1, 0, 2).contiguous()  # [B, S, F]
        new_feats = enhanced_skips_tensor.view(-1, self.F)  # [B*S, F]

        memory_size = memory_bank.size(0)
        feat_size = new_feats.size(0)
        if feat_size >= memory_size:
            mem_new = new_feats[-memory_size:]
        else:
            mem_new = torch.cat([memory_bank, new_feats], dim=0)[-memory_size:]

        enhanced_skips_list = [enhanced_skips_tensor[:, i, :] for i in range(self.S)] 

        return enhanced_skips_list, mem_new


class VoxelProjectionHead(nn.Module):
    def __init__(self, in_channels: int, projection_dim: int, norm_op_kwargs: dict):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv3d(in_channels, projection_dim, kernel_size=1, bias=False),
            nn.InstanceNorm3d(projection_dim, **norm_op_kwargs),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(projection_dim, projection_dim, kernel_size=1, bias=False)
        )

    def forward(self, x):
        return self.proj(x)  # (B, F, D, H, W)
    
class GlobalProjectionHead(nn.Module):
    def __init__(self, projection_dim: int,hidden_dim=1024):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(projection_dim, projection_dim, bias=False),
            nn.BatchNorm1d(projection_dim, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, x):
        x = F.adaptive_avg_pool3d(x, 1).flatten(1)  # (B, F)
        return self.proj(x)

class ContrastivePairAttetion(nn.Module):
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
        self.spmb = SPMB_Full(K_memory, self.S, projection_dim, momentum=0.99, use_momentum_update=self.use_momentum_update)

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
            global_feat = self.global_proj_heads[i](voxel_feat_norm)  # (B, F)

            voxel_feats.append(voxel_feat)
            global_feats.append(global_feat)


        global_enh, mem_enh = self.spmb(global_feats, self.prior_memory_bank)
        # global_enh = global_feats
        # self.prior_memory_bank = mem_enh

        with torch.no_grad():
            self.prior_memory_bank.data.copy_(mem_enh.detach())  

        # print(len(global_enh), mem_enh)
        enhanced_skips = self.skip_enhancer(skips, global_enh) 
        seg = self.decoder(enhanced_skips)
        if self.deep_supervision:
            seg = seg[0]
        # print("self.prior_memory_bank = ", self.prior_memory_bank)
        
        return seg, voxel_feats, global_feats

class TwoClassOnePosMaskedLoss(nn.Module):
    def __init__(self,
                 temperature: float = 0.07,
                 normalize: bool = True,
                 select_mode: Literal["random", "nearest", "farthest"] = "random",
                 verbose: bool = True, 
                 topk_neg: int = 10):    
        super().__init__()
        self.tau = temperature
        self.normalize = normalize
        self.select_mode = select_mode
        self.verbose = verbose
        self.topk_neg = topk_neg

    # ------------------------------------------------------------------ #
    def forward(self, feat_cls1, feat_cls2, rng: Optional[torch.Generator] = None):
        if feat_cls1.ndim != 2 or feat_cls2.ndim != 2:
            raise ValueError("inputs must be 2-D [B,F]")
        if feat_cls1.size(1) != feat_cls2.size(1):
            raise ValueError("feature-dim mismatch")

        feats = torch.cat([feat_cls1, feat_cls2], 0)          # [B,F]
        # print("feats = ",feats)
        if self.normalize:
            feats = F.normalize(feats, dim=1)
        B, _ = feats.shape
        dev = feats.device

        sim = feats @ feats.T / self.tau                      # [B,B]

        labels = torch.cat([ torch.zeros(feat_cls1.size(0),dtype=torch.long,device=dev),
                             torch.ones( feat_cls2.size(0),dtype=torch.long,device=dev)])
        same_cls = labels[:, None] == labels[None, :]

        # 1 positive per row
        pos_idx = torch.full((B,), -1, dtype=torch.long, device=dev)
        for i in range(B):
            pool = torch.nonzero(same_cls[i] & (torch.arange(B,device=dev)!=i),
                                 as_tuple=False).squeeze(1)
            if pool.numel()==0: continue
            if self.select_mode=="random":
                g = rng if rng is not None else torch.random.default_generator
                pos_idx[i]=pool[torch.randint(pool.numel(),(1,),generator=g)]
            else:
                sims = sim[i, pool]
                pos_idx[i]=pool[sims.argmax() if self.select_mode=="nearest"
                                else sims.argmin()]

        allow_mask = ~same_cls
        for i in range(B):
            if pos_idx[i]>=0: allow_mask[i,pos_idx[i]] = True
        allow_mask.fill_diagonal_(False)
        sim_masked = sim.masked_fill(~allow_mask, -1e9)

        valid = pos_idx >= 0
        if not valid.any():
            return torch.tensor(0.0, device=dev)
        
        logits  = sim_masked[valid]
        # print("logits = ", logits)
        targets = pos_idx[valid]
        loss = F.cross_entropy(logits, targets)
        return loss

class MultiScaleGlobalContrastiveLoss(nn.Module):
    def __init__(self,
                 base_loss_fn: nn.Module,
                 enable_deep_supervision: bool = False):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.enable_deep_supervision = enable_deep_supervision

    def forward(self,
                global_feat_pos: List[torch.Tensor],
                global_feat_neg: List[torch.Tensor]) -> torch.Tensor:
        is_multiscale = self.enable_deep_supervision and len(global_feat_pos) > 1
        if not is_multiscale:
            return self.base_loss_fn(global_feat_pos[0], global_feat_neg[0])

        num_scales = len(global_feat_pos)
        raw_weights = [1 / (2 ** i) for i in range(num_scales)]  # [1.0, 0.5, 0.25, ...]
        total = sum(raw_weights)
        scale_weights = [w / total for w in raw_weights]

        total_loss = 0.0
        for i in range(num_scales):
            loss_i = self.base_loss_fn(global_feat_pos[i], global_feat_neg[i])
            total_loss += scale_weights[i] * loss_i

        return total_loss