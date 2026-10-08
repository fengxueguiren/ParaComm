# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from opencood.models.sub_modules.adacom_encoder import ConvNeXt


class LearnableDictSparseComm(nn.Module):
    """
    Learnable dictionary communication module.

    It encodes BEV features into a single-channel S map, sparsifies S,
    and reconstructs BEV features with a learnable dictionary.
    """

    def __init__(self, in_channels, args):
        super(LearnableDictSparseComm, self).__init__()

        self.in_channels = in_channels
        self.dict_size = int(args.get('dict_size', 64))
        self.atom_topk = int(args.get('atom_topk', 4))

        self.sparse_method = args.get('sparse_method', 'topk').lower()
        if self.sparse_method not in ['soft_threshold', 'relu_threshold', 'topk']:
            raise ValueError('Unsupported sparse_method: %s' % self.sparse_method)
        self.soft_lambda = float(args.get('lambda', 0.02)) # only used for soft thresholding
        self.relu_threshold = float(args.get('relu_threshold', 0.02)) # only used for ReLU thresholding
        # Spatial top-k for important locations.
        self.topk = args.get('topk', None)
        self.topk_ratio = float(args.get('topk_ratio', 0.10))
        self.topk_min = int(args.get('topk_min', 1))
        self.target_comm_rate = float(args.get('target_comm_rate', -1.0))

        self.float_bits = int(args.get('float_bits', 16))
        # Sparse coefficient transmission bit-width (value bits of coeff_sparse).
        self.sparse_coeff_bits = int(args.get('sparse_coeff_bits', self.float_bits))
        self.enable_coeff_quant = bool(args.get('enable_coeff_quant', True))
        self.use_residual = bool(args.get('use_residual', False))
        self.grouped_indexing_enabled = bool(args.get('grouped_indexing_enabled', False))
        self.group_tile_size = int(args.get('group_tile_size', 8))
        self.group_atom_count = int(args.get('group_atom_count', 4))
        # When grouped indexing is enabled, choose whether to decode with
        # local gathered atoms or restore dense K-channel coefficients first.
        self.grouped_use_gather_decode = bool(args.get('grouped_use_gather_decode', False))
        # Default to strict communication simulation: only sparse S-derived
        # information is used at output, no dense feature shortcut.
        self.transmit_only_sparse = bool(args.get('transmit_only_sparse', True))
        # When true, count both sparse values and their positions.
        self.include_index_bits = bool(args.get('include_index_bits', True))

        self.recon_loss_type = str(args.get('recon_loss_type', 'l1')).lower()
        if self.recon_loss_type not in ['l1', 'mse', 'smooth_l1']:
            raise ValueError('Unsupported recon_loss_type: %s' % self.recon_loss_type)

        self.topk_train_noise = float(args.get('topk_train_noise', 1e-6))
        self.score_temperature = float(args.get('score_temperature', 1.0))
        self.score_use_sigmoid = bool(args.get('score_use_sigmoid', True))

        backbone_args = args.get('backbone', {})
        self.backbone_enabled = bool(backbone_args.get('enabled', True))
        self.backbone_type = str(backbone_args.get('type', 'convnext')).lower()
        self.backbone_dim = int(backbone_args.get('dim', in_channels))

        if self.backbone_dim == in_channels:
            self.comm_in_proj = nn.Identity()
        else:
            self.comm_in_proj = nn.Conv2d(in_channels, self.backbone_dim, kernel_size=1, bias=True)

        if self.backbone_enabled:
            if self.backbone_type != 'convnext':
                raise ValueError('Unsupported comm backbone type: %s' % self.backbone_type)
            convnext_args = {
                'dim': self.backbone_dim,
                'num_of_blocks': int(backbone_args.get('num_of_blocks', 2)),
                'kernel_size': int(backbone_args.get('kernel_size', 7)),
                'deform': bool(backbone_args.get('deform', False)),
            }
            self.comm_backbone = ConvNeXt(convnext_args)
        else:
            self.comm_backbone = nn.Identity()

        encoder_head_args = args.get('encoder_head', {})
        self.encoder_hidden_dim = int(encoder_head_args.get('hidden_dim', self.backbone_dim))
        self.encoder_num_convs = int(encoder_head_args.get('num_convs', 2))
        self.encoder_kernel_size = int(encoder_head_args.get('kernel_size', 3))

        scorer_head_args = args.get('scorer_head', {})
        self.scorer_hidden_dim = int(scorer_head_args.get('hidden_dim', self.backbone_dim))
        self.scorer_num_convs = int(scorer_head_args.get('num_convs', 2))
        self.scorer_kernel_size = int(scorer_head_args.get('kernel_size', 3))

        # Dual-head prediction on top of shared communication backbone.
        self.encoder = self._build_head(
            in_dim=self.backbone_dim,
            hidden_dim=self.encoder_hidden_dim,
            out_dim=self.dict_size,
            num_convs=self.encoder_num_convs,
            kernel_size=self.encoder_kernel_size,
        )
        self.spatial_scorer = self._build_head(
            in_dim=self.backbone_dim,
            hidden_dim=self.scorer_hidden_dim,
            out_dim=1,
            num_convs=self.scorer_num_convs,
            kernel_size=self.scorer_kernel_size,
        )
        # Linear dictionary for channel reconstruction: C x K
        self.dictionary = nn.Parameter(torch.randn(in_channels, self.dict_size) * 0.02)

        restore_hidden = int(args.get('restore_hidden', 64))
        restore_backbone_args = args.get('restore_backbone', {})
        self.share_restore_backbone_with_comm = bool(restore_backbone_args.get('share_with_comm', False))
        self.restore_backbone_enabled = bool(restore_backbone_args.get('enabled', False))
        self.restore_backbone_type = str(restore_backbone_args.get('type', 'convnext')).lower()
        self.restore_backbone_dim = int(restore_backbone_args.get('dim', in_channels))

        if self.share_restore_backbone_with_comm:
            self.restore_backbone_input_dim = self.backbone_dim
            self.restore_in_proj = None
            self.restore_backbone = None
        else:
            if self.restore_backbone_dim == in_channels:
                self.restore_in_proj = nn.Identity()
            else:
                self.restore_in_proj = nn.Conv2d(in_channels, self.restore_backbone_dim, kernel_size=1, bias=True)

            if self.restore_backbone_enabled:
                if self.restore_backbone_type != 'convnext':
                    raise ValueError('Unsupported restore backbone type: %s' % self.restore_backbone_type)
                restore_convnext_args = {
                    'dim': self.restore_backbone_dim,
                    'num_of_blocks': int(restore_backbone_args.get('num_of_blocks', 2)),
                    'kernel_size': int(restore_backbone_args.get('kernel_size', 7)),
                    'deform': bool(restore_backbone_args.get('deform', False)),
                }
                self.restore_backbone = ConvNeXt(restore_convnext_args)
            else:
                self.restore_backbone = nn.Identity()
            self.restore_backbone_input_dim = self.restore_backbone_dim

        self.channel_restorer = nn.Sequential(
            nn.Conv2d(self.restore_backbone_input_dim, restore_hidden, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(restore_hidden, in_channels, kernel_size=3, padding=1, bias=True),
        )

        if self.use_residual:
            self.residual_gate = nn.Parameter(torch.tensor(0.5))

    def _resolve_num_groups(self, channels, target_groups=8):
        groups = max(1, min(int(target_groups), int(channels)))
        while channels % groups != 0 and groups > 1:
            groups -= 1
        return groups

    def _build_head(self, in_dim, hidden_dim, out_dim, num_convs, kernel_size):
        if num_convs <= 1:
            return nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=True)

        layers = []
        current_dim = in_dim
        padding = max(0, kernel_size // 2)
        for _ in range(num_convs - 1):
            layers.append(nn.Conv2d(current_dim, hidden_dim, kernel_size=kernel_size, padding=padding, bias=False))
            # layers.append(nn.GroupNorm(self._resolve_num_groups(hidden_dim), hidden_dim))
            layers.append(nn.GELU())
            current_dim = hidden_dim
        layers.append(nn.Conv2d(current_dim, out_dim, kernel_size=1, bias=True))
        return nn.Sequential(*layers)

    def _uniform_quantize_ste(self, x, num_bits):
        # Symmetric uniform quantization with STE for end-to-end training.
        bits = max(1, int(num_bits))
        if bits >= 32:
            return x

        qmax = float((1 << (bits - 1)) - 1)
        if qmax < 1.0:
            qmax = 1.0

        scale = torch.amax(torch.abs(x).detach())
        scale = torch.clamp(scale, min=1e-8)
        x_norm = torch.clamp(x / scale, min=-1.0, max=1.0)
        x_q = torch.round(x_norm * qmax) / qmax
        x_q = x_q * scale
        return x + (x_q - x).detach()

    def _sparsify_soft_threshold(self, coeff_dense):
        return torch.sign(coeff_dense) * torch.relu(torch.abs(coeff_dense) - self.soft_lambda)

    def _sparsify_relu_threshold(self, coeff_dense):
        return torch.relu(coeff_dense - self.relu_threshold)

    def _resolve_topk(self, hw):
        if self.topk is not None:
            k = int(self.topk)
        else:
            ratio_k = int(hw * self.topk_ratio)
            k = max(ratio_k, self.topk_min)
        return max(1, min(k, hw))

    def _spatial_topk_mask(self, score_map, random_tok_noise=1e-6):
        # score_map: [N, 1, H, W]
        n, c, h, w = score_map.shape
        assert c == 1, "Spatial score map must be single-channel."

        hw = h * w
        k = self._resolve_topk(hw)

        flat = score_map.view(n, c, hw)
        abs_flat = torch.abs(flat)
        if random_tok_noise is not None: 
            abs_flat = abs_flat + torch.rand_like(abs_flat) * random_tok_noise
        _, topk_idx = torch.topk(abs_flat, k=k, dim=-1)

        mask = torch.zeros_like(flat)
        mask.scatter_(-1, topk_idx, 1.0)
        return mask.view(n, c, h, w)

    def _atom_topk(self, coeff, atom_topk):
        # coeff: [N, K, H, W], top-k on K dimension.
        n, k, h, w = coeff.shape
        keep_k = max(1, min(int(atom_topk), k))
        abs_coeff = torch.abs(coeff)
        _, idx = torch.topk(abs_coeff, k=keep_k, dim=1)
        mask = torch.zeros_like(coeff)
        mask.scatter_(1, idx, 1.0)
        return coeff * mask, mask

    def _apply_atom_sparsify(self, coeff_dense):
        if self.sparse_method == 'soft_threshold':
            coeff_sparse = self._sparsify_soft_threshold(coeff_dense)
            atom_mask = (torch.abs(coeff_sparse) > 0).float()
            return coeff_sparse, atom_mask
        if self.sparse_method == 'relu_threshold':
            coeff_sparse = self._sparsify_relu_threshold(coeff_dense)
            atom_mask = (torch.abs(coeff_sparse) > 0).float()
            return coeff_sparse, atom_mask
        if self.sparse_method == 'topk':
            return self._atom_topk(coeff_dense, self.atom_topk)
        raise ValueError('Unsupported sparse_method: %s' % self.sparse_method)

    def _grouped_sparse_select(self, coeff_dense_masked):
        # Local shared atom set per tile, then sparse select in the local set.
        n, k, h, w = coeff_dense_masked.shape
        tile = max(1, self.group_tile_size)
        tile_h = int(math.ceil(float(h) / float(tile)))
        tile_w = int(math.ceil(float(w) / float(tile)))

        pad_h = tile_h * tile - h
        pad_w = tile_w * tile - w
        if pad_h > 0 or pad_w > 0:
            coeff_pad = F.pad(coeff_dense_masked, (0, pad_w, 0, pad_h))
        else:
            coeff_pad = coeff_dense_masked

        coeff_tile = coeff_pad.view(n, k, tile_h, tile, tile_w, tile)
        # Tile score: aggregated atom strength inside each tile.
        tile_score = torch.sum(torch.abs(coeff_tile), dim=(3, 5))

        keep_m = max(1, min(int(self.group_atom_count), k))
        _, tile_atom_idx = torch.topk(tile_score, k=keep_m, dim=1)

        # Expand tile-level atom ids to per-pixel candidate ids.
        idx_hw = tile_atom_idx.repeat_interleave(tile, dim=2).repeat_interleave(tile, dim=3)
        idx_hw = idx_hw[:, :, :h, :w]

        coeff_local = torch.gather(coeff_dense_masked, dim=1, index=idx_hw)
        coeff_local_sparse, atom_local_mask = self._apply_atom_sparsify(coeff_local)

        # Always share one coefficient value per tile and local atom while keeping zeros unchanged.
        coeff_local_pad = F.pad(coeff_local_sparse, (0, pad_w, 0, pad_h)) if (pad_h > 0 or pad_w > 0) else coeff_local_sparse
        coeff_local_tile = coeff_local_pad.view(n, keep_m, tile_h, tile, tile_w, tile)
        non_zero_tile = (torch.abs(coeff_local_tile) > 0).to(coeff_local_tile.dtype)
        count_tile = torch.sum(non_zero_tile, dim=(3, 5))
        sum_tile = torch.sum(coeff_local_tile, dim=(3, 5))
        shared_tile = torch.where(count_tile > 0,
                                  sum_tile / torch.clamp(count_tile, min=1.0),
                                  torch.zeros_like(sum_tile))

        shared_hw = shared_tile.repeat_interleave(tile, dim=2).repeat_interleave(tile, dim=3)
        shared_hw = shared_hw[:, :, :h, :w]
        non_zero_hw = (torch.abs(coeff_local_sparse) > 0).to(coeff_local_sparse.dtype)
        coeff_local_sparse = shared_hw * non_zero_hw
        atom_local_mask = (torch.abs(coeff_local_sparse) > 0).float()

        bits_global_atom = max(1, int(math.ceil(math.log2(max(2, self.dict_size)))))
        group_overhead = float(n * tile_h * tile_w * tile_atom_idx.shape[1] * bits_global_atom)

        return coeff_local_sparse, atom_local_mask, idx_hw, tile_atom_idx, group_overhead

    def _decode(self, coeff_sparse):
        # D: [C, K], A: [N, K, H, W] -> F_hat: [N, C, H, W]
        return torch.einsum('ck,nkhw->nchw', self.dictionary, coeff_sparse)

    def _decode_grouped_local(self, coeff_local, idx_hw):
        # Decode from local grouped coefficients without expanding back to K.
        # coeff_local / idx_hw: [N, M, H, W], dictionary: [C, K]
        n, m, h, w = coeff_local.shape
        dict_t = self.dictionary.t()

        idx_flat = idx_hw.permute(0, 2, 3, 1).reshape(-1, m).long()
        coeff_flat = coeff_local.permute(0, 2, 3, 1).reshape(-1, m, 1)
        atom_basis = dict_t[idx_flat]
        out_flat = torch.sum(atom_basis * coeff_flat, dim=1)
        return out_flat.view(n, h, w, self.in_channels).permute(0, 3, 1, 2)

    def _group_local_to_dense(self, coeff_local, idx_hw):
        # Convert grouped local coefficients [N, M, H, W] back to dense
        # coefficient tensor [N, K, H, W] for decode path consistency.
        n, m, h, w = coeff_local.shape
        coeff_dense = torch.zeros(
            n,
            self.dict_size,
            h,
            w,
            device=coeff_local.device,
            dtype=coeff_local.dtype,
        )
        coeff_dense.scatter_(1, idx_hw.long(), coeff_local)
        return coeff_dense

    def _reconstruction_loss(self, recon_features, spatial_features):
        if self.recon_loss_type == 'mse':
            return F.mse_loss(recon_features, spatial_features)
        elif self.recon_loss_type == 'smooth_l1':
            return F.smooth_l1_loss(recon_features, spatial_features)
        elif self.recon_loss_type == 'l1':
            return F.l1_loss(recon_features, spatial_features)
        
        raise ValueError('Unsupported recon_loss_type: %s' % self.recon_loss_type)

    def _sparse_loss(self, coeff_sparse):
        return torch.mean(torch.abs(coeff_sparse))

    def _coherence_loss(self):
        # Encourage dictionary atoms to be less redundant.
        d = F.normalize(self.dictionary, p=2, dim=0)
        gram = torch.matmul(d.t(), d)
        identity = torch.eye(self.dict_size, device=gram.device, dtype=gram.dtype)
        return torch.mean((gram - identity) ** 2)

    def _budget_loss(self, comm_rate):
        if self.target_comm_rate <= 0:
            return comm_rate * 0.0
        target = torch.tensor(self.target_comm_rate, device=comm_rate.device, dtype=comm_rate.dtype)
        return (comm_rate - target) ** 2

    def forward(self, spatial_features):
        # spatial_features: [sum(record_len), C, H, W]
        n, c, h, w = spatial_features.shape

        comm_feature = self.comm_backbone(self.comm_in_proj(spatial_features))
        coeff_dense = self.encoder(comm_feature)  # [N, K, H, W]
        spatial_score = self.spatial_scorer(comm_feature)  # [N, 1, H, W]
        spatial_score = spatial_score / max(self.score_temperature, 1e-6)
        if self.score_use_sigmoid:
            spatial_score = torch.sigmoid(spatial_score)
        noise = self.topk_train_noise if self.training else None
        spatial_mask = self._spatial_topk_mask(spatial_score, random_tok_noise=noise)
        coeff_dense_masked = coeff_dense * spatial_mask
        group_overhead_bits = 0.0
        tile_atom_idx = None
        idx_hw = None
        if self.grouped_indexing_enabled:
            coeff_sparse, atom_mask, idx_hw, tile_atom_idx, group_overhead_bits = self._grouped_sparse_select(coeff_dense_masked)
        else:
            coeff_sparse, atom_mask = self._apply_atom_sparsify(coeff_dense_masked)
        # coeff_sparse = coeff_sparse * spatial_mask
        # atom_mask = atom_mask * spatial_mask

        coeff_sparse = torch.tanh(coeff_sparse) # Optional value clipping to stabilize training and quantization when using low bit-width. Can be removed if not desired.
        if self.enable_coeff_quant:
            coeff_transmit = self._uniform_quantize_ste(coeff_sparse, self.sparse_coeff_bits)
            coeff_transmit = coeff_transmit * spatial_mask
        else:
            raise NotImplementedError('Currently only quantized coefficient transmission is implemented. Please set enable_coeff_quant to True.')
            coeff_transmit = coeff_sparse

        if self.grouped_indexing_enabled:
            if self.grouped_use_gather_decode:
                decoded_features_linear = self._decode_grouped_local(coeff_transmit, idx_hw)
            else:
                coeff_transmit_dense = self._group_local_to_dense(coeff_transmit, idx_hw)
                decoded_features_linear = self._decode(coeff_transmit_dense)
        else:
            decoded_features_linear = self._decode(coeff_transmit)
        if self.share_restore_backbone_with_comm:
            restore_feat = self.comm_backbone(self.comm_in_proj(decoded_features_linear))
        else:
            restore_feat = self.restore_backbone(self.restore_in_proj(decoded_features_linear))
        decoded_features = self.channel_restorer(restore_feat)

        if self.use_residual and (not self.transmit_only_sparse): 
            # only for debugging ablation, not for final model, can be removed if not desired.
            raise NotImplementedError('Residual fusion is not implemented in this version for communication-efficient scenarios. Please set use_residual to False.')
            gate = torch.sigmoid(self.residual_gate)
            out_features = (1.0 - gate) * spatial_features + gate * decoded_features
        else:
            out_features = decoded_features

        recon_loss = self._reconstruction_loss(out_features, spatial_features)
        sparse_loss = self._sparse_loss(coeff_sparse)
        coherence_loss = self._coherence_loss()

        atom_non_zero = (torch.abs(coeff_transmit) > 0).float().sum()
        non_zero = atom_non_zero
        total = float(n * h * w)
        comm_rate = non_zero / max(total, 1.0)
        budget_loss = self._budget_loss(comm_rate)

        aux = {
            's_dense': coeff_dense,
            's_sparse': coeff_sparse,
            's_transmit': coeff_transmit,
            's_mask': atom_mask,
            'spatial_mask': spatial_mask,
            'group_idx_hw': idx_hw,
            'group_atom_idx_tile': tile_atom_idx,
            'decoded_single_channel': decoded_features_linear,
            'recon_loss': recon_loss,
            'sparse_loss': sparse_loss,
            'coherence_loss': coherence_loss,
            'budget_loss': budget_loss,
        }

        return out_features, aux
