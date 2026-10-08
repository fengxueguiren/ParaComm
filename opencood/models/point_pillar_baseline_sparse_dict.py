# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib
# Support F-Cooper, Self-Att, DiscoNet(wo KD), V2VNet, V2XViT, When2comm

import torch
import torch.nn as nn
from icecream import ic
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.fusion_in_one import MaxFusion, AttFusion, DiscoFusion, V2VNetFusion, V2XViTFusion, When2commFusion, MKDCooperFusion
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.models.comm_modules.learnable_dict_sparse_comm import LearnableDictSparseComm

class PointPillarBaselineSparseDict(nn.Module):
    """
    F-Cooper implementation with point pillar backbone.
    """
    def __init__(self, args):
        super(PointPillarBaselineSparseDict, self).__init__()

        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        is_resnet = args['base_bev_backbone'].get("resnet", False)
        if is_resnet:
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64) # or you can use ResNetBEVBackbone, which is stronger
        else:
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64) # or you can use ResNetBEVBackbone, which is stronger
        self.voxel_size = args['voxel_size']

        if args['fusion_method'] == "max":
            self.fusion_net = MaxFusion()
        if args['fusion_method'] == "att":
            self.fusion_net = AttFusion(args['att']['feat_dim'])
        if args['fusion_method'] == "disconet":
            self.fusion_net = DiscoFusion(args['disconet']['feat_dim'])
        if args['fusion_method'] == "v2vnet":
            self.fusion_net = V2VNetFusion(args['v2vnet'])
        if args['fusion_method'] == 'v2xvit':
            self.fusion_net = V2XViTFusion(args['v2xvit'])
        if args['fusion_method'] == 'when2comm':
            self.fusion_net = When2commFusion(args['when2comm'])
        if args['fusion_method'] == 'mkdcooper':
            self.fusion_net = MKDCooperFusion(args['mkdcooper'])
        ic('Using fusion method: {}'.format(args['fusion_method']))

        self.out_channel = sum(args['base_bev_backbone']['num_upsample_filter'])

        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.out_channel = args['shrink_header']['dim'][-1]

        self.compression = False
        if "compression" in args:
            self.compression = True
            self.naive_compressor = NaiveCompressor(self.out_channel, args['compression'])

        self.cls_head = nn.Conv2d(self.out_channel, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(self.out_channel, 7 * args['anchor_number'],
                                  kernel_size=1)
        self.use_dir = False
        if 'dir_args' in args.keys():
            self.use_dir = True
            self.dir_head = nn.Conv2d(self.out_channel, args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2
 
        if 'backbone_fix' in args.keys() and args['backbone_fix']:
            self.backbone_fix()

        self.comm_enabled = 'dict_comm' in args and args['dict_comm'].get('enabled', True)
        if self.comm_enabled:
            self.dict_comm = LearnableDictSparseComm(64, args['dict_comm'])
        else:
            raise ValueError("dict_comm must be enabled for PointPillarBaselineSparseDict")
    
    def set_trainable_by_prefix(self, trainable_prefixes, freeze_unlisted=True):
        if not trainable_prefixes:
            return

        prefix_tuple = tuple(trainable_prefixes)

        for name, param in self.named_parameters():
            matched = name.startswith(prefix_tuple)
            if freeze_unlisted:
                param.requires_grad = matched
            elif matched:
                param.requires_grad = True

    def count_trainable_params(self):
        trainable = 0
        frozen = 0
        for _, param in self.named_parameters():
            if param.requires_grad:
                trainable += param.numel()
            else:
                frozen += param.numel()
        return trainable, frozen

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        # calculate pairwise affine transformation matrix
        _, _, H0, W0 = batch_dict['spatial_features'].shape # original feature map shape H0, W0
        normalized_affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], H0, W0, self.voxel_size[0])
        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict['spatial_features_2d']

        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)

        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)

        spatial_features = spatial_features_2d
        comm_recon_loss = None
        comm_sparse_loss = None
        comm_coherence_loss = None
        comm_budget_loss = None
        if self.comm_enabled:
            comm_spatial_features = []
            recon_loss_list = []
            sparse_loss_list = []
            coherence_loss_list = []
            budget_loss_list = []

            start = 0
            for b in range(record_len.shape[0]):
                num_agent = int(record_len[b].item())
                end = start + num_agent
                scene_features = spatial_features[start:end]

                if num_agent <= 1:
                    comm_spatial_features.append(scene_features)
                else:
                    ego_feat = scene_features[0:1]
                    other_feat = scene_features[1:]
                    decoded_other_feat, one_aux = self.dict_comm(other_feat)
                    comm_spatial_features.append(torch.cat([ego_feat, decoded_other_feat], dim=0))
                    recon_loss_list.append(one_aux['recon_loss'])
                    sparse_loss_list.append(one_aux['sparse_loss'])
                    coherence_loss_list.append(one_aux['coherence_loss'])
                    budget_loss_list.append(one_aux['budget_loss'])

                start = end

            spatial_features = torch.cat(comm_spatial_features, dim=0)

            if len(recon_loss_list) > 0:
                comm_recon_loss = torch.stack(recon_loss_list).mean()
                comm_sparse_loss = torch.stack(sparse_loss_list).mean()
                comm_coherence_loss = torch.stack(coherence_loss_list).mean()
                comm_budget_loss = torch.stack(budget_loss_list).mean()
            else:
                device = spatial_features.device
                comm_recon_loss = torch.tensor(0.0, device=device)
                comm_sparse_loss = torch.tensor(0.0, device=device)
                comm_coherence_loss = torch.tensor(0.0, device=device)
                comm_budget_loss = torch.tensor(0.0, device=device)

        fused_feature = self.fusion_net(spatial_features_2d, record_len, normalized_affine_matrix)

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {'cls_preds': psm,
                       'reg_preds': rm}

        if self.use_dir:
            output_dict.update({'dir_preds': self.dir_head(fused_feature)})

        if comm_recon_loss is not None:
            output_dict.update({'comm_recon_loss': comm_recon_loss})
        if comm_sparse_loss is not None:
            output_dict.update({'comm_sparse_loss': comm_sparse_loss})
        if comm_coherence_loss is not None:
            output_dict.update({'comm_coherence_loss': comm_coherence_loss})
        if comm_budget_loss is not None:
            output_dict.update({'comm_budget_loss': comm_budget_loss})

        return output_dict
