import numpy as np
from easydict import EasyDict as edict
from functools import partial
import torch
from geotransformer.utils.data import precompute_data_stack_mode
from geotransformer.utils.torch import to_cuda, release_cuda
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed

from geotransformer.modules.ops import point_to_node_partition, index_select
from geotransformer.modules.registration import get_node_correspondences
from geotransformer.modules.sinkhorn import LearnableLogOptimalTransport
from geotransformer.modules.geotransformer import (
    GeometricTransformer,
    SuperPointMatching,
    SuperPointTargetGenerator,
    LocalGlobalRegistration,
)
from geotransformer.modules.kpconv import ConvBlock, ResidualBlock, UnaryBlock, LastUnaryBlock, nearest_upsample


class KPConvFPN(nn.Module):
    def __init__(self, input_dim, output_dim, init_dim, kernel_size, init_radius, init_sigma, group_norm):
        super(KPConvFPN, self).__init__()

        self.encoder1_1 = ConvBlock(input_dim, init_dim, kernel_size, init_radius, init_sigma, group_norm)
        self.encoder1_2 = ResidualBlock(init_dim, init_dim * 2, kernel_size, init_radius, init_sigma, group_norm)

        self.encoder2_1 = ResidualBlock(
            init_dim * 2, init_dim * 2, kernel_size, init_radius, init_sigma, group_norm, strided=True
        )
        self.encoder2_2 = ResidualBlock(
            init_dim * 2, init_dim * 4, kernel_size, init_radius * 2, init_sigma * 2, group_norm
        )
        self.encoder2_3 = ResidualBlock(
            init_dim * 4, init_dim * 4, kernel_size, init_radius * 2, init_sigma * 2, group_norm
        )

        self.encoder3_1 = ResidualBlock(
            init_dim * 4,
            init_dim * 4,
            kernel_size,
            init_radius * 2,
            init_sigma * 2,
            group_norm,
            strided=True,
        )
        self.encoder3_2 = ResidualBlock(
            init_dim * 4, init_dim * 8, kernel_size, init_radius * 4, init_sigma * 4, group_norm
        )
        self.encoder3_3 = ResidualBlock(
            init_dim * 8, init_dim * 8, kernel_size, init_radius * 4, init_sigma * 4, group_norm
        )

        self.encoder4_1 = ResidualBlock(
            init_dim * 8,
            init_dim * 8,
            kernel_size,
            init_radius * 4,
            init_sigma * 4,
            group_norm,
            strided=True,
        )
        self.encoder4_2 = ResidualBlock(
            init_dim * 8, init_dim * 16, kernel_size, init_radius * 8, init_sigma * 8, group_norm
        )
        self.encoder4_3 = ResidualBlock(
            init_dim * 16, init_dim * 16, kernel_size, init_radius * 8, init_sigma * 8, group_norm
        )

        self.encoder5_1 = ResidualBlock(
            init_dim * 16,
            init_dim * 16,
            kernel_size,
            init_radius * 8,
            init_sigma * 8,
            group_norm,
            strided=True,
        )
        self.encoder5_2 = ResidualBlock(
            init_dim * 16, init_dim * 32, kernel_size, init_radius * 16, init_sigma * 16, group_norm
        )
        self.encoder5_3 = ResidualBlock(
            init_dim * 32, init_dim * 32, kernel_size, init_radius * 16, init_sigma * 16, group_norm
        )

        self.decoder4 = UnaryBlock(init_dim * 48, init_dim * 16, group_norm)
        self.decoder3 = UnaryBlock(init_dim * 24, init_dim * 8, group_norm)
        self.decoder2 = LastUnaryBlock(init_dim * 12, output_dim)

    def forward(self, feats, data_dict):
        feats_list = []

        points_list = data_dict['points']
        neighbors_list = data_dict['neighbors']
        subsampling_list = data_dict['subsampling']
        upsampling_list = data_dict['upsampling']

        feats_s1 = feats
        feats_s1 = self.encoder1_1(feats_s1, points_list[0], points_list[0], neighbors_list[0])
        feats_s1 = self.encoder1_2(feats_s1, points_list[0], points_list[0], neighbors_list[0])

        feats_s2 = self.encoder2_1(feats_s1, points_list[1], points_list[0], subsampling_list[0])
        feats_s2 = self.encoder2_2(feats_s2, points_list[1], points_list[1], neighbors_list[1])
        feats_s2 = self.encoder2_3(feats_s2, points_list[1], points_list[1], neighbors_list[1])

        feats_s3 = self.encoder3_1(feats_s2, points_list[2], points_list[1], subsampling_list[1])
        feats_s3 = self.encoder3_2(feats_s3, points_list[2], points_list[2], neighbors_list[2])
        feats_s3 = self.encoder3_3(feats_s3, points_list[2], points_list[2], neighbors_list[2])

        feats_s4 = self.encoder4_1(feats_s3, points_list[3], points_list[2], subsampling_list[2])
        feats_s4 = self.encoder4_2(feats_s4, points_list[3], points_list[3], neighbors_list[3])
        feats_s4 = self.encoder4_3(feats_s4, points_list[3], points_list[3], neighbors_list[3])

        feats_s5 = self.encoder5_1(feats_s4, points_list[4], points_list[3], subsampling_list[3])
        feats_s5 = self.encoder5_2(feats_s5, points_list[4], points_list[4], neighbors_list[4])
        feats_s5 = self.encoder5_3(feats_s5, points_list[4], points_list[4], neighbors_list[4])

        latent_s5 = feats_s5
        feats_list.append(feats_s5)

        latent_s4 = nearest_upsample(latent_s5, upsampling_list[3])
        latent_s4 = torch.cat([latent_s4, feats_s4], dim=1)
        latent_s4 = self.decoder4(latent_s4)
        feats_list.append(latent_s4)

        latent_s3 = nearest_upsample(latent_s4, upsampling_list[2])
        latent_s3 = torch.cat([latent_s3, feats_s3], dim=1)
        latent_s3 = self.decoder3(latent_s3)
        feats_list.append(latent_s3)

        latent_s2 = nearest_upsample(latent_s3, upsampling_list[1])
        latent_s2 = torch.cat([latent_s2, feats_s2], dim=1)
        latent_s2 = self.decoder2(latent_s2)
        feats_list.append(latent_s2)

        feats_list.reverse()

        return feats_list


class GeoTransformer(nn.Module):
    def __init__(self, cfg):
        super(GeoTransformer, self).__init__()
        self.num_points_in_patch = cfg.model.num_points_in_patch
        self.matching_radius = cfg.model.ground_truth_matching_radius

        self.backbone = KPConvFPN(
            cfg.backbone.input_dim,
            cfg.backbone.output_dim,
            cfg.backbone.init_dim,
            cfg.backbone.kernel_size,
            cfg.backbone.init_radius,
            cfg.backbone.init_sigma,
            cfg.backbone.group_norm,
        )

        self.transformer = GeometricTransformer(
            cfg.geotransformer.input_dim,
            cfg.geotransformer.output_dim,
            cfg.geotransformer.hidden_dim,
            cfg.geotransformer.num_heads,
            cfg.geotransformer.blocks,
            cfg.geotransformer.sigma_d,
            cfg.geotransformer.sigma_a,
            cfg.geotransformer.angle_k,
            reduction_a=cfg.geotransformer.reduction_a,
        )

        self.coarse_target = SuperPointTargetGenerator(
            cfg.coarse_matching.num_targets, cfg.coarse_matching.overlap_threshold
        )

        self.coarse_matching = SuperPointMatching(
            cfg.coarse_matching.num_correspondences, cfg.coarse_matching.dual_normalization
        )

        self.fine_matching = LocalGlobalRegistration(
            cfg.fine_matching.topk,
            cfg.fine_matching.acceptance_radius,
            mutual=cfg.fine_matching.mutual,
            confidence_threshold=cfg.fine_matching.confidence_threshold,
            use_dustbin=cfg.fine_matching.use_dustbin,
            use_global_score=cfg.fine_matching.use_global_score,
            correspondence_threshold=cfg.fine_matching.correspondence_threshold,
            correspondence_limit=cfg.fine_matching.correspondence_limit,
            num_refinement_steps=cfg.fine_matching.num_refinement_steps,
        )

        self.optimal_transport = LearnableLogOptimalTransport(cfg.model.num_sinkhorn_iterations)

    def forward(self, data_dict):
        output_dict = {}

        # Downsample point clouds
        feats = data_dict['features'].detach()
        transform = data_dict['transform'].detach()

        ref_length_c = data_dict['lengths'][-1][0].item()
        ref_length_f = data_dict['lengths'][1][0].item()
        ref_length = data_dict['lengths'][0][0].item()
        points_c = data_dict['points'][-1].detach()
        points_f = data_dict['points'][1].detach()
        points = data_dict['points'][0].detach()


        ref_points_c = points_c[:ref_length_c]
        src_points_c = points_c[ref_length_c:]
        ref_points_f = points_f[:ref_length_f]
        src_points_f = points_f[ref_length_f:]
        ref_points = points[:ref_length]
        src_points = points[ref_length:]

        output_dict['ref_points_c'] = ref_points_c
        output_dict['src_points_c'] = src_points_c
        output_dict['ref_points_f'] = ref_points_f
        output_dict['src_points_f'] = src_points_f
        output_dict['ref_points'] = ref_points
        output_dict['src_points'] = src_points

        # 1. Generate ground truth node correspondences
        _, ref_node_masks, ref_node_knn_indices, ref_node_knn_masks = point_to_node_partition(
            ref_points_f, ref_points_c, self.num_points_in_patch
        )
        _, src_node_masks, src_node_knn_indices, src_node_knn_masks = point_to_node_partition(
            src_points_f, src_points_c, self.num_points_in_patch
        )

        ref_padded_points_f = torch.cat([ref_points_f, torch.zeros_like(ref_points_f[:1])], dim=0)
        src_padded_points_f = torch.cat([src_points_f, torch.zeros_like(src_points_f[:1])], dim=0)
        ref_node_knn_points = index_select(ref_padded_points_f, ref_node_knn_indices, dim=0)
        src_node_knn_points = index_select(src_padded_points_f, src_node_knn_indices, dim=0)

        gt_node_corr_indices, gt_node_corr_overlaps = get_node_correspondences(
            ref_points_c,
            src_points_c,
            ref_node_knn_points,
            src_node_knn_points,
            transform,
            self.matching_radius,
            ref_masks=ref_node_masks,
            src_masks=src_node_masks,
            ref_knn_masks=ref_node_knn_masks,
            src_knn_masks=src_node_knn_masks,
        )

        output_dict['gt_node_corr_indices'] = gt_node_corr_indices
        output_dict['gt_node_corr_overlaps'] = gt_node_corr_overlaps

        # 2. KPFCNN Encoder
        feats_list = self.backbone(feats, data_dict)

        feats_c = feats_list[-1]
        feats_f = feats_list[0]

        # 3. Conditional Transformer
        ref_feats_c = feats_c[:ref_length_c]
        src_feats_c = feats_c[ref_length_c:]
        ref_feats_c, src_feats_c = self.transformer(
            ref_points_c.unsqueeze(0),
            src_points_c.unsqueeze(0),
            ref_feats_c.unsqueeze(0),
            src_feats_c.unsqueeze(0),
        )
        ref_feats_c_norm = F.normalize(ref_feats_c.squeeze(0), p=2, dim=1)
        src_feats_c_norm = F.normalize(src_feats_c.squeeze(0), p=2, dim=1)

        output_dict['ref_feats_c'] = ref_feats_c_norm
        output_dict['src_feats_c'] = src_feats_c_norm

        # 5. Head for fine level matching
        ref_feats_f = feats_f[:ref_length_f]
        src_feats_f = feats_f[ref_length_f:]
        output_dict['ref_feats_f'] = ref_feats_f
        output_dict['src_feats_f'] = src_feats_f

        # 6. Select topk nearest node correspondences
        with torch.no_grad():
            ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_matching(
                ref_feats_c_norm, src_feats_c_norm, ref_node_masks, src_node_masks
            )

            output_dict['ref_node_corr_indices'] = ref_node_corr_indices
            output_dict['src_node_corr_indices'] = src_node_corr_indices

            # 7 Random select ground truth node correspondences during training
            if self.training:
                ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_target(
                    gt_node_corr_indices, gt_node_corr_overlaps
                )

        # 7.2 Generate batched node points & feats
        ref_node_corr_knn_indices = ref_node_knn_indices[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_indices = src_node_knn_indices[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_masks = ref_node_knn_masks[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_masks = src_node_knn_masks[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_points = ref_node_knn_points[ref_node_corr_indices]  # (P, K, 3)
        src_node_corr_knn_points = src_node_knn_points[src_node_corr_indices]  # (P, K, 3)

        ref_padded_feats_f = torch.cat([ref_feats_f, torch.zeros_like(ref_feats_f[:1])], dim=0)
        src_padded_feats_f = torch.cat([src_feats_f, torch.zeros_like(src_feats_f[:1])], dim=0)
        ref_node_corr_knn_feats = index_select(ref_padded_feats_f, ref_node_corr_knn_indices, dim=0)  # (P, K, C)
        src_node_corr_knn_feats = index_select(src_padded_feats_f, src_node_corr_knn_indices, dim=0)  # (P, K, C)

        output_dict['ref_node_corr_knn_points'] = ref_node_corr_knn_points
        output_dict['src_node_corr_knn_points'] = src_node_corr_knn_points
        output_dict['ref_node_corr_knn_masks'] = ref_node_corr_knn_masks
        output_dict['src_node_corr_knn_masks'] = src_node_corr_knn_masks

        # 8. Optimal transport
        matching_scores = torch.einsum('bnd,bmd->bnm', ref_node_corr_knn_feats, src_node_corr_knn_feats)  # (P, K, K)
        matching_scores = matching_scores / feats_f.shape[1] ** 0.5
        matching_scores = self.optimal_transport(matching_scores, ref_node_corr_knn_masks, src_node_corr_knn_masks)

        output_dict['matching_scores'] = matching_scores

        # 9. Generate final correspondences during testing
        with torch.no_grad():
            if not self.fine_matching.use_dustbin:
                matching_scores = matching_scores[:, :-1, :-1]

            ref_corr_points, src_corr_points, corr_scores, estimated_transform = self.fine_matching(
                ref_node_corr_knn_points,
                src_node_corr_knn_points,
                ref_node_corr_knn_masks,
                src_node_corr_knn_masks,
                matching_scores,
                node_corr_scores,
            )

            output_dict['ref_corr_points'] = ref_corr_points
            output_dict['src_corr_points'] = src_corr_points
            output_dict['corr_scores'] = corr_scores
            output_dict['estimated_transform'] = estimated_transform

        return output_dict

def load_data(src_points, ref_points):
    src_feats = np.ones_like(src_points[:, :1])
    ref_feats = np.ones_like(ref_points[:, :1])

    data_dict = {
        "ref_points": ref_points.astype(np.float32),
        "src_points": src_points.astype(np.float32),
        "ref_feats": ref_feats.astype(np.float32),
        "src_feats": src_feats.astype(np.float32),
        'transform': np.eye(4).astype(np.float32)
    }
    return data_dict

def make_cfg():
    _C = edict()
    # common
    _C.seed = 7351

    # model - backbone
    _C.backbone = edict()
    _C.backbone.num_stages = 5
    _C.backbone.init_voxel_size = 0.3
    _C.backbone.kernel_size = 15
    _C.backbone.base_radius = 4.25
    _C.backbone.base_sigma = 2.0
    _C.backbone.init_radius = _C.backbone.base_radius * _C.backbone.init_voxel_size
    _C.backbone.init_sigma = _C.backbone.base_sigma * _C.backbone.init_voxel_size
    _C.backbone.group_norm = 32
    _C.backbone.input_dim = 1
    _C.backbone.init_dim = 64
    _C.backbone.output_dim = 256

    # model - Global
    _C.model = edict()
    _C.model.ground_truth_matching_radius = 0.6
    _C.model.num_points_in_patch = 128
    _C.model.num_sinkhorn_iterations = 100

    # model - Coarse Matching
    _C.coarse_matching = edict()
    _C.coarse_matching.num_targets = 128
    _C.coarse_matching.overlap_threshold = 0.1
    _C.coarse_matching.num_correspondences = 256
    _C.coarse_matching.dual_normalization = True

    # model - GeoTransformer
    _C.geotransformer = edict()
    _C.geotransformer.input_dim = 2048
    _C.geotransformer.hidden_dim = 128
    _C.geotransformer.output_dim = 256
    _C.geotransformer.num_heads = 4
    _C.geotransformer.blocks = ['self', 'cross', 'self', 'cross', 'self', 'cross']
    _C.geotransformer.sigma_d = 4.8
    _C.geotransformer.sigma_a = 15
    _C.geotransformer.angle_k = 3
    _C.geotransformer.reduction_a = 'max'

    # model - Fine Matching
    _C.fine_matching = edict()
    _C.fine_matching.topk = 2
    _C.fine_matching.acceptance_radius = 0.6
    _C.fine_matching.mutual = True
    _C.fine_matching.confidence_threshold = 0.05
    _C.fine_matching.use_dustbin = False
    _C.fine_matching.use_global_score = False
    _C.fine_matching.correspondence_threshold = 3
    _C.fine_matching.correspondence_limit = None
    _C.fine_matching.num_refinement_steps = 5

    # loss - Coarse level
    _C.coarse_loss = edict()
    _C.coarse_loss.positive_margin = 0.1
    _C.coarse_loss.negative_margin = 1.4
    _C.coarse_loss.positive_optimal = 0.1
    _C.coarse_loss.negative_optimal = 1.4
    _C.coarse_loss.log_scale = 40
    _C.coarse_loss.positive_overlap = 0.1

    # loss - Fine level
    _C.fine_loss = edict()
    _C.fine_loss.positive_radius = 0.6

    # loss - Overall
    _C.loss = edict()
    _C.loss.weight_coarse_loss = 1.0
    _C.loss.weight_fine_loss = 1.0

    return _C



def registration_collate_fn_stack_mode(
    data_dicts, num_stages, voxel_size, search_radius, neighbor_limits, precompute_data=True
):
    r"""Collate function for registration in stack mode.

    Points are organized in the following order: [ref_1, ..., ref_B, src_1, ..., src_B].
    The correspondence indices are within each point cloud without accumulation.

    Args:
        data_dicts (List[Dict])
        num_stages (int)
        voxel_size (float)
        search_radius (float)
        neighbor_limits (List[int])
        precompute_data (bool)

    Returns:
        collated_dict (Dict)
    """
    batch_size = len(data_dicts)
    # merge data with the same key from different samples into a list
    collated_dict = {}
    for data_dict in data_dicts:
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            if key not in collated_dict:
                collated_dict[key] = []
            collated_dict[key].append(value)

    # handle special keys: [ref_feats, src_feats] -> feats, [ref_points, src_points] -> points, lengths
    feats = torch.cat(collated_dict.pop('ref_feats') + collated_dict.pop('src_feats'), dim=0)
    points_list = collated_dict.pop('ref_points') + collated_dict.pop('src_points')
    lengths = torch.LongTensor([points.shape[0] for points in points_list])
    points = torch.cat(points_list, dim=0)

    if batch_size == 1:
        # remove wrapping brackets if batch_size is 1
        for key, value in collated_dict.items():
            collated_dict[key] = value[0]

    collated_dict['features'] = feats
    if precompute_data:
        input_dict = precompute_data_stack_mode(points, lengths, num_stages, voxel_size, search_radius, neighbor_limits)
        collated_dict.update(input_dict)
    else:
        collated_dict['points'] = points
        collated_dict['lengths'] = lengths
    collated_dict['batch_size'] = batch_size

    return collated_dict

class CorrExtractor:
    def __init__(self):
        cfg = make_cfg()
        self.neighbor_limits = [64 ,65 ,74, 80 ,79]
        model = GeoTransformer(cfg).cuda()
        state_dict = torch.load('geotransformer-kitti.pth.tar')
        model.load_state_dict(state_dict["model"])
        self.model = model
        self.cfg = cfg

    def extract_corr(self, src_points, ref_points):
        data_dict = load_data(src_points, ref_points)
        data_dict = registration_collate_fn_stack_mode(
            [data_dict], self.cfg.backbone.num_stages, self.cfg.backbone.init_voxel_size, self.cfg.backbone.init_radius, self.neighbor_limits
        )
        data_dict = to_cuda(data_dict)
        output_dict = self.model(data_dict)
        # data_dict = release_cuda(data_dict)
        # output_dict = release_cuda(output_dict)
        ref_corr_points = output_dict['ref_corr_points']
        src_corr_points = output_dict['src_corr_points']
        corr_scores = output_dict['corr_scores']
        # estimated_transform = output_dict['estimated_transform']
        
        return torch.cat([src_corr_points, ref_corr_points], dim=-1).detach(), corr_scores.detach() #, estimated_transform.detach()

if __name__ == '__main__':
    src_points = np.load('example/scan_pts_0.npy')
    ref_points = np.load('example/scan_pts_1.npy')
    data_dict = load_data(src_points, ref_points)
    corr_extractor = CorrExtractor()
    corr_extractor.extract_corr(src_points, ref_points)