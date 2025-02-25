
import torch
import torch.nn as nn
from network.layers_equi import VNLinearLeakyReLU, VNMaxPool_downsampling, get_nearest_index, indexing_neighbor,\
    VNlayer, Rot_VN, Ts_VN, Recon
import absl.flags as flags

from losses.fs_net_loss import fs_net_loss
from losses.prop_loss import prop_rot_loss
from losses.geometry_loss import geo_transform_loss
from tools.training_utils import get_gt_v
from network.BDF import DecoderInner


FLAGS = flags.FLAGS
EPS = 1e-6

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_surface_feature(x, k=20, idx=None, x_coord=None):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if x_coord is not None:  # dynamic knn graph
            idx = knn(x_coord, k=k)  # (batch_size, num_points, k)
        else:  # fixed knn graph with input point coordinates
            idx = knn(x, k=k)
    device = torch.device('cuda')
    index = idx

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3)
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
    cross = torch.cross(feature, x, dim=-1)

    feature = torch.cat((feature - x, x, cross), dim=3).permute(0, 3, 4, 1, 2).contiguous()

    return feature, index


def get_neighbor_feature(x, x_coord, k=None):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)

    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = knn(x_coord, k=k)

    idx = idx + idx_base

    idx = idx.view(-1)


    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3)

    feature = feature.permute(0, 3, 4, 1, 2).contiguous()

    return feature


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.dim = 128//3
        self.recon_num = 3
        self.conv_0 = VNLinearLeakyReLU(3, self.dim)
        self.conv_1 = VNlayer(self.dim, self.dim)
        self.conv_2 = VNlayer(self.dim, self.dim * 2)
        self.conv_3 = VNlayer(self.dim * 2, self.dim * 2)
        self.conv_4 = VNlayer(self.dim * 2, self.dim * 4)

        self.pool_1 = VNMaxPool_downsampling(self.dim, pooling_rate=4)
        self.pool_2 = VNMaxPool_downsampling(self.dim * 2, pooling_rate=4)

        self.vn1 = VNLinearLeakyReLU(self.dim * 10, self.dim * 5, dim=4, use_batchnorm=False)
        self.vn2 = VNLinearLeakyReLU(self.dim * 5, self.dim * 2, dim=4, use_batchnorm=False)
        self.vn3 = nn.Linear(self.dim * 2, 2, bias=False)


        self.conv_scaler = nn.Sequential(
            nn.Conv1d(1267, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Conv1d(1024, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 420, 1),
            nn.BatchNorm1d(420),
            nn.ReLU(inplace=True),
        )



    def forward(self, x, norm, cat_id):

        batch_size = x.size(0)
        num_points = x.size(3)
        coord = x.view(batch_size, -1, num_points)

        if cat_id.shape[0] == 1:
            obj_idh = cat_id.view(-1, 1).repeat(cat_id.shape[0], 1)
        else:
            obj_idh = cat_id.view(-1, 1)

        one_hot = torch.zeros(batch_size, FLAGS.obj_c).to(cat_id.device).scatter_(1, obj_idh.long(), 1)

        x0, index = get_surface_feature(x, k=20)  # B, 3, 3, N, k
        x0 = self.conv_0(x0)  # B, 128//3, 3, N, k
        x0 = x0.mean(dim=-1, keepdim=False)  # B, 128//3, 3, N

        x1 = self.conv_1(x0)  # B, 128//3, 3, N

        x2 = get_neighbor_feature(x1, coord, k=4)  # B, 128//3, 3, N, k
        x2, coord2 = self.pool_1(x2, coord)  # B, 128//3, 3, N

        x2 = self.conv_2(x2)  # B, 256//3, 3, N

        x3 = self.conv_3(x2)  # B, 256, 3, N

        x4 = get_neighbor_feature(x3, coord2, k=4)  # B, 256, 3, N, k=16
        x4, coord4 = self.pool_2(x4, coord2)  # B, 256, 3, N

        x4 = self.conv_4(x4)  # B, 512, 3, N

        coord = coord.transpose(2, 1)
        index1 = get_nearest_index(coord, coord2.transpose(2, 1))
        index2 = get_nearest_index(coord, coord4.transpose(2, 1))

        x2 = indexing_neighbor(x2, index1)
        x3 = indexing_neighbor(x3, index1)
        x4 = indexing_neighbor(x4, index2)


        one_hot = one_hot.unsqueeze(2).repeat(1, 1, num_points)  # (bs, cat_one_hot, N)
        eqv_feat = torch.cat([x0, x1, x2, x3, x4], dim=1)  # B, 420*3, N

        mean_feat = eqv_feat.mean(dim=-1, keepdim=True)

        z0 = self.vn1(mean_feat)
        z0 = self.vn2(z0)
        z0 = self.vn3(z0.transpose(1, -1)).transpose(1, -1).transpose(1, 2)
        inv_gl = torch.einsum('bijm,bjkm->bikm', mean_feat, z0).view(batch_size, -1, 1) # B, 840, 1
        inv_0 = inv_gl.repeat(1, 1, num_points)
        inv_1 = (mean_feat * coord.transpose(2, 1).unsqueeze(1)).sum(dim=2, keepdim=False)
        inv_feat = torch.cat([norm, inv_0, inv_1, one_hot], dim=1)
        inv_feat = self.conv_scaler(inv_feat)

        return eqv_feat, mean_feat, inv_feat, inv_gl



class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.rot_green = Rot_VN()
        self.rot_red = Rot_VN()
        self.face_recon = Backbone()
        self.Ts = Ts_VN()
        if FLAGS.train:
            self.sdf = DecoderInner(z_dim=420)
            self.recon = Recon()

    def forward(self, points, obj_id, query):

        points = points.permute(0, 2, 1).unsqueeze(1) # b, 1, 3, n
        c = points.mean(dim=-1, keepdim=True)
        norm = torch.norm(points - c, dim=2, keepdim=False)


        feat, gf, feat_s, inv_gf = \
            self.face_recon(points - c, norm, obj_id)

        if FLAGS.train:
            bdf = self.sdf(query - c.squeeze(1), gf, inv_gf, obj_id)
            recon = self.recon(feat, feat_s) + c.squeeze(-1)

        else:
            bdf = None
            recon = None

        #  rotation
        green_R_vec, f_green_R = self.rot_green(feat, feat_s)  # b x 4
        red_R_vec, f_red_R = self.rot_red(feat, feat_s)   # b x 4
        # normalization
        p_green_R = green_R_vec / (torch.norm(green_R_vec, dim=1, keepdim=True) + 1e-6)
        p_red_R = red_R_vec / (torch.norm(red_R_vec, dim=1, keepdim=True) + 1e-6)
        # sigmoid for confidence
        f_green_R = torch.sigmoid(f_green_R)
        f_red_R = torch.sigmoid(f_red_R)

        # translation and size

        feat_for_ts = torch.cat([feat, points - c], dim=1)
        feat_s = torch.cat([feat_s, norm], dim=1)
        T, s = self.Ts(feat_for_ts, feat_s)

        Pred_T = T + c.squeeze(1).squeeze(-1)  # bs x 3
        Pred_s = s  # this s is not the object size, it is the residual

        return p_green_R, p_red_R, f_green_R, f_red_R, Pred_T, Pred_s, bdf, recon



class SE3Pose(nn.Module):
    def __init__(self, train_stage):
        super(SE3Pose, self).__init__()
        self.posenet = Network()
        self.train_stage = train_stage
        self.loss_fs_net = fs_net_loss()
        self.loss_prop = prop_rot_loss()
        self.loss_geo = geo_transform_loss()
        self.name_fs_list = ['Rot1', 'Rot2', 'Rot1_cos', 'Rot2_cos', 'Rot_regular', 'Tran', 'Size', 'R_con']
        self.name_prop_list = ['Prop_pm', 'Prop_sym']
        self.name_geo_list = ['Geo_point']
        self.loss_bdf = nn.L1Loss()
        self.loss_nocs = nn.L1Loss()

    def forward(self, PC=None, obj_id=None,
                gt_R=None, gt_t=None, gt_s=None, mean_shape=None, sym=None, aug_bb=None,
                aug_rt_t=None, aug_rt_r=None, model_point=None, nocs_scale=None, do_loss=False):
        output_dict = {}

        PC = PC.detach()  #b, n, 3

        if FLAGS.train:
            with torch.no_grad():
                PC_da, gt_R_da, gt_t_da, gt_s_da = self.data_augment(PC, gt_R, gt_t, gt_s, mean_shape, sym, aug_bb,
                                                                     aug_rt_t, aug_rt_r, model_point, nocs_scale, obj_id)
                PC = PC_da
                gt_R = gt_R_da
                gt_t = gt_t_da
                gt_s = gt_s_da

                query, gt_bdf = self.compute_bboxsdf(PC, gt_R, gt_s, gt_s, mean_shape, sym)

        else:
            query = None

        p_green_R, p_red_R, f_green_R, f_red_R, Pred_T, Pred_s, bdf, recon = self.posenet(PC, obj_id, query)

        output_dict['mask'] = None
        output_dict['sketch'] = None
        output_dict['PC'] = PC
        output_dict['p_green_R'] = p_green_R
        output_dict['p_red_R'] = p_red_R
        output_dict['f_green_R'] = f_green_R
        output_dict['f_red_R'] = f_red_R
        output_dict['Pred_T'] = Pred_T
        output_dict['Pred_s'] = Pred_s
        output_dict['gt_R'] = gt_R
        output_dict['gt_t'] = gt_t
        output_dict['gt_s'] = gt_s
        output_dict['bdf'] = bdf

        if do_loss:
            p_recon = recon
            p_T = Pred_T
            p_s = Pred_s
            pred_fsnet_list = {
                'Rot1': p_green_R,
                'Rot1_f': f_green_R,
                'Rot2': p_red_R,
                'Rot2_f': f_red_R,
                'Recon': p_recon,
                'Tran': p_T,
                'Size': p_s,
            }

            if self.train_stage == 'Backbone_only':
                gt_green_v = None
                gt_red_v = None
            else:
                gt_green_v, gt_red_v = get_gt_v(gt_R)

            gt_fsnet_list = {
                'Rot1': gt_green_v,
                'Rot2': gt_red_v,
                'Recon': PC,
                'Tran': gt_t,
                'Size': gt_s,
            }
            fsnet_loss = self.loss_fs_net(self.name_fs_list, pred_fsnet_list, gt_fsnet_list, sym)

            pred_prop_list = {
                'Recon': p_recon,
                'Rot1': p_green_R,
                'Rot2': p_red_R,
                'Tran': p_T,
                'Scale': p_s,
                'Rot1_f': f_green_R.detach(),
                'Rot2_f': f_red_R.detach(),
            }

            gt_prop_list = {
                'Points': PC,
                'R': gt_R,
                'T': gt_t,
                'Mean_shape': mean_shape,
            }

            prop_loss = self.loss_prop(self.name_prop_list, pred_prop_list, gt_prop_list, sym)
            pred_geo_list = {
                'Rot1': p_green_R,
                'Rot2': p_red_R,
                'Tran': p_T,
                'Size': p_s,
                'Rot1_f': f_green_R.detach(),
                'Rot2_f': f_red_R.detach(),
            }

            gt_geo_list = {
                'Points': PC,
                'R': gt_R,
                'T': gt_t,
                'Mean_shape': mean_shape,
            }

            geo_loss = self.loss_geo(self.name_geo_list, pred_geo_list, gt_geo_list, sym)
            loss_dict = {}
            loss_dict['fsnet_loss'] = fsnet_loss
            loss_dict['prop_loss'] = prop_loss
            loss_dict['bdf_loss'] = 20 * self.loss_bdf(bdf, gt_bdf)
            loss_dict['geo_loss'] = geo_loss


        else:
            return output_dict

        return output_dict, loss_dict

    def data_augment(self, PC, gt_R, gt_t, gt_s, mean_shape, sym, aug_bb, aug_rt_t, aug_rt_r,
                     model_point, nocs_scale, obj_ids, check_points=False):
        """
        PC torch.Size([32, 1028, 3])
        gt_R torch.Size([32, 3, 3])
        gt_t torch.Size([32, 3])
        gt_s torch.Size([32, 3])
        mean_shape torch.Size([32, 3])
        sym torch.Size([32, 4])
        aug_bb torch.Size([32, 3])
        aug_rt_t torch.Size([32, 3])
        aug_rt_r torch.Size([32, 3, 3])
        model_point torch.Size([32, 1024, 3])
        nocs_scale torch.Size([32])
        obj_ids torch.Size([32])
        """

        def defor_3D_bb_in_batch(pc, model_point, R, t, s, sym=None, aug_bb=None):
            pc_reproj = torch.matmul(R.transpose(-1, -2), (pc - t.unsqueeze(-2)).transpose(-1, -2)).transpose(-1, -2)
            sym_aug_bb = (aug_bb + aug_bb[:, [2, 1, 0]]) / 2.0
            sym_flag = (sym[:, 0] == 1).unsqueeze(-1)
            new_aug_bb = torch.where(sym_flag, sym_aug_bb, aug_bb)
            pc_reproj = pc_reproj * new_aug_bb.unsqueeze(-2)
            model_point_new = model_point * new_aug_bb.unsqueeze(-2)
            pc_new = (torch.matmul(R, pc_reproj.transpose(-2, -1)) + t.unsqueeze(-1)).transpose(-2, -1)
            s_new = s * new_aug_bb
            return pc_new, s_new, model_point_new

        def defor_3D_rt_in_batch(pc, R, t, aug_rt_t, aug_rt_r):
            pc_new = pc + aug_rt_t.unsqueeze(-2)
            t_new = t + aug_rt_t
            pc_new = torch.matmul(aug_rt_r, pc_new.transpose(-2, -1)).transpose(-2, -1)

            R_new = torch.matmul(aug_rt_r, R)
            t_new = torch.matmul(aug_rt_r, t_new.unsqueeze(-1)).squeeze(-1)
            return pc_new, R_new, t_new

        def defor_3D_bc_in_batch(pc, R, t, s, model_point, nocs_scale):
            # resize box cage along y axis, the size s is modified
            bs = pc.size(0)
            ey_up = torch.rand((bs, 1), device=pc.device) * (1.2 - 0.8) + 0.8
            ey_down = torch.rand((bs, 1), device=pc.device) * (1.2 - 0.8) + 0.8
            pc_reproj = torch.matmul(R.transpose(-1, -2), (pc - t.unsqueeze(-2)).transpose(-1, -2)).transpose(-1, -2)

            s_y = s[..., 1].unsqueeze(-1)
            per_point_resize = (pc_reproj[..., 1] + s_y / 2.0) / s_y * (ey_up - ey_down) + ey_down
            pc_reproj[..., 0] = pc_reproj[..., 0] * per_point_resize
            pc_reproj[..., 2] = pc_reproj[..., 2] * per_point_resize
            pc_new = (torch.matmul(R, pc_reproj.transpose(-2, -1)) + t.unsqueeze(-1)).transpose(-2, -1)

            new_model_point = model_point * 1.0
            model_point_resize = (new_model_point[..., 1] + s_y / 2) / s_y * (ey_up - ey_down) + ey_down
            new_model_point[..., 0] = new_model_point[..., 0] * model_point_resize
            new_model_point[..., 2] = new_model_point[..., 2] * model_point_resize

            s_new = (torch.max(new_model_point, dim=1)[0] - torch.min(new_model_point, dim=1)[
                0]) * nocs_scale.unsqueeze(-1)
            return pc_new, s_new, ey_up, ey_down

        def defor_3D_pc(pc, gt_t, r=0.2, points_defor=None, return_defor=False):

            if points_defor is None:
                points_defor = torch.rand(pc.shape).to(pc.device) * r
            new_pc = pc + points_defor * (pc - gt_t.unsqueeze(1))
            if return_defor:
                return new_pc, points_defor
            return new_pc

        def aug_bb_with_flag(PC, gt_R, gt_t, gt_s, model_point, mean_shape, sym, aug_bb, flag):
            PC_new, gt_s_new, model_point_new = defor_3D_bb_in_batch(PC, model_point, gt_R, gt_t, gt_s + mean_shape,
                                                                     sym, aug_bb)
            gt_s_new = gt_s_new - mean_shape
            PC = torch.where(flag.unsqueeze(-1), PC_new, PC)
            gt_s = torch.where(flag, gt_s_new, gt_s)
            model_point_new = torch.where(flag.unsqueeze(-1), model_point_new, model_point)
            return PC, gt_s, model_point_new

        def aug_rt_with_flag(PC, gt_R, gt_t, aug_rt_t, aug_rt_r, flag):
            PC_new, gt_R_new, gt_t_new = defor_3D_rt_in_batch(PC, gt_R, gt_t, aug_rt_t, aug_rt_r)
            PC_new = torch.where(flag.unsqueeze(-1), PC_new, PC)
            gt_R_new = torch.where(flag.unsqueeze(-1), gt_R_new, gt_R)
            gt_t_new = torch.where(flag, gt_t_new, gt_t)
            return PC_new, gt_R_new, gt_t_new

        def aug_3D_bc_with_flag(PC, gt_R, gt_t, gt_s, model_point, nocs_scale, mean_shape, flag):
            pc_new, s_new, ey_up, ey_down = defor_3D_bc_in_batch(PC, gt_R, gt_t, gt_s + mean_shape, model_point,
                                                                 nocs_scale)
            pc_new = torch.where(flag.unsqueeze(-1), pc_new, PC)
            s_new = torch.where(flag, s_new - mean_shape, gt_s)
            return pc_new, s_new, ey_up, ey_down

        def aug_pc_with_flag(PC, gt_t, flag, aug_pc_r):
            PC_new, defor = defor_3D_pc(PC, gt_t, aug_pc_r, return_defor=True)
            PC_new = torch.where(flag.unsqueeze(-1), PC_new, PC)
            return PC_new, defor

        # augmentation
        bs = PC.shape[0]

        prob_bb = torch.rand((bs, 1), device=PC.device)
        flag = prob_bb < FLAGS.aug_bb_pro
        PC, gt_s, model_point = aug_bb_with_flag(PC, gt_R, gt_t, gt_s, model_point, mean_shape, sym, aug_bb, flag)

        prob_rt = torch.rand((bs, 1), device=PC.device)
        flag = prob_rt < FLAGS.aug_rt_pro
        PC, gt_R, gt_t = aug_rt_with_flag(PC, gt_R, gt_t, aug_rt_t, aug_rt_r, flag)

        # only do bc for mug and bowl
        prob_bc = torch.rand((bs, 1), device=PC.device)
        flag = torch.logical_and(prob_bc < FLAGS.aug_bc_pro, torch.logical_or(obj_ids == 5, obj_ids == 1).unsqueeze(-1))
        PC, gt_s, _, _ = aug_3D_bc_with_flag(PC, gt_R, gt_t, gt_s, model_point, nocs_scale, mean_shape, flag)

        prob_pc = torch.rand((bs, 1), device=PC.device)
        flag = prob_pc < FLAGS.aug_pc_pro
        PC, _ = aug_pc_with_flag(PC, gt_t, flag, FLAGS.aug_pc_r)

        return PC, gt_R, gt_t, gt_s

    def compute_bboxsdf(self, PC, gt_R, gt_t, gt_s, mean_shape, sym):

        bs = gt_R.shape[0]
        re_s = gt_s + mean_shape

        querys = []
        sdfs = []
        zero = torch.tensor([0]).cuda()

        coord_pc = PC - gt_t.unsqueeze(1)
        coord_pc = torch.bmm(coord_pc, gt_R).permute(0, 2, 1)
        coord_ball = torch.tensor([[0.], [0.], [0.]]).cuda()

        with torch.cuda.device_of(gt_R):
            for i in range(bs):
                re_s_now = re_s[i, ...]  # 3
                coord_pc_now = coord_pc[i, ...] # 3 ,n

                # coord_ball = torch.randn(3, 1024).cuda()
                # coord_ball = coord_ball / torch.norm(coord_ball, dim=0, keepdim=True)
                # rand_r = torch.rand(1, 1024).cuda() ** (1/3)
                # coord_ball = coord_ball * rand_r * torch.norm(re_s_now)

                coord = torch.cat([coord_ball, coord_pc_now], dim=1)

                if sym[i, 0] == 1:
                    r = re_s_now[0]/4 + re_s_now[2]/4
                    h = re_s_now[1]/2

                    d = torch.norm(torch.stack([coord[0], coord[2]], 0), p=2, dim=0)
                    d = torch.abs(torch.stack([d, coord[1]], 0)) - torch.tensor([[h], [r]]).cuda()

                    inside = torch.minimum(torch.max(d, dim=0)[0], zero)
                    outside = torch.norm(torch.maximum(d, zero), p=2, dim=0)
                    sdf = outside + inside

                else:

                    d = torch.abs(coord) - re_s_now.unsqueeze(1) / 2
                    outside = torch.norm(torch.maximum(d, zero), p=2, dim=0)
                    inside = torch.minimum(torch.max(d, dim=0)[0], zero)
                    sdf = outside + inside

                querys.append(coord_ball)
                sdfs.append(sdf)

            querys = torch.stack(querys, 0)
            querys = torch.bmm(gt_R, querys) + gt_t.unsqueeze(-1)

        return torch.cat([querys, PC.permute(0, 2, 1)], dim=-1), torch.stack(sdfs, 0)


