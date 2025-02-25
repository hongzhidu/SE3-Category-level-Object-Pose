import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, x_coord=None):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if x_coord is not None:  # dynamic knn graph
            idx = knn(x_coord, k=k)  # (batch_size, num_points, k)
        else:  # fixed knn graph with input point coordinates
            idx = knn(x, k=k)
    device = torch.device('cuda')

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

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 4, 1, 2).contiguous()

    return feature


def get_graph_feature_cross(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

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

    return feature


def get_graph_mean(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.reshape(batch_size, -1, num_points).contiguous()
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3).mean(2, keepdim=False)
    x = x.view(batch_size, num_points, num_dims, 3)

    feature = (feature - x).permute(0, 2, 3, 1).contiguous()

    return feature


def get_shell_mean_cross(x, k=10, nk=4, idx_all=None):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.reshape(batch_size, -1, num_points).contiguous()
    if idx_all is None:
        idx_all = knn(x, k=nk * k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = []
    for i in range(nk):
        idx.append(idx_all[:, :, i * k:(i + 1) * k])
        idx[i] = idx[i] + idx_base
        idx[i] = idx[i].view(-1)

    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    x = x.view(batch_size, num_points, num_dims, 3)
    feature = []
    for i in range(nk):
        feature.append(x.view(batch_size * num_points, -1)[idx[i], :])
        feature[i] = feature[i].view(batch_size, num_points, k, num_dims, 3).mean(2, keepdim=False)
        feature[i] = feature[i] - x
        cross = torch.cross(feature[i], x, dim=3)
        feature[i] = torch.cat((feature[i], cross), dim=2)

    feature = torch.cat(feature, dim=2).permute(0, 2, 3, 1).contiguous()

    return feature


class VNLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VNLinear, self).__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        x_out = self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)
        return x_out


class VNLeakyReLU(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False):
        super(VNLeakyReLU, self).__init__()
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (x * d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(2, keepdim=True)
        x_out = mask * x + (1 - mask) * (x - (dotprod / (d_norm_sq + EPS)) * d)
        return x_out


class VNLinearLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, use_batchnorm=True):
        super(VNLinearLeakyReLU, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm

        # Conv
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)

        # BatchNorm
        self.use_batchnorm = use_batchnorm
        if use_batchnorm == True:
            self.batchnorm = VNBatchNorm(out_channels, dim=dim)

        # LeakyReLU
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Conv
        p = self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)
        # InstanceNorm
        if self.use_batchnorm == True:
            p = self.batchnorm(p)
        # LeakyReLU
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (p * d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(2, keepdim=True)
        x_out = mask * p + (1 - mask) * (p - (dotprod / (d_norm_sq + EPS)) * d)
        return x_out.contiguous()


class VNBatchNorm(nn.Module):
    def __init__(self, num_features, dim):
        super(VNBatchNorm, self).__init__()
        self.dim = dim
        if dim == 3 or dim == 4:
            self.bn = nn.BatchNorm1d(num_features)
        elif dim == 5:
            self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        norm = torch.sqrt((x * x).sum(2))
        norm_bn = self.bn(norm)
        norm = norm.unsqueeze(2)
        norm_bn = norm_bn.unsqueeze(2)
        x = x / norm * norm_bn

        return x


class VNMaxPool(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False):
        super(VNMaxPool, self).__init__()
        if share_nonlinearity:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (x * d).sum(2, keepdim=True)
        idx = dotprod.max(dim=-1, keepdim=False)[1]
        index_tuple = torch.meshgrid([torch.arange(j) for j in x.size()[:-1]]) + (idx,)
        x_max = x[index_tuple]
        return x_max


class VNStdFeature(nn.Module):
    def __init__(self, in_channels, dim=4, normalize_frame=False, share_nonlinearity=False, use_batchnorm=True):
        super(VNStdFeature, self).__init__()
        self.dim = dim
        self.normalize_frame = normalize_frame
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm

        self.vn1 = VNLinearLeakyReLU(in_channels, in_channels // 2, dim=dim, share_nonlinearity=share_nonlinearity,
                                     use_batchnorm=use_batchnorm)
        self.vn2 = VNLinearLeakyReLU(in_channels // 2, in_channels // 4, dim=dim, share_nonlinearity=share_nonlinearity,
                                     use_batchnorm=use_batchnorm)
        if normalize_frame:
            self.vn_lin = nn.Linear(in_channels // 4, 2, bias=False)
        else:
            self.vn_lin = nn.Linear(in_channels // 4, 3, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        z0 = self.vn1(x)
        z0 = self.vn2(z0)
        z0 = self.vn_lin(z0.transpose(1, -1)).transpose(1, -1)

        if self.normalize_frame:
            # make z0 orthogonal. u2 = v2 - proj_u1(v2)
            v1 = z0[:, 0, :]
            # u1 = F.normalize(v1, dim=1)
            v1_norm = torch.sqrt((v1 * v1).sum(1, keepdim=True))
            u1 = v1 / (v1_norm + EPS)
            v2 = z0[:, 1, :]
            v2 = v2 - (v2 * u1).sum(1, keepdim=True) * u1
            # u2 = F.normalize(u2, dim=1)
            v2_norm = torch.sqrt((v2 * v2).sum(1, keepdim=True))
            u2 = v2 / (v2_norm + EPS)

            # compute the cross product of the two output vectors       
            u3 = torch.cross(u1, u2)
            z0 = torch.stack([u1, u2, u3], dim=1).transpose(1, 2)
        else:
            z0 = z0.transpose(1, 2)

        if self.dim == 4:
            x_std = torch.einsum('bijm,bjkm->bikm', x, z0)
        elif self.dim == 3:
            x_std = torch.einsum('bij,bjk->bik', x, z0)
        elif self.dim == 5:
            x_std = torch.einsum('bijmn,bjkmn->bikmn', x, z0)

        return x_std, z0


# Resnet Blocks
class VNResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = VNLinear(size_in, size_h)
        self.fc_1 = VNLinear(size_h, size_out)
        self.actvn_0 = VNLeakyReLU(size_in, share_nonlinearity=False)
        self.actvn_1 = VNLeakyReLU(size_h, share_nonlinearity=False)

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = VNLinear(size_in, size_out)
        # Initialization
        nn.init.zeros_(self.fc_1.map_to_feat.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn_0(x))
        dx = self.fc_1(self.actvn_1(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class VNMaxPool_downsampling(nn.Module):

    def __init__(self, in_channels, pooling_rate=4, share_nonlinearity=False):
        super(VNMaxPool_downsampling, self).__init__()

        self.pooling_rate = pooling_rate

        if share_nonlinearity:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)

    def forward(self, x, coord):

        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''

        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (x * d).sum(2, keepdim=True)
        idx = dotprod.max(dim=-1, keepdim=False)[1]
        index_tuple = torch.meshgrid([torch.arange(j) for j in x.size()[:-1]]) + (idx,)
        x_max = x[index_tuple]

        point_num = x.size(3)
        pool_num = int(point_num / self.pooling_rate)
        sample_idx = torch.randperm(point_num)[:pool_num]
        pooled_coord = coord[:, :, sample_idx]
        pooled_feature = x_max[:, :, :, sample_idx]

        return pooled_feature, pooled_coord


def get_nearest_index(target: "(bs, v1, 3)", source: "(bs, v2, 3)"):
    """
    Return: (bs, v1, 1)
    """
    inner = torch.bmm(target, source.transpose(1, 2))  # (bs, v1, v2)
    s_norm_2 = torch.sum(source ** 2, dim=2)  # (bs, v2)
    t_norm_2 = torch.sum(target ** 2, dim=2)  # (bs, v1)
    d_norm_2 = s_norm_2.unsqueeze(1) + t_norm_2.unsqueeze(2) - 2 * inner
    nearest_index = torch.topk(d_norm_2, k=1, dim=-1, largest=False)[1]
    return nearest_index


def indexing_neighbor(x, index):
    batch_size = x.size(0)
    num_dim = x.size(1)
    num_points = x.size(3)
    num_index = index.size(1)

    x = x.view(batch_size, -1, num_points).transpose(2, 1).contiguous()

    # ss = time.time()
    if batch_size == 1:
        # id_0 = torch.arange(bs).view(-1, 1,1)
        tensor_indexed = x[torch.Tensor([[0]]).long(), index[0]].unsqueeze(dim=0)
    else:
        id_0 = torch.arange(batch_size).view(-1, 1, 1).long()
        tensor_indexed = x[id_0, index]
    # ee = time.time()
    # print('tensor_indexed time: ', str(ee - ss))
    tensor_indexed = tensor_indexed.view(batch_size, num_index, num_dim, 3).permute(0, 2, 3, 1).contiguous()

    return tensor_indexed


def get_neighbor_index(vertices: "(bs, vertice_num, 3)", neighbor_num: int):
    """
    Return: (bs, vertice_num, neighbor_num)
    """
    inner = torch.bmm(vertices, vertices.transpose(1, 2))  # (bs, v, v)
    quadratic = torch.sum(vertices ** 2, dim=2)  # (bs, v)
    distance = inner * (-2) + quadratic.unsqueeze(1) + quadratic.unsqueeze(2)
    neighbor_index = torch.topk(distance, k=neighbor_num + 1, dim=-1, largest=False)[1]
    neighbor_index = neighbor_index[:, :, 1:]
    return neighbor_index





class VNlayer(nn.Module):
    def __init__(self, in_channels, out_channels, k=10):
        super(VNlayer, self).__init__()

        self.k = k
        #self.layer0 = VNLinearLeakyReLU(in_channels * 2, in_channels, dim=5)
        self.layer1 = VNLinearLeakyReLU(in_channels, out_channels, dim=5)

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(3)

        x = x.view(batch_size, -1, num_points)

        idx = knn(x, k=self.k + 1)  # b, n, k+1
        device = torch.device('cuda')

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

        idx = idx + idx_base

        idx = idx.view(-1)

        _, num_dims, _ = x.size()
        num_dims = num_dims // 3

        x = x.transpose(2, 1).contiguous()
        feature = x.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, self.k + 1, num_dims, 3)
        #x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, self.k + 1, 1, 1)
        feature = feature.permute(0, 3, 4, 1, 2).contiguous()

        #feature = self.layer0(feature)
        feature = self.layer1(feature)

        return feature.mean(dim=-1, keepdim=False)

class GateVNlayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GateVNlayer, self).__init__()

        self.layer_v0 = VNLinear(in_channels, in_channels)
        self.layer_v1 = VNLinear(in_channels, out_channels)

        self.layer_s = nn.Conv1d(in_channels * 2, out_channels, 1)

        self.act_s = nn.ReLU(inplace=True)
        self.layer_g = nn.Conv1d(out_channels, out_channels, 1)

        self.norm_s = nn.InstanceNorm1d(out_channels)


    def forward(self, v, s):

        v = self.layer_v0(v)
        norm = torch.norm(v, dim=2, keepdim=False)
        s = self.layer_s(torch.cat([norm, s], 1))
        updated_s = self.act_s(self.norm_s(s))
        gate = torch.sigmoid(self.layer_g(s))
        v = self.layer_v1(v)
        v = v * gate.unsqueeze(2)

        # norm = torch.norm(v, dim=2, keepdim=True) + 1e-8

        return v, updated_s

# class GateVNlayer(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(GateVNlayer, self).__init__()
#
#         self.layer_v0 = VNLinear(in_channels, in_channels)
#         self.layer_v1 = VNLinear(in_channels, out_channels)
#
#         self.layer_s = nn.Conv1d(in_channels * 2, out_channels, 1)
#
#         self.act_s = nn.ReLU(inplace=True)
#         self.layer_g = nn.Conv1d(in_channels * 2, in_channels, 1)
#
#         self.norm_s = nn.LayerNorm(1028)
#
#
#     def forward(self, v, s):
#
#         norm = torch.norm(v, dim=2, keepdim=False)
#         s0 = torch.cat([norm, s], 1)
#
#         updated_s = self.act_s(self.norm_s(self.layer_s(s0)))
#
#         gate = torch.sigmoid(self.layer_g(s0)).unsqueeze(2)
#         updated_v = self.layer_v0(v)
#
#         v = updated_v * gate + (1. - gate) * v
#         v = self.layer_v1(v)
#
#
#         # norm = torch.norm(v, dim=2, keepdim=True) + 1e-8
#
#         return v, updated_s

class Vdropout(nn.Module):
    def __init__(self, rate=0.2):
        super(Vdropout, self).__init__()
        self.rate = rate

    def forward(self, v):

        if not self.training:
            return v
        v = v.transpose(2, -1)

        mask = torch.bernoulli((1 - self.rate) * torch.ones(v.shape[:-1], device=v.device)).unsqueeze(-1)
        v = mask * v / (1 - self.rate)

        return v.transpose(2, -1)

class Rot_VN(nn.Module):
    def __init__(self):
        super(Rot_VN, self).__init__()

        self.rot_0 = GateVNlayer(420, 210)
        self.rot_1 = GateVNlayer(210, 126)
        self.rot_2 = GateVNlayer(126, 42)

        self.drop_s = nn.Dropout(0.5)
        self.drop_v = Vdropout(0.5)

        self.rot = VNLinear(42, 1)
        self.conf = nn.Conv1d(42, 1, 1)

    def forward(self, v, s):

        v, s = self.rot_0(v, s)
        v, s = self.rot_1(v, s)
        v, s = self.rot_2(v, s)

        v = v.mean(-1, keepdim=False)
        v = self.rot(self.drop_v(v))

        s = s.mean(-1, keepdim=True)
        s = self.conf(self.drop_s(s))

        return v.squeeze(1), s.squeeze(1).squeeze(1)


class Ts_VN(nn.Module):
    def __init__(self):
        super(Ts_VN, self).__init__()

        self.T_0 = GateVNlayer(421, 210)
        self.T_1 = GateVNlayer(210, 126)
        self.T_2 = GateVNlayer(126, 42)

        self.drop_s = nn.Dropout(0.5)
        self.drop_v = Vdropout(0.5)

        self.T = VNLinear(42, 1)
        self.scale = nn.Conv1d(42, 3, 1)

    def forward(self, v, s):
        v, s = self.T_0(v, s)
        v, s = self.T_1(v, s)
        v, s = self.T_2(v, s)

        v = v.mean(-1, keepdim=False)
        v = self.T(self.drop_v(v))

        s = s.mean(-1, keepdim=True)
        s = self.scale(self.drop_s(s))

        return v.squeeze(1), s.squeeze(-1)

class Recon(nn.Module):
    def __init__(self):
        super(Recon, self).__init__()


        self.Recon_0 = GateVNlayer(420, 210)
        self.Recon_1 = GateVNlayer(210, 126)
        self.Recon_2 = GateVNlayer(126, 42)

        self.Recon = VNLinear(42, 1)

    def forward(self, v, s):
        v, s = self.Recon_0(v, s)
        v, s = self.Recon_1(v, s)
        v, s = self.Recon_2(v, s)

        v = self.Recon(v).squeeze(1)


        return v.permute(0, 2, 1)