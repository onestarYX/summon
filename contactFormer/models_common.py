import os.path as osp
import torch
import torch.utils.data
from torch import nn
import numpy as np
import trimesh
import openmesh as om
import posa_utils


def get_norm_layer(channels=None, normalization_mode=None, num_groups=None, affine=True):
    if num_groups is None:
        num_groups = num_groups
    if channels is None:
        channels = channels
    if normalization_mode is None:
        normalization_mode = normalization_mode
    if normalization_mode == 'batch_norm':
        return nn.BatchNorm1d(channels)
    elif normalization_mode == 'instance_norm':
        return (nn.InstanceNorm1d(channels))
    elif normalization_mode == 'layer_norm':
        return (nn.LayerNorm(channels))
    elif normalization_mode == 'group_norm':
        return (nn.GroupNorm(num_groups, channels, affine=affine))


class SpiralConv(nn.Module):
    def __init__(self, in_channels, out_channels, indices, add_virtual_node):
        super(SpiralConv, self).__init__()
        self.indices = indices
        self.add_virtual_node = add_virtual_node
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_length = indices.size(1)   # 9
        if self.add_virtual_node:
            self.layer = nn.Linear(in_channels * (self.seq_length + 1), out_channels)
        else:
            self.layer = nn.Linear(in_channels * self.seq_length, out_channels) # Corresponding to eqn (8) in paper
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.layer.weight)
        torch.nn.init.constant_(self.layer.bias, 0)

    def forward(self, x):
        # indices: (655, 9)
        n_nodes, _ = self.indices.size()
        bs = x.size(0)
        if x.dim() == 2:
            x = torch.index_select(x, 0, self.indices.view(-1))
            x = x.view(n_nodes, -1)
        elif x.dim() == 3:
            # x: (bs, 655, d_feat)
            if self.add_virtual_node:
                global_node_feat = torch.max(x, dim=1, keepdim=True)[0]   # (64, 1, d_feat)
                global_node_feat = global_node_feat.expand(-1, n_nodes, -1) # (64, 655, d_feat)
                x = torch.index_select(x, dim=1, index=self.indices.view(-1))  # (64, 655 * 9, d_feat)
                x = x.view(bs, n_nodes, -1)  # (bs, 655, 9 * d_feat)
                x = torch.cat((x, global_node_feat), dim=-1)    # (64, 655, 10 * d_feat)
            else:
                x = torch.index_select(x, dim=1, index=self.indices.view(-1))  # (64, 655 * 9, d_feat)
                x = x.view(bs, n_nodes, -1)  # (bs, 655, 9 * d_feat)
        else:
            # x: (bs, seg_len, 655, d_feat),
            seg_len = x.size(1)
            if self.add_virtual_node:
                global_node_feat = torch.max(x, dim=2, keepdim=True)[0]  # (64, seg_len, 1, d_feat)
                global_node_feat = global_node_feat.expand(-1, -1, n_nodes, -1) # (64, seg_len, 655, d_feat)
                x = torch.index_select(x, dim=2, index=self.indices.view(-1))  # (bs, seg_len, 655 * 9, d_feat)
                x = x.view(bs, seg_len, n_nodes, -1)  # (bs, seg_len, 655, 9 * d_feat)
                x = torch.cat((x, global_node_feat), dim=-1)  # (64, seg_len, 655, 10 * d_feat)
            else:
                x = torch.index_select(x, dim=2, index=self.indices.view(-1))  # (bs, seg_len, 655 * 9, d_feat)
                x = x.view(bs, seg_len, n_nodes, -1)  # (bs, seg_len, 655, 9 * d_feat)

        x = self.layer(x)   # (bs, seg_len, 655, 64) or (bs, 655, 64) if dim==3
        return x

    def __repr__(self):
        return '{}({}, {}, seq_length={})'.format(self.__class__.__name__,
                                                  self.in_channels,
                                                  self.out_channels,
                                                  self.seq_length)


class Spiral_block(nn.Module):
    def __init__(self, in_channels, out_channels, indices, normalization_mode=None, num_groups=None,
                 non_lin=True, add_virtual_node=False):
        # in_channels = 46 or 64, out_channels = 64, indices: (655, 9).
        super(Spiral_block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization_mode = normalization_mode
        self.non_lin = non_lin

        self.conv = SpiralConv(in_channels, out_channels, indices, add_virtual_node)  # indices: (655/164/41, 9)
        if self.normalization_mode is not None:
            if self.out_channels % num_groups != 0:
                num_groups = self.out_channels
            self.norm = get_norm_layer(self.out_channels, normalization_mode, num_groups)
        if self.non_lin:
            self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.normalization_mode is not None:
            if x.dim() == 4:
                x = self.norm(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
            else:
                x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        if self.non_lin:
            x = self.relu(x)
        return x


def load_ds_us_param(ds_us_dir, level, seq_length, use_cuda=True):
    ds_us_dir = osp.abspath(ds_us_dir)
    device = torch.device("cuda" if use_cuda else "cpu")
    level = level + 2  # Start from 2
    # m.vertices: (N, 3), m.faces: (F, 3), m--T_pose, for getting spiral indices.
    m = trimesh.load(osp.join(ds_us_dir, 'mesh_{}.obj'.format(level)), process=False)
    # spiral_indices: (N, 9), nv: N
    spiral_indices = torch.tensor(posa_utils.extract_spirals(om.TriMesh(m.vertices, m.faces), seq_length)).to(device)
    nv = m.vertices.shape[0]
    verts_T_pose = torch.tensor(m.vertices, dtype=torch.float32).to(device)

    # A: adjacency matrix; D: downsampling matrix; U: upsampling matrix
    # A: (N, N); U: (N', N); D: (N, N'); N'>N
    A, U, D = posa_utils.get_graph_params(ds_us_dir, level, use_cuda=use_cuda)
    A = A.to_dense()
    U = U.to_dense()
    D = D.to_dense()
    return nv, spiral_indices, A, U, D, verts_T_pose


class ds_us_fn(nn.Module):
    def __init__(self, M):
        super(ds_us_fn, self).__init__()
        self.M = M

    def forward(self, x):
        return torch.matmul(self.M, x)



class fc_block(nn.Module):
    def __init__(self, in_features, out_features, normalization_mode=None, drop_out=False, non_lin=True):
        super(fc_block, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.normalization_mode = normalization_mode
        self.non_lin = non_lin
        self.drop_out = drop_out

        self.lin = nn.Linear(in_features, out_features)
        if self.normalization_mode is not None:
            self.norm = get_norm_layer(self.out_features, self.normalization_mode)

        if self.non_lin:
            self.relu = nn.ReLU()
        if self.drop_out:
            self.drop_out_layer = nn.Dropout(0.5)

    def forward(self, x):
        x = self.lin(x)
        if self.normalization_mode is not None:
            x = self.norm(x)
        if self.non_lin:
            x = self.relu(x)

        return x



class GraphLin(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphLin, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layer = nn.Linear(in_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.layer.weight)
        torch.nn.init.constant_(self.layer.bias, 0)

    def forward(self, x):
        x = self.layer(x)
        return x


class GraphLin_block(nn.Module):
    def __init__(self, in_channels, out_channels, normalization_mode=None, num_groups=None,
                 drop_out=False, non_lin=True):
        super(GraphLin_block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization_mode = normalization_mode
        self.non_lin = non_lin
        self.drop_out = drop_out

        self.conv = GraphLin(in_channels, out_channels)
        if self.normalization_mode is not None:
            if self.out_channels % num_groups != 0:
                num_groups = self.out_channels
            self.norm = get_norm_layer(self.out_channels, normalization_mode, num_groups)
        if self.non_lin:
            self.relu = nn.ReLU()
        if self.drop_out:
            self.drop_out_layer = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        if self.normalization_mode is not None:
            if x.dim() == 3:
                x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                x = self.norm(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        if self.non_lin:
            x = self.relu(x)
        if self.drop_out:
            x = self.drop_out_layer(x)
        return x