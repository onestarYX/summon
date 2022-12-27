import os
import os.path as osp
import pickle
import numpy as np
from tqdm import tqdm
import open3d as o3d
import json
from scipy.spatial.transform import Rotation as R
import torch
import torch.nn as nn
import torch.nn.functional as F
import smplx
import trimesh
import pandas as pd
import scipy

import eulerangles

import pdb

def scipy_to_pytorch(x):
    """Convert scipy sparse matrices to pytorch sparse matrix."""
    x = scipy.sparse.coo_matrix(x)
    i = torch.LongTensor(np.array([x.row, x.col]))
    v = torch.FloatTensor(x.data)
    return torch.sparse.FloatTensor(i, v, x.shape)


def adjmat_sparse(adjmat, nsize=1):
    """Create row-normalized sparse graph adjacency matrix."""
    adjmat = scipy.sparse.csr_matrix(adjmat)
    if nsize > 1:
        orig_adjmat = adjmat.copy()
        for _ in range(1, nsize):
            adjmat = adjmat * orig_adjmat
    adjmat.data = np.ones_like(adjmat.data)
    for i in range(adjmat.shape[0]):
        adjmat[i, i] = 1
    num_neighbors = np.array(1 / adjmat.sum(axis=-1))
    adjmat = adjmat.multiply(num_neighbors)
    adjmat = scipy.sparse.coo_matrix(adjmat)
    row = adjmat.row
    col = adjmat.col
    data = adjmat.data
    i = torch.LongTensor(np.array([row, col]))
    v = torch.from_numpy(data).float()
    adjmat = torch.sparse.FloatTensor(i, v, adjmat.shape)
    return adjmat


def get_graph_params(ds_us_dir, device, layer=1, **kwargs):
    """Load and process graph adjacency matrix and upsampling/downsampling matrices."""
    device = torch.device(device)

    A = scipy.sparse.load_npz(osp.join(ds_us_dir, 'A_{}.npz'.format(layer)))
    D = scipy.sparse.load_npz(osp.join(ds_us_dir, 'D_{}.npz'.format(layer)))
    U = scipy.sparse.load_npz(osp.join(ds_us_dir, 'U_{}.npz'.format(layer)))

    D = scipy_to_pytorch(D).to(device)
    U = scipy_to_pytorch(U).to(device)
    A = adjmat_sparse(A).to(device)
    return A, U, D

class SparseMM(torch.autograd.Function):
    """Redefine sparse @ dense matrix multiplication to enable backpropagation.
    The builtin matrix multiplication operation does not support backpropagation in some cases.
    """

    @staticmethod
    def forward(ctx, sparse, dense):
        ctx.req_grad = dense.requires_grad
        ctx.save_for_backward(sparse)
        return torch.matmul(sparse, dense)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        sparse, = ctx.saved_tensors
        if ctx.req_grad:
            grad_input = torch.matmul(sparse.t(), grad_output)
        return None, grad_input

def spmm(sparse, dense):
    return SparseMM.apply(sparse, dense)

class ds_us(nn.Module):
    """docstring for ds_us."""

    def __init__(self, M):
        super(ds_us, self).__init__()
        self.M = M

    def forward(self, x):
        """Upsample/downsample mesh. X: B*N*C"""
        out = []
        for i in range(x.shape[0]):
            y = x[i]
            y = spmm(self.M, y)
            out.append(y)
        x = torch.stack(out, dim=0)
        return x


def load_body_model(model_folder, num_pca_comps=6, batch_size=1, gender='male', **kwargs):
    model_params = dict(model_path=model_folder,
                        model_type='smplx',
                        ext='npz',
                        num_pca_comps=num_pca_comps,
                        create_global_orient=True,
                        create_body_pose=True,
                        create_betas=True,
                        create_left_hand_pose=True,
                        create_right_hand_pose=True,
                        create_expression=True,
                        create_jaw_pose=True,
                        create_leye_pose=True,
                        create_reye_pose=True,
                        create_transl=True,
                        batch_size=batch_size)

    body_model = smplx.create(gender=gender, **model_params)
    return body_model

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + (kmat @ kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def normalize_orientation(verts_can, associated_joints, device):
    """
    Compute a rotation about z-axis to make pose from the first frame facing directly out of the screen, then applies
    this rotation to poses from all the following frame.
    Parameters
    ----------
    verts_can: a sequence of canonical vertices.
    associate_joints: (Nverts, 1), tensor of associated joints for each vertex
    device: the device on which tensors reside
    """
    n_verts = verts_can.shape[1]
    first_frame = verts_can[0]
    joint1_indices = (associated_joints == 1)
    joint2_indices = (associated_joints == 2)
    verts_in_joint1 = first_frame[joint1_indices]
    verts_in_joint2 = first_frame[joint2_indices]
    joint1 = torch.mean(verts_in_joint1, dim=0).reshape(1, 3)
    joint2 = torch.mean(verts_in_joint2, dim=0).reshape(1, 3)
    orig_direction = (joint1 - joint2).squeeze().detach().cpu().numpy()
    orig_direction[2] = 0  # Project the original direction to the xy plane.
    dest_direction = np.array([1, 0, 0])
    rot_mat = rotation_matrix_from_vectors(orig_direction, dest_direction)
    rot_mat = torch.tensor(rot_mat, dtype=torch.float32).to(device)
    verts_can = torch.matmul(rot_mat, verts_can.permute(2, 0, 1).reshape(3, -1).to(device))
    verts_can = verts_can.reshape(3, -1, n_verts).permute(1, 2, 0)
    return verts_can

def pkl_to_canonical(pkl_file_path, device, dtype, batch_size, gender='neutral', model_folder=None, cam_path=None,
                     **kwargs):
    with open(pkl_file_path, 'rb') as f:
        param = pickle.load(f)
    num_pca_comps = 6
    body_model = load_body_model(model_folder, num_pca_comps, batch_size, gender).to(device)

    # Transform global_orient and transl using cam2world
    cam2world_trans = torch.tensor(json.load(open(cam_path, "r"))).to(device)

    # body_param_list: ['betas', 'global_orient', 'body_pose', 'transl', 'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'expression']
    body_param_list = [name for name, _ in body_model.named_parameters()]
    torch_param = {}
    for key in param.keys():
        if key in body_param_list:
            torch_param[key] = torch.tensor(param[key], dtype=torch.float32).to(device)

    torch_param['betas'] = torch_param['betas'][:, :10]
    torch_param['left_hand_pose'] = torch_param['left_hand_pose'][:, :num_pca_comps]
    torch_param['right_hand_pose'] = torch_param['right_hand_pose'][:, :num_pca_comps]

    body_model.reset_params(**torch_param)
    body_model_output = body_model(return_verts=True)
    pelvis = body_model_output.joints[:, 0, :].reshape(1, 3)
    pelvis = torch.cat((pelvis, torch.tensor([[1]]).to(device)), 1)
    pelvis = torch.matmul(cam2world_trans, pelvis.t()).t()[:, :3]
    ## Get the rotation matrix for normalizing the orientation of the body
    # joint1 = body_model_output.joints[:, 1, :].reshape(1, 3)
    # joint1 = torch.cat((joint1, torch.tensor([[1]]).to(device)), 1)
    # joint1 = torch.matmul(cam2world_trans, joint1.t()).t()[:, :3]
    # joint2 = body_model_output.joints[:, 2, :].reshape(1, 3)
    # joint2 = torch.cat((joint2, torch.tensor([[1]]).to(device)), 1)
    # joint2 = torch.matmul(cam2world_trans, joint2.t()).t()[:, :3]
    # orig_direction = (joint1 - joint2).squeeze().detach().cpu().numpy()
    # orig_direction[2] = 0    # Project the original direction to the xy plane.
    # dest_direction = np.array([1, 0, 0])
    # rot_mat = rotation_matrix_from_vectors(orig_direction, dest_direction)
    # rot_mat = torch.tensor(rot_mat, dtype=dtype).to(device)

    vertices = body_model_output.vertices.squeeze()
    vertices = torch.cat((vertices, torch.ones(vertices.shape[0], 1).to(device)), 1)
    vertices = torch.matmul(cam2world_trans, vertices.t()).t()
    vertices = vertices[:, :3]

    vertices_can = vertices - pelvis
    # joint2 = joint2 - pelvis
    # vertices_can = torch.matmul(rot_mat, (vertices_can - joint2).t()).t() + joint2

    return vertices_can, vertices


def load_scene_data(device, name, sdf_dir, use_semantics, no_obj_classes, **kwargs):
    R = torch.tensor(eulerangles.euler2mat(np.pi / 2, 0, 0, 'sxyz'), dtype=torch.float32, device=device)
    t = torch.zeros(1, 3, dtype=torch.float32, device=device)

    with open(osp.join(sdf_dir, name + '.json'), 'r') as f:
        sdf_data = json.load(f)
        grid_dim = sdf_data['dim']
        badding_val = sdf_data['badding_val']
        grid_min = torch.tensor(np.array(sdf_data['min']), dtype=torch.float32, device=device)
        grid_max = torch.tensor(np.array(sdf_data['max']), dtype=torch.float32, device=device)
        voxel_size = (grid_max - grid_min) / grid_dim
        bbox = torch.tensor(np.array(sdf_data['bbox']), dtype=torch.float32, device=device)

    sdf = np.load(osp.join(sdf_dir, name + '_sdf.npy')).astype(np.float32)
    sdf = sdf.reshape(grid_dim, grid_dim, grid_dim, 1)
    sdf = torch.tensor(sdf, dtype=torch.float32, device=device)

    semantics = scene_semantics = None
    if use_semantics:
        semantics = np.load(osp.join(sdf_dir, name + '_semantics.npy')).astype(np.float32).reshape(grid_dim, grid_dim,
                                                                                                   grid_dim, 1)
        # Map `seating=34` to `Sofa=10`. `Seating is present in `N0SittingBooth only`
        semantics[semantics == 34] = 10
        # Map falsly labelled`Shower=34` to `lightings=28`.
        semantics[semantics == 25] = 28
        scene_semantics = torch.tensor(np.unique(semantics), dtype=torch.long, device=device)
        scene_semantics = torch.zeros(1, no_obj_classes, dtype=torch.float32, device=device).scatter_(-1,
                                                                                                      scene_semantics.reshape(
                                                                                                          1, -1), 1)

        semantics = torch.tensor(semantics, dtype=torch.float32, device=device)

    return {'R': R, 't': t, 'grid_dim': grid_dim, 'grid_min': grid_min,
            'grid_max': grid_max, 'voxel_size': voxel_size,
            'bbox': bbox, 'badding_val': badding_val,
            'sdf': sdf, 'semantics': semantics, 'scene_semantics': scene_semantics}

def read_sdf(vertices, sdf_grid, grid_dim, grid_min, grid_max, mode='bilinear'):
    assert vertices.dim() == 3
    assert sdf_grid.dim() == 4
    # sdf_normals: B*dim*dim*dim*3
    batch_size = vertices.shape[0]
    nv = vertices.shape[1]
    sdf_grid = sdf_grid.unsqueeze(0).permute(0, 4, 1, 2, 3)  # B*C*D*D*D
    norm_vertices = (vertices - grid_min) / (grid_max - grid_min) * 2 - 1
    x = F.grid_sample(sdf_grid,
                      norm_vertices[:, :, [2, 1, 0]].view(batch_size, nv, 1, 1, 3),
                      padding_mode='border', mode=mode, align_corners=True)
    x = x.permute(0, 2, 3, 4, 1)
    return x

