import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

def contact_loss(contact_points, object_points, weight=100):
    dists = torch.cdist(contact_points, object_points)
    dists, _ = torch.min(dists, 1)
    assert dists.shape[0] == contact_points.shape[0]
    contact_loss = weight * torch.sum(dists ** 2) / contact_points.shape[0]
    return contact_loss


def compute_signed_distances(
    sdf, sdf_centroid, sdf_extents,
    query_points
):
    query_pts_norm = (query_points - sdf_centroid) * 2 / sdf_extents.max()
    nv = query_pts_norm.shape[0]
    query_pts_norm = query_pts_norm.unsqueeze(0)[:,:,[2, 1, 0]]
    
    signed_dists = F.grid_sample(sdf.unsqueeze(0).unsqueeze(0), query_pts_norm.view(1, nv, 1, 1, 3), padding_mode='border', align_corners=True)
    signed_dists = signed_dists.squeeze()
    
    return signed_dists


def penetration_loss(
    sdf, sdf_centroid, sdf_extents,
    object_points,
    pen_thresh=0,
    weight=10
):
    signed_dists = compute_signed_distances(sdf, sdf_centroid, sdf_extents, object_points)
    
    neg_dists_mask = signed_dists.lt(pen_thresh).flatten()
    neg_dists = signed_dists[neg_dists_mask] ** 2
    if len(neg_dists) == 0:
        pen_loss = torch.tensor(0.0)
    else:
        pen_loss = weight * neg_dists.sum()
        
    return pen_loss, signed_dists


def grid_search(
    obj_c,
    obj_points_centered,
    obj_center_x, obj_center_y,
    obj_min_x, obj_min_y,
    obj_max_x, obj_max_y,
    contact_points_tensor,
    contact_min_x, contact_min_y,
    contact_max_x, contact_max_y,
    sdf, sdf_centroid, sdf_extents,
    grid_search_contact_weight,
    grid_search_pen_thresh,
    grid_search_classes_pen_weight,
    device
):
    grid_best_loss = float('inf')
    grid_best_rot_deg = 0
    grid_best_transl_x = 0
    grid_best_transl_y = 0
    grid_best_points = None
    min_x_transl = contact_min_x - obj_max_x
    min_y_transl = contact_min_y - obj_max_y
    max_x_transl = contact_max_x - obj_min_x
    max_y_transl = contact_max_y - obj_min_y
    for rot_deg in range(0, 360, 10):
        r = R.from_euler('XYZ', [0, 0, rot_deg], degrees=True)
        rot_obj_pts = r.apply(obj_points_centered)
        for x_transl_step in range(11):
            for y_transl_step in range(11):
                x = min_x_transl + ((max_x_transl - min_x_transl) / 10 * x_transl_step)
                y = min_y_transl + ((max_y_transl - min_y_transl) / 10 * y_transl_step)
                trans_obj_pts = np.copy(rot_obj_pts)
                trans_obj_pts[:, 0] += obj_center_x + x
                trans_obj_pts[:, 1] += obj_center_y + y
                trans_obj_pts_tensor = torch.Tensor(trans_obj_pts).float().to(device)
                ct_loss = contact_loss(contact_points_tensor, trans_obj_pts_tensor, grid_search_contact_weight)
                pen_loss, signed_dists = penetration_loss(
                    sdf, sdf_centroid, sdf_extents, 
                    trans_obj_pts_tensor,
                    pen_thresh=grid_search_pen_thresh,
                    weight=grid_search_classes_pen_weight[obj_c]
                )      
                total_loss = ct_loss.item() + pen_loss.item()
                if total_loss < grid_best_loss:
                    grid_best_loss = total_loss
                    grid_best_rot_deg = rot_deg
                    grid_best_transl_x = x
                    grid_best_transl_y = y
                    grid_best_points = trans_obj_pts
    return grid_best_loss, grid_best_rot_deg, grid_best_transl_x, grid_best_transl_y, grid_best_points
     

def optimization(
    obj_c,
    obj_points_centered,
    grid_center_x, grid_center_y,
    grid_rot_deg,
    contact_points_tensor,
    contact_min_x, contact_min_y,
    contact_max_x, contact_max_y,
    sdf, sdf_centroid, sdf_extents,
    opt_contact_weight,
    opt_pen_thresh,
    opt_classes_pen_weight,
    lr, opt_steps,
    device
):
    r = R.from_euler('XYZ', [0, 0, grid_rot_deg], degrees=True)
    opt_points = r.apply(obj_points_centered)
    init_points = np.copy(opt_points)
    init_points[:, 0] += grid_center_x
    init_points[:, 1] += grid_center_y
    init_points_tensor = torch.Tensor(init_points).float().to(device)
    ct_loss = contact_loss(contact_points_tensor, init_points_tensor, opt_contact_weight)
    pen_loss, signed_dists = penetration_loss(
        sdf, sdf_centroid, sdf_extents, 
        init_points_tensor,
        pen_thresh=opt_pen_thresh,
        weight=opt_classes_pen_weight[obj_c]
    )
    best_loss = ct_loss + pen_loss
    best_loss = best_loss.item()
    best_rot = 0
    best_transl_x = 0
    best_transl_y = 0
    best_points = init_points
    opt_rot = nn.Parameter(torch.Tensor([0.01]).float().to(device))
    opt_transl_x = nn.Parameter(torch.Tensor([0.001]).float().to(device))
    opt_transl_y = nn.Parameter(torch.Tensor([0.001]).float().to(device))
    opt = torch.optim.Adam([opt_rot, opt_transl_x, opt_transl_y], lr=lr, weight_decay=0.0001)
    opt_points = torch.Tensor(opt_points).float().to(device)
    opt_center = torch.Tensor([grid_center_x, grid_center_y]).float().to(device)
    for opt_step in tqdm(range(opt_steps)):
        opt.zero_grad()
        rot_mat = torch.zeros(3, 3).float().to(device)
        rot_mat[0, 0] = torch.cos(opt_rot)
        rot_mat[0, 1] = -torch.sin(opt_rot)
        rot_mat[1, 0] = torch.sin(opt_rot)
        rot_mat[1, 1] = torch.cos(opt_rot)
        rot_mat[2, 2] = 1
        obj_points_curr = torch.matmul(rot_mat.unsqueeze(0), opt_points.unsqueeze(-1)).squeeze()
        obj_points_curr[:, :2] += opt_center
        obj_points_curr[:, 0] += opt_transl_x
        obj_points_curr[:, 1] += opt_transl_y
        ct_loss = contact_loss(contact_points_tensor, obj_points_curr, opt_contact_weight)
        pen_loss, signed_dists = penetration_loss(
            sdf, sdf_centroid, sdf_extents, 
            obj_points_curr,
            pen_thresh=opt_pen_thresh,
            weight=opt_classes_pen_weight[obj_c]
        )                
        total_loss = ct_loss + pen_loss
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_rot = opt_rot.item()
            best_transl_x = opt_transl_x.item()
            best_transl_y = opt_transl_y.item()
            best_points = obj_points_curr.detach().cpu().numpy()
        total_loss.backward() 
        opt.step()
    return best_loss, best_rot, best_transl_x, best_transl_y, best_points

