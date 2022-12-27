import os
import argparse
import torch
import torch.nn.functional as F


def compute_recon_loss(gt_batch, pr_batch, mask=None, semantics_w=1.0, reduction='mean', **kwargs):
    batch_size, _, n_verts, _ = gt_batch.shape

    if mask is not None:
        reduction = 'none'
        # mask = mask.unsqueeze(-1).expand(-1, -1, n_verts)
        gt_batch = gt_batch[mask == 1].unsqueeze(0)
        pr_batch = pr_batch[mask == 1].unsqueeze(0)

    targets = gt_batch.argmax(dim=-1).type(torch.long)
    recon_loss_semantics = semantics_w * F.cross_entropy(pr_batch.permute(0, 3, 1, 2), targets, reduction=reduction)
    semantics_recon_acc = (targets == torch.argmax(pr_batch, dim=-1)).float()
    if mask is not None:
        # recon_loss_semantics *= mask
        # recon_loss_semantics = torch.sum(recon_loss_semantics) / (torch.sum(mask) * n_verts)
        recon_loss_semantics = torch.mean(recon_loss_semantics)
        # semantics_recon_acc *= mask
        # semantics_recon_acc = torch.sum(semantics_recon_acc) / (torch.sum(mask) * n_verts)
        semantics_recon_acc = torch.mean(semantics_recon_acc)
    else:
        semantics_recon_acc = torch.mean(semantics_recon_acc)

    return recon_loss_semantics, semantics_recon_acc


def compute_recon_loss_posa(gt_batch, pr_batch, semantics_w=1.0, reduction='mean', **kwargs):
    batch_size, n_verts, _ = gt_batch.shape
    device = gt_batch.device
    dtype = gt_batch.dtype

    recon_loss_semantics = torch.zeros(1, dtype=dtype, device=device)
    semantics_recon_acc = torch.zeros(1, dtype=dtype, device=device)

    targets = gt_batch.argmax(dim=-1).type(torch.long).reshape(batch_size, -1)
    recon_loss_semantics = semantics_w * F.cross_entropy(pr_batch.permute(0, 2, 1), targets,
                                                         reduction=reduction)
    semantics_recon_acc = torch.mean((targets == torch.argmax(pr_batch, dim=-1)).float())

    return recon_loss_semantics, semantics_recon_acc


def compute_ir_recon_loss(gt_batch, pr_batch):
    loss_metric = torch.nn.MSELoss()
    return loss_metric(pr_batch, gt_batch)


def compute_delta(vertices_can, seg_len):
    assert vertices_can.dim() == 4, "The dim of vertices_can must be 4!"
    half_seg_len = seg_len // 2
    center_frame_verts = vertices_can[:, half_seg_len, :, :].unsqueeze(1)
    vertices_can = vertices_can - center_frame_verts
    vertices_can[:, half_seg_len, :, :] = center_frame_verts[:, 0, :, :]
    return vertices_can
    # test1 = torch.range(start=0, end=23).reshape(2, 2, 3, 2)
    # test2 = torch.ones(2, 1, 3, 2)
    # test2[1] = 2
    # result = test1 - test2
    # print(result)


def compute_iou(gt, pred):
    intersection = pred[gt == 1]
    union = torch.clone(pred)
    union[gt == 1] = 1
    if torch.sum(union) == 0:
        return 1
    else:
        return torch.sum(intersection) / torch.sum(union)

def compute_f1_score(gt, pred):
    tp = torch.sum(pred[gt == 1])
    pred_p = torch.sum(pred)
    gt_p = torch.sum(gt)
    if pred_p == 0:
        precision = 1
    else:
        precision = tp / pred_p

    if gt_p == 0:
        recall = 1
    else:
        recall = tp / gt_p

    if precision + recall == 0:
        return 0
    else:
        return (2 * precision * recall) / (precision + recall)


def compute_tpr(gt, pred):
    tp = torch.sum(pred[gt == 1])
    p = torch.sum(gt)
    if p == 0:
        return 1
    return tp / p


def compute_tnr(gt, pred):
    tn = torch.sum((pred == 0)[gt == 0])
    n = torch.sum(gt == 0)
    if n == 0:
        return 1
    return tn / n

def onehot_encode(semantic_tensor, num_obj_classes, device):
    '''
    semantic_tensor: (..., num_obj_classes)

    return onehot_encoder: (..., num_obj_classes, 1)
    '''
    semantic_tensor = torch.zeros(*semantic_tensor.shape, num_obj_classes, dtype=torch.float32, device=device)\
                        .scatter_(-1, semantic_tensor.unsqueeze(-1).type(torch.long), 1.)
    return semantic_tensor

def compute_consistency_metric(pred, verts, eps=0.1, use_xy=True):
    pred = torch.argmax(pred, dim=-1)
    pred = pred.reshape(-1)
    verts = verts.reshape(-1, 3)
    contact_indices = (pred != 0)
    pred = pred[contact_indices]
    verts = verts[contact_indices]
    total_pts = verts.shape[0]
    if use_xy:
        verts = verts[:, :2]
    inconsistent_pts = 0
    for i in range(total_pts):
        cur_pred = pred[i]
        if cur_pred == 0:
            continue
        cur_vert = verts[i].unsqueeze(0)
        cur_dist = torch.cdist(cur_vert.unsqueeze(0), verts.unsqueeze(0)).squeeze()
        neighbor_indices = torch.logical_and(cur_dist < eps, pred != 0)
        if (neighbor_indices.sum() == 1):
            continue
        neighbors = pred[neighbor_indices]
        neighbors_mode = torch.mode(neighbors)[0]
        if (neighbors == cur_pred).sum() < (neighbors == neighbors_mode).sum():
            inconsistent_pts += 1

    return inconsistent_pts / total_pts


if __name__ == "__main__":
    gt = torch.tensor([0, 0, 1, 1, 1])
    pred = torch.tensor([1, 0, 1, 0, 1])
    compute_iou(gt, pred)
