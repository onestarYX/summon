import os
import os.path as osp
import math
import argparse
import torch
import torch.nn.functional as F
import time
from torch.utils.data import DataLoader
from dataset import ProxSegDataset, ProxDataset_ds
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from posa_utils import count_parameters
from posa_models import POSA
from general_utils import compute_recon_loss_posa, compute_recon_loss
import numpy as np

"""
Running sample:
python ./train_posa.py ../data/posa_temp --log_dir ../training/vanilla_posa/tb_log --save_ckpt_dir ../training/vanilla_posa/model_ckpt --epochs 400 --save_interval 40 --steps_per_epoch 200
"""
def train():
    model.train()
    total_recon_loss_semantics = 0
    total_semantics_recon_acc = 0
    total_KLD_loss = 0
    total_train_loss = 0

    n_steps = 0
    if use_dataset == "default":
        for verts_can, contacts_s in tqdm(train_data_loader):
            bs, seg_len, n_verts, _ = verts_can.shape

            # verts_can: (bs, seg_len, Nverts, 3), contacts_s: (bs, seg_len, Nverts, 8)
            verts_can = verts_can.to(device)
            contacts_s = contacts_s.to(device)
            # Remove seg_len dimension since we don't need it for training POSA
            verts_can = verts_can.reshape(bs * seg_len, n_verts, -1)
            gt_cf = contacts_s.reshape(bs * seg_len, n_verts, -1)   # ground-truth contact features

            optimizer.zero_grad()

            # pr_cf: (bs, 655, 8), mu: (bs, 256), logvar: (bs, 256)
            pr_cf, mu, logvar = model(gt_cf, verts_can)
            recon_loss_semantics, semantics_recon_acc = compute_recon_loss_posa(gt_cf, pr_cf, **args_dict)
            KLD = kl_w * (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / (bs * seg_len)
            loss = KLD + recon_loss_semantics

            loss.backward()
            optimizer.step()

            total_recon_loss_semantics += recon_loss_semantics.item()
            total_semantics_recon_acc += semantics_recon_acc.item()
            total_train_loss += loss.item()
            total_KLD_loss += KLD.item()
            n_steps += 1
    elif use_dataset == "bi":
        for verts_can, contacts_s, mask in tqdm(train_data_loader):
            bs, _, n_verts, _ = verts_can.shape
            # verts_can: (bs, seg_len, Nverts, 3), contacts_s: (bs, seg_len, Nverts, 8)
            verts_can = verts_can.to(device)
            contacts_s = contacts_s.to(device)  # ground truth contact features
            mask = mask.to(device)

            verts_can = verts_can.reshape(bs * max_frame, n_verts, -1)
            gt_cf = contacts_s.reshape(bs * max_frame, n_verts, -1)

            optimizer.zero_grad()

            # pr_cf: (bs, seg_len, 655, 8), mu: (bs, 256), logvar: (bs, 256)
            pr_cf, mu, logvar = model(gt_cf, verts_can)
            recon_loss_semantics, semantics_recon_acc = compute_recon_loss(gt_cf.unsqueeze(0), pr_cf.unsqueeze(0), mask=mask, **args_dict)

            real_seg_len = (int)(mask.sum())
            mu = mu[:real_seg_len]
            logvar = logvar[:real_seg_len]
            KLD = kl_w * (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / mask.sum()

            loss = KLD + recon_loss_semantics

            loss.backward()
            optimizer.step()

            total_recon_loss_semantics += recon_loss_semantics.item()
            total_semantics_recon_acc += semantics_recon_acc.item()
            total_train_loss += loss.item()
            total_KLD_loss += KLD.item()
            n_steps += 1

    total_recon_loss_semantics /= n_steps
    total_train_loss /= n_steps
    total_semantics_recon_acc /= n_steps
    total_KLD_loss /= n_steps

    writer.add_scalar('recon_loss_semantics/train', total_recon_loss_semantics, e)
    writer.add_scalar('total_semantics_recon_acc/train', total_semantics_recon_acc, e)
    writer.add_scalar('total/train_total_loss', total_train_loss, e)
    writer.add_scalar('KLD/train_KLD', total_KLD_loss, e)

    print(
        '====> Total_train_loss: {:.4f}, KLD = {:.4f}, Recon_loss_semantics = {:.4f} , Semantics_recon_acc = {:.4f}'.format(
            total_train_loss, total_KLD_loss, total_recon_loss_semantics, total_semantics_recon_acc))
    return total_train_loss


def validate():
    model.eval()
    with torch.no_grad():
        total_recon_loss_semantics = 0
        total_semantics_recon_acc = 0

        n_steps = 0
        if use_dataset == "default":
            for verts_can, contacts_s in tqdm(valid_data_loader):
                bs, seg_len, n_verts, _ = verts_can.shape

                # verts_can: (bs, seg_len, Nverts, 3), contacts: (bs, seg_len, Nverts, 1), contacts_s: (bs, seg_len, Nverts, 42)
                verts_can = verts_can.to(device)
                contacts_s = contacts_s.to(device)
                # Remove seg_len dimension since we don't need it for training POSA
                verts_can = verts_can.reshape(bs * seg_len, n_verts, -1)
                gt_cf = contacts_s.reshape(bs * seg_len, n_verts, -1)

                # pr_cf: (bs, 655, 43), mu: (bs, 256), logvar: (bs, 256)
                z = torch.tensor(np.random.normal(0, 1, (bs * seg_len, 256)).astype(np.float32)).to(device)
                pr_cf = model.decoder(z, verts_can)
                recon_loss_semantics, semantics_recon_acc = compute_recon_loss_posa(gt_cf, pr_cf, **args_dict)

                total_recon_loss_semantics += recon_loss_semantics.item()
                total_semantics_recon_acc += semantics_recon_acc.item()
                n_steps += 1
        elif use_dataset == "bi":
            for verts_can, contacts_s, mask in tqdm(valid_data_loader):
                bs, _, n_verts, _ = verts_can.shape
                # verts_can: (bs, seg_len, Nverts, 3), contacts: (bs, seg_len, Nverts, 1), contacts_s: (bs, seg_len, Nverts, 42)
                verts_can = verts_can.to(device).squeeze()
                contacts_s = contacts_s.to(device)
                mask = mask.to(device)

                verts_can = verts_can.reshape(bs * max_frame, n_verts, -1)
                gt_cf = contacts_s.reshape(bs * max_frame, n_verts, -1)

                # pr_cf: (bs, seg_len, 655, 43), mu: (bs, 256), logvar: (bs, 256)
                z = torch.tensor(np.random.normal(0, 1, (max_frame, 256)).astype(np.float32)).to(device)
                posa_out = model.decoder(z, verts_can)
                pr_cf = posa_out
                recon_loss_semantics, semantics_recon_acc = compute_recon_loss(gt_cf.unsqueeze(0), pr_cf.unsqueeze(0), mask=mask, **args_dict)

                total_recon_loss_semantics += recon_loss_semantics.item()
                total_semantics_recon_acc += semantics_recon_acc.item()
                n_steps += 1

        total_recon_loss_semantics /= n_steps
        total_semantics_recon_acc /= n_steps

        writer.add_scalar('recon_loss_semantics/validate', total_recon_loss_semantics, e)
        writer.add_scalar('total_semantics_recon_acc/validate', total_semantics_recon_acc, e)

        print(
            '====> Recon_loss_semantics = {:.4f} , Semantics_recon_acc = {:.4f}'.format(
                total_recon_loss_semantics, total_semantics_recon_acc))
        return total_recon_loss_semantics, total_semantics_recon_acc


if __name__ == '__main__':
    # torch.manual_seed(0)
    print(torch.version.cuda)
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--train_data_dir", type=str, default="../data/new_posa_temp_train",
                        help="path to POSA_temp dataset dir")
    parser.add_argument("--valid_data_dir", type=str, default="../data/new_posa_temp_valid",
                        help="path to POSA_temp dataset dir")
    parser.add_argument("--load_ckpt_path", type=str, default=None,
                        help="checkpoint path to load")
    parser.add_argument("--out_dir", type=str, default="../new_training/")
    parser.add_argument("--experiment", type=str, default="test_posa")
    parser.add_argument("--save_ckpt", type=bool, default=True)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seg_len", type=int, default=1)
    parser.add_argument("--mesh_ds_dir", type=str, default="../data/mesh_ds",
                        help="the path to the tpose body mesh (primarily for loading faces)")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--step_multiplier", type=int, default=1)
    parser.add_argument("--use_dataset", type=str, default="default")
    parser.add_argument('--fix_ori', dest='fix_ori', action='store_const', const=True, default=False,
                        help="fix orientation of each segment with the rotation calculated from the first frame")
    parser.add_argument("--jump_step", type=int, default=1)
    parser.add_argument("--max_frame", type=int, default=128)
    parser.add_argument("--f_vert", type=int, default=64)

    args = parser.parse_args()
    args_dict = vars(args)

    # Parse arguments
    train_data_dir = args_dict['train_data_dir']
    valid_data_dir = args_dict['valid_data_dir']
    load_ckpt_path = args_dict['load_ckpt_path']
    out_dir = args_dict['out_dir']
    experiment = args_dict['experiment']
    save_ckpt = args_dict['save_ckpt']
    save_interval = args_dict['save_interval']
    lr = args_dict['lr']
    bs = args_dict['batch_size']
    epochs = args_dict['epochs']
    seg_len = args_dict['seg_len']
    mesh_ds_dir = args_dict['mesh_ds_dir']
    num_workers = args_dict['num_workers']
    step_multiplier = args_dict['step_multiplier']
    use_dataset = args_dict['use_dataset']
    fix_ori = args_dict['fix_ori']
    max_frame = args_dict['max_frame']
    jump_step = args_dict['jump_step']
    f_vert = args_dict['f_vert']

    save_ckpt_dir = os.path.join(out_dir, experiment, "model_ckpt")
    log_dir = os.path.join(out_dir, experiment, "tb_log")
    os.makedirs(save_ckpt_dir, exist_ok=True)
    device = torch.device("cuda")
    dtype = torch.float32
    kl_w = 0.05

    model = POSA(ds_us_dir=mesh_ds_dir, use_semantics=True, channels=f_vert).to(device)
    print("Total trainable parameters: {}".format(count_parameters(model)))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, threshold=0.0001, patience=10, verbose=True)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.8, verbose=True)
    # milestones = [1000]
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, verbose=True)

    best_valid_loss = float('inf')
    best_recon_acc = -float('inf')

    starting_epoch = 0
    if load_ckpt_path is not None:
        checkpoint = torch.load(load_ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        starting_epoch = checkpoint['epoch'] + 1
        print('loading stats of epoch {}'.format(starting_epoch))

    if use_dataset == "default":
        train_dataset = ProxSegDataset(train_data_dir, train_seg_len=seg_len, step_multiplier=step_multiplier)
        valid_dataset = ProxSegDataset(valid_data_dir, train_seg_len=seg_len, step_multiplier=step_multiplier)
    elif use_dataset == "bi":
        train_dataset = ProxDataset_ds(train_data_dir, max_frame=max_frame, fix_orientation=fix_ori,
                                       step_multiplier=step_multiplier, jump_step=jump_step)
        valid_dataset = ProxDataset_ds(valid_data_dir, max_frame=max_frame, fix_orientation=fix_ori,
                                       step_multiplier=step_multiplier, jump_step=jump_step)
    train_data_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
    valid_data_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
    writer = SummaryWriter(log_dir)

    for e in range(starting_epoch, epochs):
        print('Training epoch {}'.format(e))
        start = time.time()
        total_train_loss = train()
        training_time = time.time() - start
        print('training_time = {:.4f}'.format(training_time))

        start = time.time()
        total_valid_loss, total_semantics_recon_acc = validate()
        training_time = time.time() - start
        print('validation_time = {:.4f}'.format(training_time))

        scheduler.step(total_train_loss)

        if save_ckpt and e % save_interval == save_interval - 1:
            data = {
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'total_train_loss': total_train_loss,
                'total_valid_loss': total_valid_loss
            }
            torch.save(data, osp.join(save_ckpt_dir, 'epoch_{:04d}.pt'.format(e)))

        if total_valid_loss < best_valid_loss:
            print("Updated best model due to new lowest valid_loss. Current epoch: {}".format(e))
            best_valid_loss = total_valid_loss
            data = {
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'total_train_loss': total_train_loss,
                'total_valid_loss': total_valid_loss,
            }
            torch.save(data, osp.join(save_ckpt_dir, 'best_model_valid_loss.pt'))

        if total_semantics_recon_acc > best_recon_acc:
            print("Updated best model due to new highest semantics_recon_acc. Current epoch: {}".format(e))
            best_recon_acc = total_semantics_recon_acc
            data = {
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'total_train_loss': total_train_loss,
                'total_valid_loss': total_valid_loss,
                'total_semantics_recon_acc': total_semantics_recon_acc
            }
            torch.save(data, osp.join(save_ckpt_dir, 'best_model_recon_acc.pt'))