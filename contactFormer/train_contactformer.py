import os
import os.path as osp
import math
import argparse
import torch
import torch.nn.functional as F
import time
from torch.utils.data import DataLoader
from contactFormer import ContactFormer
from posa_utils import count_parameters
from dataset import ProxDataset_ds
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from general_utils import compute_recon_loss, compute_delta
import numpy as np
"""
Running sample:
python train_contactformer.py --train_data_dir ../data/proxd_train --valid_data_dir ../data/proxd_valid --fix_ori --epochs 1000 --jump_step 8
"""

def train():
    model.train()
    torch.autograd.set_detect_anomaly(True)
    total_recon_loss_semantics = 0
    total_semantics_recon_acc = 0
    total_KLD_loss = 0
    total_train_loss = 0

    n_steps = 0
    for verts_can, contacts_s, mask in tqdm(train_data_loader):
        # verts_can: (bs, seg_len, Nverts, 3), contacts_s: (bs, seg_len, Nverts, 8)
        verts_can = verts_can.to(device)
        gt_cf = contacts_s.to(device)  # ground truth contact features
        mask = mask.to(device)

        optimizer.zero_grad()

        # pr_cf: (bs, seg_len, 655, 8), mu: (bs, 256), logvar: (bs, 256)
        pr_cf, mu, logvar = model(gt_cf, verts_can, mask)
        recon_loss_semantics, semantics_recon_acc = compute_recon_loss(gt_cf, pr_cf, mask=mask, **args_dict)

        expanded_mask = mask.unsqueeze(-1).expand(-1, -1, mu.shape[-1])
        mu *= expanded_mask
        logvar *= expanded_mask
        KLD = kl_w * (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / expanded_mask.sum()

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
        for verts_can, contacts_s, mask in tqdm(valid_data_loader):
            # verts_can: (bs, seg_len, Nverts, 3), contacts: (bs, seg_len, Nverts, 1), contacts_s: (bs, seg_len, Nverts, 42)
            verts_can = verts_can.to(device).squeeze()
            gt_cf = contacts_s.to(device)
            mask = mask.to(device)

            # pr_cf: (bs, seg_len, 655, 43), mu: (bs, 256), logvar: (bs, 256)
            z = torch.tensor(np.random.normal(0, 1, (max_frame, 256)).astype(np.float32)).to(device)
            posa_out = model.posa.decoder(z, verts_can)
            if decoder_mode == 0:
                pr_cf = posa_out.unsqueeze(0)
            else:
                pr_cf = model.decoder(posa_out, mask)
            recon_loss_semantics, semantics_recon_acc = compute_recon_loss(gt_cf, pr_cf, mask=mask, **args_dict)

            total_recon_loss_semantics += recon_loss_semantics.item()
            total_semantics_recon_acc += semantics_recon_acc.item()
            n_steps += 1

        total_recon_loss_semantics /= n_steps
        total_semantics_recon_acc /= n_steps

        writer.add_scalar('recon_loss_semantics/validate', total_recon_loss_semantics, e)
        writer.add_scalar('total_semantics_recon_acc/validate', total_semantics_recon_acc, e)

        print(
            '====>Recon_loss_semantics = {:.4f} , Semantics_recon_acc = {:.4f}'.format(
                total_recon_loss_semantics, total_semantics_recon_acc))
        return total_recon_loss_semantics, total_semantics_recon_acc


if __name__ == '__main__':
    # torch.manual_seed(0)
    print(torch.version.cuda)
    # torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--train_data_dir", type=str, default="../data/proxd_train",
                        help="path to POSA_temp dataset dir")
    parser.add_argument("--valid_data_dir", type=str, default="../data/proxd_valid",
                        help="path to POSA_temp dataset dir")
    parser.add_argument("--load_ckpt", type=str, default=None,
                        help="load a checkpoint as the continue point for training")
    parser.add_argument("--posa_path", type=str, default="../training/posa/model_ckpt/best_model_recon_acc.pt")
    parser.add_argument("--out_dir", type=str, default="../training/", help="Folder that stores checkpoints and training logs")
    parser.add_argument("--experiment", type=str, default="default_experiment",
                        help="Experiment name. Checkpoints and training logs will be saved in out_dir/experiment folder.")
    parser.add_argument("--save_interval", type=int, default=50, help="Epoch interval for saving model checkpoints.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument('--fix_ori', dest='fix_ori', action='store_const', const=True, default=False,
                        help="fix orientation of each segment with the rotation calculated from the first frame")
    parser.add_argument("--encoder_mode", type=int, default=1,
                        help="Encoder mode (different number represents different variants of encoder)")
    parser.add_argument("--decoder_mode", type=int, default=1,
                        help="Decoder mode (different number represents different variants of decoder)")
    parser.add_argument("--n_layer", type=int, default=3, help="number of layers in transformer")
    parser.add_argument("--n_head", type=int, default=4, help="number of heads in transformer")
    parser.add_argument("--dim_ff", type=int, default=512, help="dimension of hidden layers in positionwise MLP in the transformer")
    parser.add_argument("--f_vert", type=int, default=64, help="dimension of the embeddings for body vertices")
    parser.add_argument("--num_workers", type=int, default=0, help="number of workers for dataloader")
    parser.add_argument("--jump_step", type=int, default=8, help="frame skip size for each input motion sequence")
    parser.add_argument("--max_frame", type=int, default=256, help="The maximum length of motion sequence (after frame skipping) which model accepts.")

    args = parser.parse_args()
    args_dict = vars(args)

    # Parse arguments
    train_data_dir = args_dict['train_data_dir']
    valid_data_dir = args_dict['valid_data_dir']
    load_ckpt = args_dict['load_ckpt']
    save_interval = args_dict['save_interval']
    out_dir = args_dict['out_dir']
    experiment = args_dict['experiment']
    lr = args_dict['lr']
    epochs = args_dict['epochs']
    fix_ori = args_dict['fix_ori']
    encoder_mode = args_dict['encoder_mode']
    decoder_mode = args_dict['decoder_mode']
    n_layer = args_dict['n_layer']
    n_head = args_dict['n_head']
    num_workers = args_dict['num_workers']
    jump_step = args_dict['jump_step']
    max_frame = args_dict['max_frame']
    dim_ff = args_dict['dim_ff']
    f_vert = args_dict['f_vert']
    posa_path = args_dict['posa_path']

    save_ckpt_dir = os.path.join(out_dir, experiment, "model_ckpt")
    log_dir = os.path.join(out_dir, experiment, "tb_log")
    os.makedirs(save_ckpt_dir, exist_ok=True)
    device = torch.device("cuda")
    dtype = torch.float32
    kl_w = 0.5

    train_dataset = ProxDataset_ds(train_data_dir, max_frame=max_frame, fix_orientation=fix_ori,
                                   step_multiplier=1, jump_step=jump_step)
    train_data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=num_workers)
    valid_dataset = ProxDataset_ds(valid_data_dir, max_frame=max_frame, fix_orientation=fix_ori,
                                   step_multiplier=1, jump_step=jump_step)
    valid_data_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True, num_workers=num_workers)

    model = ContactFormer(seg_len=max_frame, encoder_mode=encoder_mode, decoder_mode=decoder_mode,
                          n_layer=n_layer, n_head=n_head, f_vert=f_vert, dim_ff=dim_ff, d_hid=512,
                          posa_path=posa_path).to(device)
    print(
        f"Training using model----encoder_mode: {encoder_mode}, decoder_mode: {decoder_mode}, max_frame: {max_frame}, "
        f"using_data: {train_data_dir}, epochs: {epochs}, "
        f"n_layer: {n_layer}, n_head: {n_head}, f_vert: {f_vert}, dim_ff: {dim_ff}, jump_step: {jump_step}")
    print("Total trainable parameters: {}".format(count_parameters(model)))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, threshold=0.0001, patience=10, verbose=True)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1000, gamma=0.1, verbose=True)
    # milestones = [200, 400, 600, 800]
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.5, verbose=True)

    best_valid_loss = float('inf')
    best_recon_acc = -float('inf')

    starting_epoch = 0
    if load_ckpt is not None:
        checkpoint = torch.load(load_ckpt)
        model.load_state_dict(checkpoint['model_state_dict'])
        starting_epoch = checkpoint['epoch'] + 1
        print('loading stats of epoch {}'.format(starting_epoch))


    writer = SummaryWriter(log_dir)

    for e in range(starting_epoch, epochs):
        print('Training epoch {}'.format(e))
        start = time.time()
        total_train_loss = train()
        training_time = time.time() - start
        print('training_time = {:.4f}'.format(training_time))

        start = time.time()
        total_valid_loss, total_semantics_recon_acc = validate()
        validation_time = time.time() - start
        print('validation_time = {:.4f}'.format(validation_time))

        scheduler.step(total_train_loss)

        if e % save_interval == save_interval - 1:
            data = {
                'epoch': e,
                'model_state_dict': model.state_dict(),
                # 'optimizer_state_dict': optimizer.state_dict(),
                'total_train_loss': total_train_loss,
                'total_valid_loss': total_valid_loss,
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
