import os
import numpy as np
import argparse
import torch
from contactFormer import ContactFormer
from tqdm import tqdm
import data_utils as du

# Example usage
# python predict_contact.py ../data/amass --load_model ../training/contactformer/model_ckpt/best_model_recon_acc.pt --output_dir ../results/amass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("data_dir", type=str,
                        help="path to POSA_temp dataset dir")
    parser.add_argument("--load_model", type=str, default="../training/model_ckpt/epoch_0045.pt",
                        help="checkpoint path to load")
    parser.add_argument("--encoder_mode", type=int, default=1,
                        help="different number represents different variants of encoder")
    parser.add_argument("--decoder_mode", type=int, default=1,
                        help="different number represents different variants of decoder")
    parser.add_argument("--n_layer", type=int, default=3, help="Number of layers in transformer")
    parser.add_argument("--n_head", type=int, default=4, help="Number of heads in transformer")
    parser.add_argument("--jump_step", type=int, default=8, help="Frame skip size for each input motion sequence")
    parser.add_argument("--dim_ff", type=int, default=512,
                        help="Dimension of hidden layers in positionwise MLP in the transformer")
    parser.add_argument("--f_vert", type=int, default=64, help="Dimension of the embeddings for body vertices")
    parser.add_argument("--max_frame", type=int, default=256,
                        help="The maximum length of motion sequence (after frame skipping) which model accepts.")
    parser.add_argument("--posa_path", type=str, default="../training/posa/model_ckpt/epoch_0349.pt",
                        help="The POSA model checkpoint that ContactFormer can pre-load")
    parser.add_argument("--output_dir", type=str, default="../results/output")
    parser.add_argument("--save_probability", dest='save_probability', action='store_const', const=True, default=False,
                        help="Save the probability of each contact labels, instead of the most possible contact label")

    # Parse arguments and assign directories
    args = parser.parse_args()
    args_dict = vars(args)

    data_dir = args_dict['data_dir']
    ckpt_path = args_dict['load_model']
    encoder_mode = args_dict['encoder_mode']
    decoder_mode = args_dict['decoder_mode']
    n_layer = args_dict['n_layer']
    n_head = args_dict['n_head']
    jump_step = args_dict['jump_step']
    max_frame = args_dict['max_frame']
    dim_ff = args_dict['dim_ff']
    f_vert = args_dict['f_vert']
    posa_path = args_dict['posa_path']
    output_dir = args_dict['output_dir']
    save_probability = args_dict['save_probability']

    device = torch.device("cuda")
    num_obj_classes = 8
    # For fix_ori
    fix_ori = True
    ds_weights = torch.tensor(np.load("./support_files/downsampled_weights.npy"))
    associated_joints = torch.argmax(ds_weights, dim=1)
    os.makedirs(output_dir, exist_ok=True)

    seq_name_list = []
    vertices_file_list = os.listdir(data_dir)
    seq_name_list = [file_name.split('_verts_can')[0] for file_name in vertices_file_list]
    list_set = set(seq_name_list)
    seq_name_list = list(list_set)

    # Load in model checkpoints and set up data stream
    model = ContactFormer(seg_len=max_frame, encoder_mode=encoder_mode, decoder_mode=decoder_mode,
                          n_layer=n_layer, n_head=n_head, f_vert=f_vert, dim_ff=dim_ff, d_hid=512,
                          posa_path=posa_path).to(device)
    model.eval()
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    for seq_name in seq_name_list:
        print("Test scene: {}".format(seq_name))

        verts_can = torch.tensor(np.load(os.path.join(data_dir, seq_name + "_verts_can.npy"))).to(device).to(torch.float32)

        # Loop over video frames to get predictions
        verts_can_batch = verts_can[::jump_step]
        if fix_ori:
            verts_can_batch = du.normalize_orientation(verts_can_batch, associated_joints, device)
        if verts_can_batch.shape[0] > max_frame:
            verts_can_batch = verts_can_batch[:max_frame]

        mask = torch.zeros(1, max_frame, device=device)
        mask[0, :verts_can_batch.shape[0]] = 1
        verts_can_padding = torch.zeros(max_frame - verts_can_batch.shape[0], *verts_can_batch.shape[1:], device=device)
        verts_can_batch = torch.cat((verts_can_batch, verts_can_padding), dim=0)

        z = torch.tensor(np.random.normal(0, 1, (max_frame, 256)).astype(np.float32)).to(device)

        with torch.no_grad():
            posa_out = model.posa.decoder(z, verts_can_batch)
            if decoder_mode == 0:
                pr_cf = posa_out.unsqueeze(0)
            else:
                pr_cf = model.decoder(posa_out, mask)

        pred = pr_cf.squeeze()
        pred = pred[:(int)(mask.sum())]

        cur_output_path = os.path.join(output_dir, seq_name + ".npy")
        if save_probability:
            softmax = torch.nn.Softmax(dim=2)
            pred_npy = softmax(pred).detach().cpu().numpy()
        else:
            pred_npy = torch.argmax(pred, dim=-1).unsqueeze(-1).detach().cpu().numpy()
        np.save(cur_output_path, pred_npy)