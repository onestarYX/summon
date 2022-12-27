import os
import math
import numpy as np
import argparse
import torch
from contactFormer import ContactFormer
import vis_utils
import trimesh
import open3d as o3d
from random import randrange
from tqdm import tqdm
import general_utils
import data_utils as du
import sklearn.cluster

"""
Running sample:
python test_contactformer.py ../data/proxd_valid/ --load_model ../training/contactformer/model_ckpt/best_model_recon_acc.pt --model_name contactformer --fix_ori --test_on_valid_set --output_dir ../test_output
python test_contactformer.py ../data/proxd_valid/ --load_model ../training/contactformer/model_ckpt/best_model_recon_acc.pt --model_name contactformer --fix_ori --single_seq_name MPH112_00151_01 --save_video --output_dir ../test_output
"""
def list_mean(list):
    acc = 0.
    for item in list:
        acc += item
    return acc / len(list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("data_dir", type=str,
                        help="path to POSA_temp dataset dir")
    parser.add_argument("--load_model", type=str, default="../training/contactformer/model_ckpt/best_model_recon_acc.pt",
                        help="checkpoint path to load")
    parser.add_argument("--posa_path", type=str, default="../training/posa/model_ckpt/epoch_0349.pt")
    parser.add_argument("--visualize", dest="visualize", action='store_const', const=True, default=False)
    parser.add_argument("--scene_dir", type=str, default="../data/scenes",
                        help="the path to the scene mesh")
    parser.add_argument("--tpose_mesh_dir", type=str, default="../mesh_ds",
                        help="the path to the tpose body mesh (primarily for loading faces)")
    parser.add_argument("--save_video", dest='save_video', action='store_const', const=True, default=False)
    parser.add_argument("--output_dir", type=str, default="../test_output",
                        help="the path to save test results")
    parser.add_argument("--cam_setting_path", type=str, default="./support_files/ScreenCamera_0.json",
                        help="the path to camera settings in open3d")
    parser.add_argument("--single_seq_name", type=str, default="BasementSittingBooth_00142_01")
    parser.add_argument("--test_on_train_set", dest='do_train', action='store_const', const=True, default=False)
    parser.add_argument("--test_on_valid_set", dest='do_valid', action='store_const', const=True, default=False)
    parser.add_argument("--model_name", type=str, default="default_model",
                        help="The name of the model we are testing. This is also the suffix for result text file name.")
    parser.add_argument("--fix_ori", dest='fix_ori', action='store_const', const=True, default=False)
    parser.add_argument("--encoder_mode", type=int, default=1,
                        help="Encoder mode (different number represents different versions of encoder)")
    parser.add_argument("--decoder_mode", type=int, default=1,
                        help="Decoder mode (different number represents different versions of decoder)")
    parser.add_argument("--n_layer", type=int, default=3)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--jump_step", type=int, default=8)
    parser.add_argument("--dim_ff", type=int, default=512)
    parser.add_argument("--f_vert", type=int, default=64)
    parser.add_argument("--max_frame", type=int, default=256)

    # Parse arguments and assign directories
    args = parser.parse_args()
    args_dict = vars(args)

    data_dir = args_dict['data_dir']
    scene_dir = args_dict['scene_dir']
    tpose_mesh_dir = args_dict['tpose_mesh_dir']
    ckpt_path = args_dict['load_model']
    save_video = args_dict['save_video']
    output_dir = args_dict['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    cam_path = args_dict['cam_setting_path']
    single_seq_name = args_dict['single_seq_name']
    model_name = args_dict['model_name']
    fix_ori = args_dict['fix_ori']
    encoder_mode = args_dict['encoder_mode']
    decoder_mode = args_dict['decoder_mode']
    do_train = args_dict['do_train']
    do_valid = args_dict['do_valid']
    visualize = args_dict['visualize']
    n_layer = args_dict['n_layer']
    n_head = args_dict['n_head']
    jump_step = args_dict['jump_step']
    max_frame = args_dict['max_frame']
    dim_ff = args_dict['dim_ff']
    f_vert = args_dict['f_vert']
    posa_path = args_dict['posa_path']

    # Argument logic check
    if visualize and save_video:
        save_video = False
    if do_train or do_valid:
        save_video = False
        visualize = False

    device = torch.device("cuda")
    num_obj_classes = 8
    # For fix_ori
    ds_weights = torch.tensor(np.load("./support_files/downsampled_weights.npy"))
    associated_joints = torch.argmax(ds_weights, dim=1)


    seq_name_list = []
    iou_s_list = []
    f1_s_list = []
    recon_semantics_acc_list = []
    inconsistency_list = []
    seq_class_acc = [[] for _ in range(num_obj_classes)]

    vertices_dir = os.path.join(data_dir, "vertices")
    contacts_s_dir = os.path.join(data_dir, "semantics")
    vertices_can_dir = os.path.join(data_dir, "vertices_can")
    if do_valid or do_train:
        vertices_file_list = os.listdir(vertices_dir)
        seq_name_list = [file_name.split('_verts')[0] for file_name in vertices_file_list]
    else:
        seq_name_list = [single_seq_name]

    # Load in model checkpoints and set up data stream
    model = ContactFormer(seg_len=max_frame, encoder_mode=encoder_mode, decoder_mode=decoder_mode,
                          n_layer=n_layer, n_head=n_head, f_vert=f_vert, dim_ff=dim_ff, d_hid=512,
                          posa_path=posa_path).to(device)
    model.eval()
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])


    for seq_name in seq_name_list:
        if save_video or visualize:
            save_seq_dir = os.path.join(output_dir, seq_name)
            os.makedirs(save_seq_dir, exist_ok=True)
            save_seq_model_dir = os.path.join(save_seq_dir, model_name)
            os.makedirs(save_seq_model_dir, exist_ok=True)
            output_image_dir = os.path.join(save_seq_model_dir, cam_path.split("/")[-1])
            os.makedirs(output_image_dir, exist_ok=True)
        print("Test scene: {}".format(seq_name))

        verts = torch.tensor(np.load(os.path.join(vertices_dir, seq_name + "_verts.npy"))).to(device)
        verts_can = torch.tensor(np.load(os.path.join(vertices_can_dir, seq_name + "_verts_can.npy"))).to(device)
        contacts_s = torch.tensor(np.load(os.path.join(contacts_s_dir, seq_name + "_cfs.npy"))).to(device)

        if save_video:
            visualizer = o3d.visualization.Visualizer()
            visualizer.create_window()

        if visualize or save_video:
            scene_name = seq_name.split("_")[0]
            test_scene_mesh_path = "{}/{}.ply".format(scene_dir, scene_name)
            scene = o3d.io.read_triangle_mesh(test_scene_mesh_path)
            down_sample_level = 2
            # tpose for getting face_arr
            tpose_mesh_path = os.path.join(tpose_mesh_dir, "mesh_{}.obj".format(down_sample_level))
            faces_arr = trimesh.load(tpose_mesh_path, process=False).faces

        # Loop over video frames to get predictions
        # Metrics for semantic labels
        iou_s = 0
        f1_s = 0
        recon_semantics_acc = 0
        inconsistency = 0
        class_acc_list = [[] for _ in range(num_obj_classes)]
        class_acc = dict()

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
        for i in tqdm(range(pred.shape[0])):
            if mask[0, i] == 0:
                break
            idx_in_seq = i * jump_step
            cur_pred_semantic = torch.argmax(pred[i], dim=1)
            # Compute metrics for semantic labels
            cur_gt_semantic = contacts_s[idx_in_seq].squeeze()
            # cur_gt_semantic = torch.argmax(cur_gt_semantic, dim=1)
            recon_semantics_acc += torch.mean((cur_pred_semantic == cur_gt_semantic).to(torch.float32))

            cur_gt_semantic = general_utils.onehot_encode(cur_gt_semantic, num_obj_classes, device)
            cur_pred_semantic = general_utils.onehot_encode(cur_pred_semantic, num_obj_classes, device)
            num_obj_classes_appeared = 0
            iou_s_cur = 0
            f1_s_cur = 0
            for class_idx in range(num_obj_classes):
                if torch.sum(cur_gt_semantic[:, class_idx] > 0):
                    num_obj_classes_appeared += 1
                    iou_s_cur += general_utils.compute_iou(cur_gt_semantic[:, class_idx],
                                                           cur_pred_semantic[:, class_idx])
                    f1_s_cur += general_utils.compute_f1_score(cur_gt_semantic[:, class_idx],
                                                               cur_pred_semantic[:, class_idx])
            iou_s += iou_s_cur / num_obj_classes_appeared
            f1_s += f1_s_cur / num_obj_classes_appeared

            # Visualization
            if visualize or save_video:
                vis = []
                vis += [scene]
                vis += vis_utils.show_sample(verts[idx_in_seq, :, :], cur_pred_semantic.unsqueeze(0), faces_arr, True)
            if visualize:
                o3d.visualization.draw_geometries(vis)
            if save_video:
                for geometry in vis:
                    visualizer.add_geometry(geometry)

                ctr = visualizer.get_view_control()
                parameters = o3d.io.read_pinhole_camera_parameters(cam_path)
                ctr.convert_from_pinhole_camera_parameters(parameters)
                visualizer.poll_events()
                visualizer.update_renderer()
                visualizer.capture_screen_image(
                    os.path.join(output_image_dir, "frame_{:04d}.png".format(idx_in_seq)))
                for geometry in vis:
                    visualizer.remove_geometry(geometry)

        pred = pred[:(int)(mask.sum())]

        verts = verts[::jump_step]
        verts = verts[:(int)(mask.sum())]
        inconsistency = general_utils.compute_consistency_metric(pred, verts)
        inconsistency_list.append(inconsistency)

        iou_s /= mask.sum(); iou_s_list.append(iou_s)
        f1_s /= mask.sum(); f1_s_list.append(f1_s)
        recon_semantics_acc /= mask.sum(); recon_semantics_acc_list.append(recon_semantics_acc)

        if save_video:
            visualizer.destroy_window()
            f = open(os.path.join(save_seq_model_dir, "results.txt"), "w")
            f.write("IOU for semantic labels: {:.4f}".format(iou_s) + '\n')
            f.write("F1-score for semantic labels: {:.4f}".format(f1_s) + '\n')
            f.write("Reconstructed Semantics Accuracy: {:.4f}".format(recon_semantics_acc) + '\n')
            f.write("Inconsistency score: {:.4f}".format(inconsistency) + '\n')

    if do_valid:
        f = open(os.path.join(output_dir, "validation_results_{}.txt".format(model_name)), "w")
        for i, seq_name in enumerate(seq_name_list):
            f.write(f"IOU for {seq_name}: {iou_s_list[i]:.4f}\n")
        f.write("\n")
        for i, seq_name in enumerate(seq_name_list):
            f.write(f"F1-score for {seq_name}: {f1_s_list[i]:.4f}\n")
        f.write("\n")
        for i, seq_name in enumerate(seq_name_list):
            f.write(f"Reconstructed Semantics Accuracy for {seq_name}: {recon_semantics_acc_list[i]:.4f}\n")
        f.write("\n")
        for i, seq_name in enumerate(seq_name_list):
            f.write(f"Inconsistency score for {seq_name}: {inconsistency_list[i]:.4f}\n")
        f.write("\n")
        f.write("IOU for semantic labels: {:.4f}".format(list_mean(iou_s_list)) + '\n')
        f.write("F1-score for semantic labels: {:.4f}".format(list_mean(f1_s_list)) + '\n')
        f.write("Reconstructed Semantics Accuracy: {:.4f}".format(list_mean(recon_semantics_acc_list)) + '\n')
        f.write("Inconsistency Score: {:.4f}".format(list_mean(inconsistency_list)) + '\n')

    if do_train:
        f = open(os.path.join(output_dir, "train_results_{}.txt".format(model_name)), "w")
        for i, seq_name in enumerate(seq_name_list):
            f.write(f"IOU for {seq_name}: {iou_s_list[i]:.4f}\n")
        f.write("\n")
        for i, seq_name in enumerate(seq_name_list):
            f.write(f"F1-score for {seq_name}: {f1_s_list[i]:.4f}\n")
        f.write("\n")
        for i, seq_name in enumerate(seq_name_list):
            f.write(f"Reconstructed Semantics Accuracy for {seq_name}: {recon_semantics_acc_list[i]:.4f}\n")
        f.write("\n")
        f.write("IOU for semantic labels: {:.4f}".format(list_mean(iou_s_list)) + '\n')
        f.write("F1-score for semantic labels: {:.4f}".format(list_mean(f1_s_list)) + '\n')
        f.write("Reconstructed Semantics Accuracy: {:.4f}".format(list_mean(recon_semantics_acc_list)) + '\n')