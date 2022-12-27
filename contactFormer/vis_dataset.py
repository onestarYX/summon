import os
import numpy as np
import argparse
import torch
import vis_utils
import trimesh
import open3d as o3d
from tqdm import tqdm
import eulerangles
"""
python vis_dataset.py --save_video --cam_setting_path support_files/ScreenCamera_7.json --seq_name MPH11_00151_01
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--data_dir", type=str, default="../data/new_posa_temp_train",
                        help="path to POSA_temp dataset dir")
    parser.add_argument("--save_video", dest='save_video', action='store_const', const=True, default=False)
    parser.add_argument("--save_video_dir", type=str, default="../new_output/gt_dataset_video/",
                        help="the path to save results from temp posa")
    parser.add_argument("--cam_setting_path", type=str, default="./support_files/ScreenCamera_0.json",
                        help="the path to camera settings in open3d")
    parser.add_argument("--scene_dir", type=str, default="/home/yixing/research/human_3d/POSA/POSA_dir/scenes",
                        help="the path to the scene mesh")
    parser.add_argument("--tpose_mesh_dir", type=str, default="../data/mesh_ds",
                        help="the path to the tpose body mesh (primarily for loading faces)")
    parser.add_argument("--seq_name", type=str, default="N0Sofa_00034_01")
    parser.add_argument("--single_frame", type=int, default=-1)
    parser.add_argument("--show_canonical", dest='show_canonical', action='store_const', const=True, default=False)

    args = parser.parse_args()
    args_dict = vars(args)

    data_dir = args_dict['data_dir']
    scene_dir = args_dict['scene_dir']
    tpose_mesh_dir = args_dict['tpose_mesh_dir']
    save_video = args_dict['save_video']
    save_video_dir = args_dict['save_video_dir']
    single_frame = args_dict['single_frame']
    show_canonical = args_dict['show_canonical']
    cam_path = args_dict['cam_setting_path']

    device = torch.device("cuda")
    no_obj_classes = 8

    contacts_s_dir = os.path.join(data_dir, "semantics")
    vertices_can_dir = os.path.join(data_dir, "vertices_can")
    vertices_dir = os.path.join(data_dir, "vertices")

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()

    seq_name = args_dict['seq_name']
    print("Testing sequence: ", seq_name)
    verts = torch.tensor(np.load(os.path.join(vertices_dir, seq_name + "_verts.npy"))).to(device)
    verts_can = torch.tensor(np.load(os.path.join(vertices_can_dir, seq_name + "_verts_can.npy"))).to(device)
    contacts_s = torch.tensor(np.load(os.path.join(contacts_s_dir, seq_name + "_cfs.npy"))).to(device)

    down_sample_level = 2
    # tpose for getting face_arr
    tpose_mesh_path = os.path.join(tpose_mesh_dir, "mesh_{}.obj".format(down_sample_level))
    faces_arr = trimesh.load(tpose_mesh_path, process=False).faces

    for frame in tqdm(range(verts.shape[0])):
        if single_frame != -1:
            if frame != single_frame:
                continue
        verts_batch = verts[frame].unsqueeze(0)
        verts_can_batch = verts_can[frame]
        R_can = torch.tensor(eulerangles.euler2mat(-np.pi / 2, 0, 0, 'sxyz'), dtype=verts_can_batch.dtype, device=device)
        verts_can_batch = torch.matmul(R_can, verts_can_batch.t()).t()
        verts_can_batch = verts_can_batch.unsqueeze(0)
        contacts_s_batch = contacts_s[frame]
        contacts_s_batch = torch.zeros(*contacts_s_batch.shape, no_obj_classes, dtype=torch.float32).to(device)\
                                        .scatter_(-1, contacts_s_batch.unsqueeze(-1).type(torch.long), 1.).unsqueeze(0)
        # contacts_s_batch = torch.zeros_like(contacts_s_batch)

        # visualization
        vis = []
        if show_canonical:
            vis += vis_utils.show_sample(verts_can_batch, contacts_s_batch, faces_arr, True)
            # vis.append(o3d.geometry.TriangleMesh.create_coordinate_frame())
        else:
            scene_name = seq_name.split("_")[0]
            scene_mesh_path = "{}/{}.ply".format(scene_dir, scene_name)
            scene = o3d.io.read_triangle_mesh(scene_mesh_path)
            vis += [scene]
            vis += vis_utils.show_sample(verts_batch, contacts_s_batch, faces_arr, True)

        if not save_video:
            o3d.visualization.draw_geometries(vis)
        elif save_video and single_frame == -1:
            for geometry in vis:
                visualizer.add_geometry(geometry)
                # visualizer.update_geometry(geometry)

            ctr = visualizer.get_view_control()
            parameters = o3d.io.read_pinhole_camera_parameters(cam_path)
            ctr.convert_from_pinhole_camera_parameters(parameters)
            visualizer.poll_events()
            visualizer.update_renderer()
            output_image_dir = os.path.join(save_video_dir, seq_name + "_semantics")
            # output_image_dir = os.path.join(save_video_dir, seq_name)
            os.makedirs(output_image_dir, exist_ok=True)
            # output_image_dir = os.path.join(output_image_dir, cam_path)
            # os.makedirs(output_image_dir, exist_ok=True)
            visualizer.capture_screen_image(os.path.join(output_image_dir, "frame_{:04d}.png".format(frame)))
            for geometry in vis:
                visualizer.remove_geometry(geometry)

    visualizer.destroy_window()