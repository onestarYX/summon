import os
import argparse
import torch
import numpy as np
import open3d as o3d
import trimesh
from tqdm import tqdm

import data_utils as du
import vis_utils as vu

device = "cuda"
dtype = torch.float32
use_semantics = True
no_obj_classes = 42
batch_size = 1
gender = 'neutral'
default_color = [1.0, 0.75, 0.8]


smplx_model_path = "../data/smplx/models/SMPLX_NEUTRAL.pkl"

posa_cam2world = "../../POSA/POSA_dir/cam2world"
posa_scene = "../../POSA/POSA_dir/scenes"
test_scene_name = "MPH16"
test_scene_mesh_path = "{}/{}.ply".format(posa_scene, test_scene_name)
test_sdf_dir = "../../POSA/POSA_dir/sdf"
test_cam_path = "{}/{}.json".format(posa_cam2world, test_scene_name)
test_pkl_file_path = "../data/PROXD_temp/MPH16_00157_01/results/s001_frame_00228__00.00.07.577/000.pkl"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset for temporal_POSA")
    parser.add_argument("--data_dir", type=str, default="../data/PROXD_temp",
                        help="path to POSA_temp dataset dir")
    parser.add_argument("--scene_dir", type=str, default="../../POSA/POSA_dir/scenes",
                        help="path to PROX scene meshes dir")
    parser.add_argument("--sdf_dir", type=str, default="../../POSA/POSA_dir/sdf",
                        help="path to scene sdf dir")
    parser.add_argument("--cam_dir", type=str, default="../../POSA/POSA_dir/cam2world",
                        help="path to scene camera setting dir")
    parser.add_argument("--output_dir", type=str, default="../data/posa_temp",
                        help="path to save generated dataset")
    parser.add_argument("--smplx_model_dir", type=str, default="../../POSA/POSA_dir/smplx_models",
                        help="path to save generated dataset")
    parser.add_argument("--mesh_ds_us_dir", type=str, default="../../POSA/POSA_dir/mesh_ds",
                        help="path to save generated dataset")
    parser.add_argument("--downsample", type=bool, default=False,
                        help="flag indicating whether to downsample meshes.")

    # Parse arguments and assign directories
    args = parser.parse_args()
    args_dict = vars(args)
    smplx_model_dir = args_dict['smplx_model_dir']
    downsample = args_dict['downsample']
    ds_us_dir = args_dict['mesh_ds_us_dir']
    if downsample:
        tpose_mesh_path = (ds_us_dir + "/mesh_{}.obj").format(2)
    else:
        tpose_mesh_path = (ds_us_dir + "/mesh_{}.obj").format(0)
    pose_seq_dir = args_dict['data_dir']
    sdf_dir = args_dict['sdf_dir']
    scene_dir = args_dict['scene_dir']
    cam_dir = args_dict['cam_dir']

    output_dir = args_dict['output_dir']
    save_verts_dir = os.path.join(output_dir, "vertices")
    save_verts_can_dir = os.path.join(output_dir, "vertices_can")
    save_cf_dir = os.path.join(output_dir, "contacts")
    save_cfs_dir = os.path.join(output_dir, "contacts_semantics")

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    if not os.path.isdir(save_verts_dir):
        os.mkdir(save_verts_dir)

    if not os.path.isdir(save_verts_can_dir):
        os.mkdir(save_verts_can_dir)

    if not os.path.isdir(save_cf_dir):
        os.mkdir(save_cf_dir)

    if not os.path.isdir(save_cfs_dir):
        os.mkdir(save_cfs_dir)

    _, _, D_1 = du.get_graph_params(ds_us_dir, device, layer=1)
    down_sample_fn = du.ds_us(D_1).to(device)

    _, _, D_2 = du.get_graph_params(ds_us_dir, device, layer=2)
    down_sample_fn2 = du.ds_us(D_2).to(device)

    for pose_seq_name in (os.listdir(pose_seq_dir)):
        # pose_seq_name is video name.
        print("Currently computing ", pose_seq_name)
        vertices_list = []
        vertices_can_list = []
        contacts_list = []
        contacts_semantics_list = []

        # Get scene name. (Like 'BasementSittingBooth')
        name_end_idx = pose_seq_name.find('_')
        scene_name = pose_seq_name[:name_end_idx]
        # Get scene data.
        scene_data = du.load_scene_data(device=device, name=scene_name, sdf_dir=sdf_dir,
                                     use_semantics=use_semantics, no_obj_classes=no_obj_classes)
        # camera path for current scene.
        cam_path = os.path.join(cam_dir, scene_name + ".json")

        # directory path for each frame
        pose_seq_path = os.path.join(pose_seq_dir, pose_seq_name, "results")
        pkl_files_list = os.listdir(pose_seq_path)
        pkl_files_list.sort()

        for i, f in enumerate(tqdm(pkl_files_list)):
            pkl_file_path = os.path.join(pose_seq_path, f, "000.pkl")
            vertices_can, vertices = du.pkl_to_canonical(pkl_file_path, device,
                                                      dtype, batch_size, gender,
                                                      model_folder=smplx_model_dir,
                                                      cam_path=cam_path)
            vertices = vertices.unsqueeze(0)
            if downsample:
                vertices = down_sample_fn.forward(vertices)
                vertices = down_sample_fn2.forward(vertices)

            vertices_can = vertices_can.unsqueeze(0)
            if downsample:
                vertices_can = down_sample_fn.forward(vertices_can)
                vertices_can = down_sample_fn2.forward(vertices_can)

            cf = du.read_sdf(vertices, scene_data['sdf'],
                          scene_data['grid_dim'], scene_data['grid_min'], scene_data['grid_max'],
                          mode="bilinear").squeeze()
            temp = torch.zeros_like(cf)
            temp[cf < 0.05] = 1
            cf = temp

            cf_semantics = du.read_sdf(vertices, scene_data['semantics'],
                                    scene_data['grid_dim'], scene_data['grid_min'], scene_data['grid_max'],
                                    mode="nearest").squeeze()
            cf_semantics[cf != 1] = 0

            vertices_list.append(vertices.squeeze().cpu().detach().numpy())
            vertices_can_list.append(vertices_can.squeeze().cpu().detach().numpy())
            contacts_list.append(cf.cpu().detach().numpy())
            contacts_semantics_list.append(cf_semantics.cpu().detach().numpy())

            # Visualization
            # if (torch.sum(cf) > 0):
            #     viz = []
            #     scene = o3d.io.read_triangle_mesh("{}/{}.ply".format(posa_scene, scene_name))
            #     viz += [scene]
            #     faces_arr = trimesh.load(tpose_mesh_path, process=False).faces
            #
            #     cf_batch = cf.view(1, cf.shape[0], 1)
            #     cf_semantics = cf_semantics.unsqueeze(0)
            #     cf_semantics = torch.zeros(cf_semantics.shape[0], cf_semantics.shape[1], no_obj_classes,
            #                         dtype=dtype, device=device).scatter_(-1, cf_semantics.unsqueeze(-1).type(
            #                         torch.long), 1.)
            #
            #     cf_batch = torch.cat((cf_batch, cf_semantics), dim=-1)
            #     viz += vu.show_sample(vertices, cf_batch, faces_arr, True)
            #     o3d.visualization.draw_geometries(viz)

        vertices_list = np.array(vertices_list)
        vertices_can_list = np.array(vertices_can_list)
        contacts_list = np.array(contacts_list)
        contacts_semantics_list = np.array(contacts_semantics_list)

        np.save(os.path.join(save_verts_dir, pose_seq_name + "_verts"), vertices_list)
        np.save(os.path.join(save_verts_can_dir, pose_seq_name + "_verts_can"), vertices_can_list)
        np.save(os.path.join(save_cf_dir, pose_seq_name + "_cf"), contacts_list)
        np.save(os.path.join(save_cfs_dir, pose_seq_name + "_cfs"), contacts_semantics_list)







    # For testing.
    # scene_data = du.load_scene_data(device=device, name=test_scene_name, sdf_dir=sdf_dir,
    #                              use_semantics=use_semantics, no_obj_classes=no_obj_classes)
    #
    # _, test_vertices = du.pkl_to_canonical(test_pkl_file_path, device, dtype, batch_size, gender,
    #                                        model_folder=smplx_model_dir, cam_path=test_cam_path)
    # test_vertices = test_vertices.unsqueeze(0)
    #
    # cf = du.read_sdf(test_vertices, scene_data['sdf'], scene_data['grid_dim'], scene_data['grid_min'],
    #                  scene_data['grid_max'], mode="bilinear").squeeze()
    # temp = torch.zeros_like(cf)
    # temp[cf < 0.05] = 1
    # cf = temp
    # cf_semantics = du.read_sdf(test_vertices, scene_data['semantics'], scene_data['grid_dim'], scene_data['grid_min'],
    #                            scene_data['grid_max'], mode="bilinear").squeeze()
    # cf_semantics[cf != 1] = 0
    # print(cf.shape)
    # print(cf_semantics.shape)
    # print(cf_semantics.max())
    # cf_semantics = cf_semantics.unsqueeze(0)
    #
    # viz = []
    # scene = o3d.io.read_triangle_mesh(test_scene_mesh_path)
    # viz += [scene]
    # faces_arr = trimesh.load(tpose_mesh_path, process=False).faces
    # cf_batch = cf.view(1, cf.shape[0], 1)
    # cf_semantics = torch.zeros(cf_semantics.shape[0], cf_semantics.shape[1], no_obj_classes, dtype=dtype,
    #                            device=device).scatter_(-1, cf_semantics.unsqueeze(-1).type(torch.long), 1.)
    # cf_batch = torch.cat((cf_batch, cf_semantics), dim=-1)
    # viz += vu.show_sample(test_vertices, cf_batch, faces_arr, True)
    # o3d.visualization.draw_geometries(viz)
