import os
import pickle

import smplx

import torch

import numpy as np

from tqdm import tqdm

from utils import get_graph_params, ds_us


def pickle_amass_vertices(input_path, output_path, output_name, smplx_models_path, device, num_pca_comps=6):
    parameters = np.load(input_path)

    gender = str(parameters["gender"])
    betas = parameters["betas"][:10]
    
    frames = []
    for i in range(len(parameters["poses"])):
        frame = {}
        frame["root_orient"] = parameters["root_orient"][i]
        frame["trans"] = parameters["trans"][i]
        frame["pose_body"] = parameters["pose_body"][i]
        frames.append(frame)
    
    model_params = dict(model_path=smplx_models_path,
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
                        batch_size=1)
    body_model = smplx.create(gender=gender, **model_params).to(device)
    
    _, _, D_1 = get_graph_params("mesh_ds", 1, device)
    ds1 = ds_us(D_1).to(device)
    _, _, D_2 = get_graph_params("mesh_ds", 2, device)
    ds2 = ds_us(D_2).to(device)
    
    all_vertices = []
    all_vertices_can = []
    all_vertices_ds2 = []
    all_vertices_can_ds2 = []
    torch_params = {}
    torch_params['betas'] = torch.tensor(betas, dtype=torch.float32).to(device).unsqueeze(0)
    for i in tqdm(range(len(frames))):
        frame = frames[i]
        torch_params['global_orient'] = torch.tensor(frame['root_orient'], dtype=torch.float32).to(device).unsqueeze(0)
        torch_params['transl'] = torch.tensor(frame['trans'], dtype=torch.float32).to(device).unsqueeze(0)
        torch_params['body_pose'] = torch.tensor(frame['pose_body'], dtype=torch.float32).to(device).flatten().unsqueeze(0)
        body_model.reset_params(**torch_params)
        body_model_output = body_model(return_verts=True)
        vertices = body_model_output.vertices.squeeze()
        pelvis = body_model_output.joints[:, 0, :].reshape(1, 3)
        vertices_can = vertices - pelvis
        all_vertices.append(vertices.detach().cpu().numpy())
        all_vertices_can.append(vertices_can.detach().cpu().numpy())
        vertices_ds2 = ds2(ds1(vertices.unsqueeze(0).permute(0,2,1))).permute(0,2,1).squeeze()
        vertices_can_ds2 = ds2(ds1(vertices_can.unsqueeze(0).permute(0,2,1))).permute(0,2,1).squeeze()
        all_vertices_ds2.append(vertices_ds2.detach().cpu().numpy())
        all_vertices_can_ds2.append(vertices_can_ds2.detach().cpu().numpy())
    
    all_vertices = np.array(all_vertices)
    all_vertices_can = np.array(all_vertices_can)
    all_vertices_ds2 = np.array(all_vertices_ds2)
    all_vertices_can_ds2 = np.array(all_vertices_can_ds2)
    
    np.save(os.path.join(output_path, output_name + "_verts.npy"), all_vertices)
    np.save(os.path.join(output_path, output_name + "_verts_can.npy"), all_vertices_can)
    np.save(os.path.join(output_path, output_name + "_verts_ds2.npy"), all_vertices_ds2)
    np.save(os.path.join(output_path, output_name + "_verts_can_ds2.npy"), all_vertices_can_ds2)
    

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("input_path",
                        type=str)
    parser.add_argument("output_path",
                        type=str)
    parser.add_argument("output_name",
                        type=str)
    args = parser.parse_args()
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
                 
    if torch.cuda.is_available():
        print("Using cuda")
        device = torch.device("cuda")
    else:
        print("Using cpu")
        device = torch.device("cpu")
    print()    
    
    smplx_models = "smplx_models"
    
    pickle_amass_vertices(args.input_path, args.output_path, args.output_name, smplx_models, device)
    
    
    
