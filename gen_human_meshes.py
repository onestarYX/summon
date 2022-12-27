import argparse
import os

import trimesh

import numpy as np
import open3d as o3d

from tqdm import tqdm

from utils import read_mpcat40, pred_subset_to_mpcat40, create_o3d_mesh_from_vertices_faces


def gen_human_meshes(vertices_path, output_path):
    vertices = np.load(open(vertices_path, "rb"))
    # If your input human vertices are full resolution SMPL-X bodies, use mesh_0.obj
    # faces = trimesh.load(os.path.join("mesh_ds", "mesh_0.obj"), process=False).faces
    faces = trimesh.load(os.path.join("mesh_ds", "mesh_2.obj"), process=False).faces

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print("Saving human meshes to", output_path)
    for frame in tqdm(range(vertices.shape[0])):
        vertices_frame = vertices[frame]
        mesh = create_o3d_mesh_from_vertices_faces(vertices_frame, faces)
        vertex_colors = np.ones_like(vertices_frame)
        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        o3d.io.write_triangle_mesh(os.path.join(output_path, "human_" + str(frame) + ".ply"), mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("sequence_name",
                        type=str)
    parser.add_argument("vertices_path",
                        type=str)
    parser.add_argument("vertices_ds2_path",
                        type=str)
    args = parser.parse_args()
    
    sequence_name = args.sequence_name
    
    vertices = np.load(open(args.vertices_path, "rb"))
    faces = trimesh.load(os.path.join("mesh_ds", "mesh_0.obj"), process=False).faces
    
    vertices_ds2 = np.load(open(args.vertices_ds2_path, "rb"))
    faces_ds2 = trimesh.load(os.path.join("mesh_ds", "mesh_2.obj"), process=False).faces
    
    output_base_path = os.path.join("models", sequence_name, "human", "mesh")
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)
    
    print("Saving full resolution meshes to", output_base_path)
    for frame in tqdm(range(vertices.shape[0])): 
        vertices_frame = vertices[frame]
        mesh = create_o3d_mesh_from_vertices_faces(vertices_frame, faces)
        vertex_colors = np.ones_like(vertices_frame)
        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        o3d.io.write_triangle_mesh(os.path.join(output_base_path, "human_" + str(frame) + ".ply"), mesh)  
    
    output_base_path = os.path.join("models", sequence_name, "human", "mesh_ds2")
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)
    
    print("Saving downsampled meshes to", output_base_path)
    for frame in tqdm(range(vertices_ds2.shape[0])):
        vertices_frame = vertices_ds2[frame]
        mesh = create_o3d_mesh_from_vertices_faces(vertices_frame, faces_ds2)
        vertex_colors = np.ones_like(vertices_frame)
        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        o3d.io.write_triangle_mesh(os.path.join(output_base_path, "human_ds2_" + str(frame) + ".ply"), mesh)
    
