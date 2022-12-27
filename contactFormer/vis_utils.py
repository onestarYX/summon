import os
import os.path as osp
import numpy as np
import open3d as o3d
import torch
import pandas as pd

default_color = [1.0, 0.75, 0.8]

def batch2features(in_batch, use_semantics, **kwargs):
    in_batch = in_batch.squeeze(0)
    if torch.is_tensor(in_batch):
        in_batch = in_batch.detach().cpu().numpy()
    x = in_batch[:, 0]
    x_semantics = None
    if use_semantics:
        x_semantics = in_batch[:, 1:]
    return x, x_semantics

def show_contact_fn(vertices, x, faces_arr, **kwargs):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy().squeeze()

    if torch.is_tensor(vertices):
        vertices = vertices.detach().cpu().numpy().squeeze()

    x = (x > 0.5).astype(np.int)
    vertex_colors = np.ones((vertices.shape[0], 3)) * np.array(default_color)
    vertex_colors[x == 1, :3] = [0.0, 0.0, 1.0]
    body_gt_contact = create_o3d_mesh_from_np(vertices=vertices, faces=faces_arr,
                                              vertex_colors=vertex_colors)
    return [body_gt_contact]

def show_semantics_fn(vertices, x_semantics, faces_arr, **kwargs):
    semantics_color_coding = get_semantics_color_coding()
    if torch.is_tensor(x_semantics):
        x_semantics = x_semantics.detach().cpu().numpy().squeeze()
    if torch.is_tensor(vertices):
        vertices = vertices.detach().cpu().numpy().squeeze()

    x_semantics = np.argmax(x_semantics, axis=1)
    vertex_colors = np.ones((vertices.shape[0], 3)) * np.array(default_color)

    vertex_colors[:, :3] = np.take(semantics_color_coding, list(x_semantics), axis=0) / 255.0
    vertex_colors[x_semantics == 0, :] = default_color

    body = create_o3d_mesh_from_np(vertices=vertices, faces=faces_arr,
                                   vertex_colors=vertex_colors)
    return [body]

def create_o3d_mesh_from_np(vertices, faces, vertex_colors=[]):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    mesh.vertex_normals = o3d.utility.Vector3dVector([])
    mesh.triangle_normals = o3d.utility.Vector3dVector([])
    mesh.compute_vertex_normals()
    return mesh

def get_semantics_color_coding():
    matter_port_label_filename = '../mpcat40_modified.tsv'
    matter_port_label_filename = osp.expandvars(matter_port_label_filename)
    df = pd.read_csv(matter_port_label_filename, sep='\t')
    color_coding_hex = list(df['hex'])  # list of str
    color_coding_rgb = hex2rgb(color_coding_hex)
    return color_coding_rgb

def hex2rgb(hex_color_list):
    rgb_list = []
    for hex_color in hex_color_list:
        h = hex_color.lstrip('#')
        rgb = list(int(h[i:i + 2], 16) for i in (0, 2, 4))
        rgb_list.append(rgb)

    return np.array(rgb_list)


def show_sample(vertices, semantics, faces_arr, use_semantics, make_canonical=True, use_shift=True, **kwargs):
    results = []
    if torch.is_tensor(vertices):
        vertices = vertices.detach().cpu().numpy().squeeze()
    # if not make_canonical:
    #     vertex_colors = np.ones((vertices.shape[0], 3)) * np.array(default_color)
    #     body = create_o3d_mesh_from_np(vertices=vertices, faces=faces_arr,
    #                                    vertex_colors=vertex_colors)
    #     results.append(body)
    #     vertices = vertices + np.array([0, 0.0, 2.0])

    shift = 0

    x_semantics_mesh = show_semantics_fn(vertices + np.array([0.0, 0.0, shift]).reshape(1, 3), semantics,
                                         faces_arr, **kwargs)
    results += x_semantics_mesh

    return results