import json
import os
import time

import torch
import torch.nn as nn

import trimesh

import numpy as np
import open3d as o3d

import scipy.sparse


class ds_us(nn.Module):
    """docstring for ds_us."""

    def __init__(self, M):
        super(ds_us, self).__init__()
        self.M = M

    def forward(self, x):
        """Upsample/downsample mesh. X: B*C*N"""
        out = []
        x = x.transpose(1, 2)
        for i in range(x.shape[0]):
            y = x[i]
            y = spmm(self.M, y)
            out.append(y)
        x = torch.stack(out, dim=0)
        return x.transpose(2, 1)


def adjmat_sparse(adjmat, nsize=1):
    """Create row-normalized sparse graph adjacency matrix."""
    adjmat = scipy.sparse.csr_matrix(adjmat)
    if nsize > 1:
        orig_adjmat = adjmat.copy()
        for _ in range(1, nsize):
            adjmat = adjmat * orig_adjmat
    adjmat.data = np.ones_like(adjmat.data)
    for i in range(adjmat.shape[0]):
        adjmat[i, i] = 1
    num_neighbors = np.array(1 / adjmat.sum(axis=-1))
    adjmat = adjmat.multiply(num_neighbors)
    adjmat = scipy.sparse.coo_matrix(adjmat)
    row = adjmat.row
    col = adjmat.col
    data = adjmat.data
    i = torch.LongTensor(np.array([row, col]))
    v = torch.from_numpy(data).float()
    adjmat = torch.sparse.FloatTensor(i, v, adjmat.shape)
    return adjmat


def scipy_to_pytorch(x):
    """Convert scipy sparse matrices to pytorch sparse matrix."""
    x = scipy.sparse.coo_matrix(x)
    i = torch.LongTensor(np.array([x.row, x.col]))
    v = torch.FloatTensor(x.data)
    return torch.sparse.FloatTensor(i, v, x.shape)


def get_graph_params(ds_us_dir, layer, device, **kwargs):
    """Load and process graph adjacency matrix and upsampling/downsampling matrices."""
    A = scipy.sparse.load_npz(os.path.join(ds_us_dir, 'A_{}.npz'.format(layer)))
    D = scipy.sparse.load_npz(os.path.join(ds_us_dir, 'D_{}.npz'.format(layer)))
    U = scipy.sparse.load_npz(os.path.join(ds_us_dir, 'U_{}.npz'.format(layer)))

    D = scipy_to_pytorch(D).to(device)
    U = scipy_to_pytorch(U).to(device)
    A = adjmat_sparse(A).to(device)
    return A, U, D

    
class SparseMM(torch.autograd.Function):
    """Redefine sparse @ dense matrix multiplication to enable backpropagation.
    The builtin matrix multiplication operation does not support backpropagation in some cases.
    """

    @staticmethod
    def forward(ctx, sparse, dense):
        ctx.req_grad = dense.requires_grad
        ctx.save_for_backward(sparse)
        return torch.matmul(sparse, dense)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        sparse, = ctx.saved_tensors
        if ctx.req_grad:
            grad_input = torch.matmul(sparse.t(), grad_output)
        return None, grad_input


def spmm(sparse, dense):
    return SparseMM.apply(sparse, dense)


pred_subset_to_mpcat40 = np.array([
    0,  # void
    1,  # wall
    2,  # floor
    3,  # chair
    10, # sofa
    5,  # table
    11, # bed
    19, # stool
])


"""
Reads contact label names and color scheme of mpcat40 in [0,1] rgb 
Adapted from https://github.com/mohamedhassanmus/POSA/blob/main/src/viz_utils.py

Args:
    path: path to mpcat40.tsv; default "/mpcat40.tsv"
    
Returns:
    index lookup list of contact label names; index nmupy array of rgb colors

"""
def read_mpcat40(path="mpcat40.tsv"):
    import pandas as pd
    mpcat40 = pd.read_csv(path, sep='\t')
    label_names = list(mpcat40['mpcat40'])
    color_coding_hex = list(mpcat40['hex'])
    color_coding_rgb = []
    for hex_color in color_coding_hex:
        h = hex_color.lstrip('#')
        rgb = list(int(h[i:i + 2], 16) for i in (0, 2, 4))
        color_coding_rgb.append(rgb)
    color_coding_rgb = np.array(color_coding_rgb) / 255.0
    return label_names, color_coding_rgb


"""
Creates Open3D TriangleMesh object from vertices and faces numpy arrays

Args:
    vertices:   numpy array of vertices
    faces:      numpy array of faces

Returns:
    mesh:       Open3D TriangleMesh object
"""
def create_o3d_mesh_from_vertices_faces(vertices, faces):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_colors = o3d.utility.Vector3dVector([])
    mesh.vertex_normals = o3d.utility.Vector3dVector([])
    mesh.triangle_normals = o3d.utility.Vector3dVector([])
    return mesh


"""
Creates Open3D PointCloud object from points and optional color

Args:
    points:     numpy array of 3D coordinates
    colors:     numpy array of RGB colors; default: all white

Returns:
    pcd:        Open3D PointCloud object
"""
def create_o3d_pcd_from_points(points, colors=None):
    if colors is None:
        colors = np.ones_like(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


"""
Creates Open3D TriangleMesh object from vertices and faces numpy arrays

Args:
    vertices:   numpy array of vertices
    faces:      numpy array of faces

Returns:
    mesh:       Open3D TriangleMesh object
"""
def create_o3d_mesh_from_vertices_faces(vertices, faces):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_colors = o3d.utility.Vector3dVector([])
    mesh.vertex_normals = o3d.utility.Vector3dVector([])
    mesh.triangle_normals = o3d.utility.Vector3dVector([])
    return mesh


"""
Converts an Open3D TriangleMesh object to a trimesh object
This mesh only transfers vertices, faces, vertex colors,  vertex normals, face_normals

Args:
    o3d_mesh: Open3D TriangleMesh object
    
Returns:
    mesh: trimesh object
"""
def trimesh_from_o3d(o3d_mesh):
    vertices = np.array(o3d_mesh.vertices)
    faces = np.array(o3d_mesh.triangles)
    vertex_colors = np.array(o3d_mesh.vertex_colors)
    o3d_mesh.compute_vertex_normals()
    vertex_normals = np.array(o3d_mesh.vertex_normals)
    o3d_mesh.compute_triangle_normals()
    face_normals = np.array(o3d_mesh.triangle_normals)
    mesh = trimesh.Trimesh(
        vertices = vertices,
        faces = faces,
        face_normals = face_normals,
        vertex_normals = vertex_normals,
        vertex_colors = vertex_colors,
        process = False
    )
    return mesh


"""
Generates SDF from a trimesh object, and save relevant files to a given directory.

Args:
    mesh:                   input trimesh object
    dest_json_path:         output JSON path
    dest_sdf_path:          output SDF path
    dest_voxel_mesh_path:   output voxel mesh (obj) path; default "" and will not save voxel mesh
    grid_dim:               SDF grid_dim (N); default 256
    print_time              if True, prints time taken for generating SDF; default True
    
Returns:
    centroid:               centroid of the SDF
    extents:                extents of the SDF
    sdf:                    numy array of N x N x N containing SDF values 
"""
def generate_sdf(mesh, dest_json_path, dest_sdf_path, dest_voxel_mesh_path="", grid_dim=256, print_time=True):
    from mesh_to_sdf import mesh_to_voxels, sample_sdf_near_surface
    # Save centroid and extents data used for transforming vertices to [-1,1] while query
    # vertices = mesh.vertices - mesh.bounding_box.centroid
    # vertices *= 2 / np.max(mesh.bounding_box.extents)
    centroid = mesh.bounding_box.centroid
    extents = mesh.bounding_box.extents
    # Save centroid and extents as SDF
    json_dict = {}
    json_dict['centroid'] = centroid.tolist()
    json_dict['extents'] = extents.tolist()
    json_dict['grid_dim'] = grid_dim
    json.dump(json_dict, open(dest_json_path, 'w'))
    
    if print_time:
        start_time = time.time() 
    
    sdf = mesh_to_voxels(mesh, voxel_resolution=grid_dim)
    
    if dest_voxel_mesh_path != "":
        import skimage.measure
        vertices, faces, normals, _ = skimage.measure.marching_cubes(sdf, level=0)
        voxel_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
        voxel_mesh.export(open(dest_voxel_mesh_path, 'w'), file_type='obj')
    
    if print_time:
        print("Generating SDF took {0} seconds".format(time.time()-start_time))
    
    np.save(dest_sdf_path, sdf)
    
    centroid = np.copy(centroid)
    extents = np.copy(extents)
    
    return centroid, extents, sdf
    

"""
Creates per frame human mesh for sequence

Args:
    vertices_path: directory storing npy files of vertices
    faces_path:    path to faces obj file; default: POSA_dir/mesh_ds/mesh_2.obj
    
Returns:
    meshes: list of Open3D TriangleMesh representing human at each frame
"""
def read_sequence_human_mesh(vertices_path, faces_path=os.path.join("mesh_ds", "mesh_2.obj")):
    vertices = np.load(open(vertices_path, "rb"))
    faces = trimesh.load(faces_path, process=False).faces
    meshes = []
    for frame in range(vertices.shape[0]):
        meshes.append(create_o3d_mesh_from_vertices_faces(vertices[frame], faces))
    return meshes


"""
Merges a list of meshes into a single mesh

Args:
    meshes: a list of Open3D TriangleMesh
    
Returns:
    mesh: merged mesh
"""
def merge_meshes(meshes, skip_step=0):
    mesh_vertices = []
    mesh_faces = []
    num_verts_seen = 0
    if skip_step == 0:
        idxs = list(range(0, len(meshes)))
    else:
        idxs = list(range(0, len(meshes), skip_step))
    for i in idxs:
        mesh = meshes[i]
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.triangles)
        mesh_vertices.append(vertices)
        mesh_faces.append(faces + num_verts_seen)
        num_verts_seen += vertices.shape[0]
    mesh_vertices = np.concatenate(mesh_vertices, axis=0)
    mesh_faces = np.concatenate(mesh_faces, axis=0)
    return create_o3d_mesh_from_vertices_faces(mesh_vertices, mesh_faces)
    

"""
Given vertices and faces array, write mesh to path

Args:
    vertices:   mesh vertices as numpy array
    faces:      mesh faces as numpy array
    path:       output path
"""
def write_verts_faces_obj(vertices, faces, path):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.export(open(path, 'w'), file_type='obj')
    
    
    
"""
Given vertices and contact labels, estimate floor height

Args:
    vertices:           human mesh vertices as numpy array, shape: [frames, 655, 3]
    contact_labels:     contact labels for each vertex, shape: [frames, 655, 1]
    floor_offset:       floor distance from lowest cluster median, along negative up axis, in meters; default 0.0
"""
def estimate_floor_height(vertices, contact_labels, floor_offset=0.0):
    from sklearn.cluster import DBSCAN
    
    floor_verts_heights = []
    for frame in range(contact_labels.shape[0]):
        floor_verts = vertices[frame, contact_labels[frame] == 2]
        if len(floor_verts) > 0:
            floor_verts_heights.append(floor_verts[:,2].min())
    floor_verts_heights = np.array(floor_verts_heights)
    clustering = DBSCAN(eps=0.005, min_samples=3).fit(np.expand_dims(floor_verts_heights, axis=1))    
    min_median = float('inf')
    all_labels = clustering.labels_
    for label in np.unique(all_labels):
        clustered_heights = floor_verts_heights[all_labels == label]
        median = np.median(clustered_heights)
        if median < min_median:
            min_median = median
    return min_median - floor_offset
    

"""
Given object vertices and faces, rotate object around X axis for 90 degrees, and move object such that lowest vertex has z = 0

Args:
    vertices:   object mesh vertices as numpy array
    faces:      object mesh faces as numpy array
    write_path: if not empty, write aligned object to given path; default ""
"""    
def align_obj_to_floor(verts, faces, write_path=""):
    from scipy.spatial.transform import Rotation as R
    # zyx is different from ZYX!!
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_euler.html
    # "Up to 3 characters belonging to the set {‘X’, ‘Y’, ‘Z’} for intrinsic rotations, or {‘x’, ‘y’, ‘z’} for extrinsic rotations"
    r = R.from_euler('XYZ', np.asarray([90, 0, 0]), degrees=True) 
    aligned_verts = r.apply(verts) 
    min_z_val = aligned_verts[:, 2].min()
    height_trans = 0 - min_z_val  
    aligned_verts[:, 2] += height_trans
    if write_path:
        print("Writing floor aligned obj to", write_path)
        write_verts_faces_obj(aligned_verts, faces, write_path)
    return aligned_verts

