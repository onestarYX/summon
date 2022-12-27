import argparse
import json
import math
import os
import time

import numpy as np
import open3d as o3d

import torch

from scipy.spatial.transform import Rotation as R

import config

from place_obj_opt import grid_search, optimization
from utils import read_mpcat40, pred_subset_to_mpcat40, estimate_floor_height, read_sequence_human_mesh, merge_meshes, generate_sdf, trimesh_from_o3d, create_o3d_pcd_from_points, write_verts_faces_obj, create_o3d_mesh_from_vertices_faces, align_obj_to_floor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--sequence_name", type=str)
    parser.add_argument("--vertices_path", type=str)
    parser.add_argument("--contact_labels_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--input_probability",
                        action='store_true', 
                        default=False)
    args = parser.parse_args()
    
    sequence_name = args.sequence_name
    output_dir = args.output_dir
    
    vertices = np.load(open(args.vertices_path, "rb"))
    
    contact_labels = np.load(open(args.contact_labels_path, "rb"))
    
    if args.input_probability:
        contact_labels = np.argmax(contact_labels, axis=-1)
    
    label_names, color_coding_rgb = read_mpcat40()
    
    contact_labels = contact_labels.squeeze()
    
    # Map contact labels to predicted subset
    vertices_down = []
    contact_labels_mapped = []
    for frame in range(contact_labels.shape[0]):
        contact_labels_mapped.append(pred_subset_to_mpcat40[contact_labels[frame]])
        vertices_down.append(vertices[frame * 8])
    vertices = np.array(vertices_down)
    contact_labels = np.array(contact_labels_mapped)
    
    # Load fitting parameters
    classes_eps = config.classes_eps
    pcd_down_voxel_size = config.voxel_size
    voting_eps = config.voting_eps
    cluster_min_points = config.cluster_min_points
    if sequence_name in config.params:
        params = config.params[sequence_name]
    else:
        print("Sequence specific parameters undefined, using default")
        print()
        print()
        params = config.params["default"]
    grid_search_contact_weight = params["grid_search_contact_weight"]
    grid_search_pen_thresh = params["grid_search_pen_thresh"]
    grid_search_classes_pen_weight = params["grid_search_classes_pen_weight"]
    lr = params["lr"]
    opt_steps = params["opt_steps"]
    opt_contact_weight = params["opt_contact_weight"]
    opt_pen_thresh = params["opt_pen_thresh"]
    opt_classes_pen_weight = params["opt_classes_pen_weight"]
    
    # Get cuda
    if torch.cuda.is_available():
        print("Using cuda")
        device = torch.device("cuda")
    else:
        print("Using cpu")
        device = torch.device("cpu")
    print()
    print()
    
    # Create human SDF
    human_meshes = read_sequence_human_mesh(args.vertices_path)
    merged_human_meshes = merge_meshes(human_meshes)
    grid_dim = 256
    human_sdf_base_path = os.path.join(output_dir, sequence_name, "human")
    if not os.path.exists(human_sdf_base_path):
        os.makedirs(human_sdf_base_path) 
    sdf_path = os.path.join(human_sdf_base_path, "sdf.npy")
    json_path = os.path.join(human_sdf_base_path, "sdf.json")
    if os.path.exists(sdf_path) and os.path.exists(json_path):
        print("Human SDF already exists, reading from file")
        json_sdf_info = json.load(open(json_path, 'r'))
        centroid = np.asarray(json_sdf_info['centroid'])    # 3
        extents = np.asarray(json_sdf_info['extents'])      # 3
        sdf = np.load(sdf_path)
    else:
        print("Generating human SDF")
        centroid, extents, sdf = generate_sdf(trimesh_from_o3d(merged_human_meshes), json_path, sdf_path)
    sdf = torch.Tensor(sdf).float().to(device)
    centroid = torch.Tensor(centroid).float().to(device)
    extents = torch.Tensor(extents).float().to(device)
    print()
    print()
    
    # Estimate floor height
    floor_height = estimate_floor_height(vertices, contact_labels)
    print("Estimated floor height is", floor_height)
    print()
    print()
    
    # Local majority voting
    print("Performing local majority voting")
    cluster_contact_points = []
    cluster_contact_labels = []
    num_frames = contact_labels.shape[0]
    for obj_c in classes_eps:
        contact_points_class = []
        for frame in range(num_frames):
            contact_points_class.extend(vertices[frame][contact_labels[frame] == obj_c])
        if len(contact_points_class) == 0:
            continue
        contact_points_class = np.array(contact_points_class)
        contact_pcd = o3d.geometry.PointCloud()
        contact_pcd.points = o3d.utility.Vector3dVector(contact_points_class)
        contact_pcd = contact_pcd.voxel_down_sample(voxel_size=pcd_down_voxel_size)
        contact_points_class = np.array(contact_pcd.points)
        cluster_contact_points.extend(contact_points_class)
        cluster_contact_labels.extend(np.full((contact_points_class.shape[0],), obj_c))
    cluster_contact_points = np.array(cluster_contact_points)
    cluster_contact_labels = np.array(cluster_contact_labels)
    contact_pcd = o3d.geometry.PointCloud()
    contact_pcd.points = o3d.utility.Vector3dVector(cluster_contact_points)
    print()
    print()
    
    # Cluster all points
    print("Clustering all points with eps", voting_eps, "...")
    start_time = time.time()
    cluster_labels = np.array(contact_pcd.cluster_dbscan(eps=voting_eps, min_points=cluster_min_points, print_progress=False))
    max_label = cluster_labels.max()
    print("Clustering took {0} seconds".format(time.time()-start_time))
    print("Num clusters", max_label + 1)
    voted_vertices = []
    voted_labels = []
    for label in range(max_label + 1):
        cluster_points = cluster_contact_points[cluster_labels == label]
        if len(cluster_points) < cluster_min_points:
            continue
        majority_label = np.argmax(np.bincount(cluster_contact_labels[cluster_labels == label]))
        print("Cluster", label, "has", len(cluster_points), "points with majority label of", majority_label, label_names[majority_label])
        voted_vertices.extend(cluster_points)
        voted_labels.extend(np.full(cluster_points.shape[0], majority_label))
    voted_vertices = np.array(voted_vertices)
    voted_labels = np.array(voted_labels)
    voted_vertices = np.expand_dims(voted_vertices, axis=0)
    voted_labels = np.expand_dims(voted_labels, axis=0)
    vertices = voted_vertices
    contact_labels = voted_labels
    print()
    print()
    
    # Cluster points by contact label 
    print("Clustering object points by contact label")
    print()
    clusters_classes = []
    clusters_points = []
    objects_indices = []
    num_frames = contact_labels.shape[0]
    for obj_c in classes_eps:
        print("Object class", obj_c, label_names[obj_c])
        contact_points = []
        for frame in range(num_frames):
            contact_points.extend(vertices[frame][contact_labels[frame] == obj_c])
        print("Num_points", len(contact_points))
        if len(contact_points) == 0:
            print()
            continue
        contact_points = np.array(contact_points)
        contact_pcd = o3d.geometry.PointCloud()
        contact_pcd.points = o3d.utility.Vector3dVector(contact_points)
        contact_pcd = contact_pcd.voxel_down_sample(voxel_size=pcd_down_voxel_size)
        contact_points = np.array(contact_pcd.points)
        print("After downsampling, have", len(contact_points), "points")
        print("Clustering with eps", classes_eps[obj_c], "...")
        start_time = time.time()
        cluster_labels = np.array(contact_pcd.cluster_dbscan(eps=classes_eps[obj_c], min_points=cluster_min_points, print_progress=False))
        max_label = cluster_labels.max()
        print("Clustering took {0} seconds".format(time.time()-start_time))
        print("Num clusters", max_label + 1)
        for label in range(max_label + 1):
            clusters_classes.append(obj_c)
            clusters_points.append(contact_points[cluster_labels == label])
            objects_indices.append(label)
        print()
    print()
    
    # For each cluster, fit best object
    # Iterate by object class, then by clusters of that class
    for i, obj_c in enumerate(clusters_classes):
        cluster_points = clusters_points[i]
        cluster_points_tensor = torch.Tensor(cluster_points).float().to(device)
        obj_idx = objects_indices[i]
        obj_class_str = label_names[obj_c]
        obj_class_path = os.path.join("3D_Future", "models", obj_class_str)
        print("Object class", obj_class_str, "Object index", obj_idx, "Num points", cluster_points.shape[0])
        print()
        cluster_base_path = os.path.join(output_dir, sequence_name, "fit_best_obj", obj_class_str, str(obj_idx))
        if not os.path.exists(cluster_base_path):
            os.makedirs(cluster_base_path)
        # Save cluster PCD for visualization
        cluster_pcd_colors = np.zeros_like(cluster_points)
        cluster_pcd_colors += color_coding_rgb[obj_c]
        cluster_pcd = create_o3d_pcd_from_points(cluster_points, cluster_pcd_colors)
        o3d.io.write_point_cloud(os.path.join(cluster_base_path, "cluster_pcd.ply"), cluster_pcd)
        # Get contact position
        contact_max_x = cluster_points[:,0].max()
        contact_min_x = cluster_points[:,0].min()
        contact_max_y = cluster_points[:,1].max()
        contact_min_y = cluster_points[:,1].min()
        contact_center_x = (contact_max_x + contact_min_x) / 2
        contact_center_y = (contact_max_y + contact_min_y) / 2
        # Info about best fitted object
        best_obj_loss = float("inf")
        best_obj_id = ""
        # For each candidate object
        for obj_dir in os.listdir(obj_class_path):
            obj_path = os.path.join(obj_class_path, obj_dir, "raw_model.obj")
            print("Trying obj at", obj_path)
            obj_mesh = o3d.io.read_triangle_mesh(obj_path)
            obj_verts = np.array(obj_mesh.vertices)
            obj_faces = np.array(obj_mesh.triangles)
            save_obj_base_path = os.path.join(cluster_base_path, obj_dir)
            if not os.path.exists(save_obj_base_path):
                os.makedirs(save_obj_base_path)
            # Align object to floor
            floor_aligned_obj_path = os.path.join(save_obj_base_path, "floor_aligned.obj")
            print("Writing floor aligned object to", floor_aligned_obj_path)
            floor_aligned_verts = align_obj_to_floor(obj_verts, obj_faces, floor_aligned_obj_path)
            transformed_verts = np.copy(floor_aligned_verts)
            transformed_verts[:, 2] += floor_height
            # Transform object to cluster centroid
            obj_max_x = transformed_verts[:,0].max()
            obj_min_x = transformed_verts[:,0].min()
            obj_max_y = transformed_verts[:,1].max()
            obj_min_y = transformed_verts[:,1].min()
            obj_center_x = (obj_max_x + obj_min_x) / 2
            obj_center_y = (obj_max_y + obj_min_y) / 2
            x_transl = contact_center_x - obj_center_x
            y_transl = contact_center_y - obj_center_y
            transformed_verts[:, 0] += x_transl
            transformed_verts[:, 1] += y_transl
            obj_center_x += x_transl
            obj_center_y += y_transl
            obj_max_x += x_transl
            obj_max_y += y_transl
            obj_min_x += x_transl
            obj_min_y += y_transl
            transformed_obj_path = os.path.join(save_obj_base_path, "transformed.obj")
            print("Writing transformed object to", transformed_obj_path)
            write_verts_faces_obj(transformed_verts, obj_faces, transformed_obj_path)
            # Sample points from centered mesh
            obj_max_z = transformed_verts[:,2].max()
            obj_min_z = transformed_verts[:,2].min()
            print("x size", (obj_max_x - obj_min_x))
            print("y size", (obj_max_y - obj_min_y))
            print("z size", (obj_max_z - obj_min_z))
            x_pts = int(math.ceil((obj_max_x - obj_min_x) * config.pts_per_unit))
            y_pts = int(math.ceil((obj_max_y - obj_min_y) * config.pts_per_unit))
            z_pts = int(math.ceil((obj_max_z - obj_min_z) * config.pts_per_unit))
            num_sample_points = x_pts * y_pts * z_pts
            print("Sampling", num_sample_points, "points")
            centered_verts = np.copy(transformed_verts)
            centered_verts[:, 0] -= obj_center_x
            centered_verts[:, 1] -= obj_center_y
            obj_pcd = create_o3d_mesh_from_vertices_faces(centered_verts, obj_faces).sample_points_poisson_disk(number_of_points=num_sample_points)
            obj_pcd = obj_pcd.voxel_down_sample(voxel_size=config.voxel_size)
            obj_points_centered = np.array(obj_pcd.points)
            print("After downsampling, have", len(obj_points_centered), "points")
            # Grid search
            print("Grid Searching...")
            start_time = time.time()
            grid_best_loss, grid_best_rot_deg, grid_best_transl_x, grid_best_transl_y, grid_best_points = grid_search(
                obj_c,
                obj_points_centered,
                obj_center_x, obj_center_y,
                obj_min_x, obj_min_y,
                obj_max_x, obj_max_y,
                cluster_points_tensor,
                contact_min_x, contact_min_y,
                contact_max_x, contact_max_y,
                sdf, centroid, extents,
                grid_search_contact_weight,
                grid_search_pen_thresh,
                grid_search_classes_pen_weight,
                device
            )
            print("Grid search took {0} seconds".format(time.time()-start_time))
            print("Best loss", grid_best_loss)
            print("Best Rotation in degrees", grid_best_rot_deg, "Best x translation", grid_best_transl_x, "Best y translation", grid_best_transl_y)
            r = R.from_euler('XYZ', [0, 0, grid_best_rot_deg], degrees=True)
            candidate_verts_centered = r.apply(centered_verts)
            candidate_verts = np.copy(candidate_verts_centered)
            candidate_verts[:, 0] += obj_center_x + grid_best_transl_x
            candidate_verts[:, 1] += obj_center_y + grid_best_transl_y
            grid_search_best_path = os.path.join(save_obj_base_path, "grid_search_best.obj")
            print("Writing best grid search result to", grid_search_best_path)
            write_verts_faces_obj(candidate_verts, obj_faces, grid_search_best_path)
            json_dict = {}
            json_dict["loss"] = grid_best_loss
            json_dict["rot_deg"] = grid_best_rot_deg
            json_dict["transl_x"] = grid_best_transl_x
            json_dict["transl_y"] = grid_best_transl_y
            json.dump(json_dict, open(os.path.join(save_obj_base_path, "grid_search_best.json"), 'w'))
            grid_pcd_colors = np.zeros_like(grid_best_points)
            grid_pcd_colors += color_coding_rgb[obj_c]
            grid_pcd = create_o3d_pcd_from_points(grid_best_points, grid_pcd_colors)
            o3d.io.write_point_cloud(os.path.join(save_obj_base_path, "grid_search_best.ply"), grid_pcd)
            # Optimization
            grid_center_x = obj_center_x + grid_best_transl_x
            grid_center_y = obj_center_y + grid_best_transl_y
            print("Optimizing...")
            start_time = time.time()
            best_loss, best_rot, best_transl_x, best_transl_y, best_points = optimization(
                obj_c,
                obj_points_centered,
                grid_center_x, grid_center_y,
                grid_best_rot_deg,
                cluster_points_tensor,
                contact_min_x, contact_min_y,
                contact_max_x, contact_max_y,
                sdf, centroid, extents,
                opt_contact_weight,
                opt_pen_thresh,
                opt_classes_pen_weight,
                lr, opt_steps,
                device
            )
            print("Optimization took {0} seconds".format(time.time()-start_time))
            print("Best loss", best_loss)
            print("Best Rotation in degrees", best_rot/math.pi*180, "Best x translation", best_transl_x, "Best y translation", best_transl_y)
            r = R.from_euler('XYZ', [0, 0, best_rot], degrees=False)
            opt_obj_verts = r.apply(candidate_verts_centered)
            opt_obj_verts[:, 0] += grid_center_x + best_transl_x
            opt_obj_verts[:, 1] += grid_center_y + best_transl_y
            opt_best_path = os.path.join(save_obj_base_path, "opt_best.obj")
            print("Writing best optimization result to", opt_best_path)
            write_verts_faces_obj(opt_obj_verts, obj_faces, opt_best_path)
            json_dict = {}
            json_dict["loss"] = best_loss
            json_dict["rot_deg"] = best_rot/math.pi*180
            json_dict["transl_x"] = best_transl_x
            json_dict["transl_y"] = best_transl_y
            json.dump(json_dict, open(os.path.join(save_obj_base_path, "opt_best.json"), 'w'))
            opt_pcd_colors = np.zeros_like(best_points)
            opt_pcd_colors += color_coding_rgb[obj_c]
            opt_pcd = create_o3d_pcd_from_points(best_points, opt_pcd_colors)
            o3d.io.write_point_cloud(os.path.join(save_obj_base_path, "opt_best.ply"), opt_pcd)
            print()
            if best_loss < best_obj_loss:
                best_obj_loss = best_loss
                best_obj_id = obj_dir
        print("Best fitted object has ID", best_obj_id)
        json_dict = {}
        json_dict["best_obj_id"] = best_obj_id
        json.dump(json_dict, open(os.path.join(cluster_base_path, "best_obj_id.json"), 'w'))
        print()
