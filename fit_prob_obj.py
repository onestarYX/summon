import argparse
import os

from utils import read_mpcat40, pred_subset_to_mpcat40, estimate_floor_height


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("sequence_name",
                        type=str)
    parser.add_argument("vertices_path",
                        type=str)
    parser.add_argument("contact_probs_path",
                        type=str)
    parser.add_argument("sample_count",
                        type=int)
    args = parser.parse_args()
    
    sequence_name = args.sequence_name
    
    vertices = np.load(open(args.vertices_path, "rb"))
    
    label_names, color_coding_rgb = read_mpcat40()
    
    contact_probs = np.load(open(args.contact_labels_path, "rb"))
    
    contact_labels = np.argmax(contact_labels, axis=-1)
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
    human_sdf_base_path = os.path.join("models", sequence_name, "human")
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
    
    # Use most probable contact labels for floor estimation
    floor_height = estimate_floor_height(vertices, contact_labels)
    print("Estimated floor height is", floor_height)
    print()
    print()
    
    # Use most probable non-contact labels for contact clustering
    
