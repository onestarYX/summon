import os
import argparse
import torch
import numpy as np
import open3d as o3d
import trimesh
from tqdm import tqdm

import data_utils as du
import vis_utils as vu


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset for temporal_POSA")
    parser.add_argument("--data_dir", type=str, default="../data/posa_temp_train/contacts_semantics",
                        help="path to POSA_temp dataset dir")
    parser.add_argument("--output_dir", type=str, default="../data/new_posa_temp_train")

    # Parse arguments and assign directories
    args = parser.parse_args()
    semantic_dir = args.data_dir
    output_dir = args.output_dir
    save_cfs_dir = os.path.join(output_dir, "semantics")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(save_cfs_dir, exist_ok=True)

    for file_name in tqdm(os.listdir(semantic_dir)):
        semantics = np.load(os.path.join(semantic_dir, file_name))
        scene_name = file_name.split("_")[0]

        # Remove old 4(door), 6(picture), 7(cabinet)
        semantics[semantics == 4] = 0
        semantics[semantics == 6] = 0
        semantics[semantics == 7] = 0
        # Move 10(sofa) to 4, 8(cushion) to 4 or 6
        semantics[semantics == 10] = 4
        if scene_name == "MPH112":  # Only in MPH112, cushion is on a bed
            semantics[semantics == 8] = 6
        else:
            semantics[semantics == 8] = 4
        # Move 11(bed) to 6
        semantics[semantics == 11] = 6
        # Move 19(stool) to 7
        semantics[semantics == 19] = 7
        # Remove all other semantics
        semantics[semantics > 7] = 0

        np.save(os.path.join(save_cfs_dir, file_name), semantics)