import os
from pathlib import Path
import numpy as np
import argparse
import open3d as o3d
from gen_human_meshes import gen_human_meshes
import json
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--fitting_results_path", type=str, help="Path to the fitting results of some motion sequence")
    parser.add_argument("--vertices_path", type=str, help="Path to human vertices of some motion sequence")

    args = parser.parse_args()
    input_dir = Path(args.fitting_results_path)
    vertices_path = Path(args.vertices_path)
    seq_name = input_dir.stem

    # Check if human meshes are there
    human_mesh_dir = input_dir / 'human' / 'mesh'
    if not human_mesh_dir.exists():
        human_mesh_dir.mkdir()
        gen_human_meshes(vertices_path=vertices_path, output_path=human_mesh_dir)

    # Rendering results
    output_dir = input_dir / 'rendering'
    output_dir.mkdir(exist_ok=True)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])

    res_dir = input_dir / 'fit_best_obj'
    obj_mesh_list = []
    for obj_class_dir in res_dir.iterdir():
        for obj_dir in obj_class_dir.iterdir():
            with open(str(obj_dir / 'best_obj_id.json'), "r") as f:
                best_obj_json = json.load(f)
            best_obj_id = best_obj_json['best_obj_id']
            best_obj_path = obj_dir / best_obj_id / 'opt_best.obj'
            obj_mesh = o3d.io.read_triangle_mesh(str(best_obj_path))
            obj_mesh.compute_vertex_normals()
            obj_mesh_list.append(obj_mesh)

    frame = 0
    for _ in tqdm(human_mesh_dir.iterdir()):
        mesh_list = [] + obj_mesh_list
        human_mesh_path = human_mesh_dir / f"human_{frame}.ply"
        human_mesh = o3d.io.read_triangle_mesh(str(human_mesh_path))
        human_mesh.compute_vertex_normals()
        mesh_list.append(human_mesh)

        for geometry in mesh_list:
            vis.add_geometry(geometry)

        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(str(output_dir / f"frame_{frame:04d}.png"))

        for geometry in mesh_list:
            vis.remove_geometry(geometry)
        frame += 1

    vis.destroy_window()