import argparse
import os
import numpy as np
import torch
from pathlib import Path

from atiss.scripts.training_utils import load_config
from atiss.scene_synthesis.networks import build_network
from atiss.scene_synthesis.datasets.threed_future_dataset import ThreedFutureDataset


def sample_in_bbox(class_probs, translation_probs, bbox, rejected_points, trials=1000):
    """Do rejection sampling to sample the class and translation from the given
    probabilities."""

    def in_bbox(bbox, x, y, z):
        return (
                bbox[0] <= x <= bbox[1] and
                bbox[2] <= y <= bbox[3] and
                bbox[4] <= z <= bbox[5]
        )

    def sample_dmll(probs, mu, s):
        i = np.random.choice(len(probs), p=probs)
        u = np.random.rand()
        return np.clip(
            mu[i] + s[i] * (np.log(u) - np.log(1 - u)),
            -1,
            1
        )

    # Prepare the probs for sampling (casting to numpy basically)
    translation_probs = [
        [
            (p.cpu().numpy().ravel(), mu.cpu().numpy().ravel(), s.cpu().numpy().ravel())
            for (p, mu, s) in lc
        ] for lc in translation_probs
    ]

    # How many trials to do before giving up
    N = trials

    # Sample the class labels
    classes = np.random.choice(len(class_probs), N, p=class_probs)
    for i in range(N):
        if classes[i] >= len(translation_probs):
            continue

        c = classes[i]
        x, y, z = [sample_dmll(*di) for di in translation_probs[c]]
        # print("Suggesting", object_types[c], "at", x, y, z)
        if in_bbox(bbox, x, y, z):
            return c, (x, y, z)
        else:
            rejected_points.append([x, y, z])

    raise RuntimeError("Couldn't sample in the bbox")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--fitting_results_path", type=str, help="Fitting result directory for some sequence")
    parser.add_argument("--path_to_model", type=str, help="Path to ATISS model checkpoint")
    args = parser.parse_args()

    fitting_results_path = Path(args.fitting_results_path) / 'fit_best_obj'

    # if torch.cuda.is_available():
    #     print("Using cuda")
    #     device = torch.device("cuda")
    # else:
    #     print("Using cpu")
    device = torch.device("cpu")

    config_path = os.path.join("atiss", "config", "bedrooms_eval_config.yaml")
    config = load_config(config_path)

    # This assumes bedroom model and dataset
    weight_file = args.path_to_model
    network, _, _ = build_network(
        30, 23,
        config, weight_file, device=device
    )
    network.eval()

    object_types = [
        'armchair',
        'bookshelf',
        'cabinet',
        'ceiling_lamp',
        'chair',
        'children_cabinet',
        'coffee_table',
        'desk',
        'double_bed',
        'dressing_chair',
        'dressing_table',
        'kids_bed',
        'nightstand',
        'pendant_lamp',
        'shelf',
        'single_bed',
        'sofa',
        'stool',
        'table',
        'tv_stand',
        'wardrobe']

    total_num_obj = 0
    for obj_class_dir in fitting_results_path.iterdir():
        for obj_dir in obj_class_dir.iterdir():
            total_num_obj += 1

    boxes = {}
    boxes['class_labels'] = torch.zeros((1, total_num_obj, 23)).to(device)
    boxes['translations'] = torch.zeros((1, total_num_obj, 3)).to(device)
    boxes['sizes'] = torch.zeros((1, total_num_obj, 3)).to(device)
    boxes['angles'] = torch.zeros((1, total_num_obj, 1)).to(device)

    # Fill in object attributes
    item_idx = 0
    for obj_class_dir in fitting_results_path.iterdir():
        for obj_dir in obj_class_dir.iterdir():
            obj_class = obj_class_dir.stem
            obj_class_idx = object_types.index(obj_class)
            boxes['class_labels'][0, item_idx, obj_class_idx] = 1
            item_idx += 1
            # TODO: to get a better estimation of next class distribution, we shall fill in translations/angles/sizes

    # Entire space of room is available
    room_mask = torch.ones((1, 1, 64, 64)).to(device)
    class_prob = network.distribution_classes(boxes, room_mask)
    out_path = Path(args.fitting_results_path) / 'atiss_out.npy'
    np.save(out_path, class_prob.cpu().detach().numpy())