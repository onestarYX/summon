classes_eps = {
    3:  0.2, # chair
    5:  1.0, # table
    10: 0.8, # sofa
    11: 1.0, # bed
    19: 0.2  # stool
}

voting_eps = 0.1

voxel_size = 0.04

cluster_min_points = 10

pts_per_unit = 20

params = {
    "default": {
        "grid_search_contact_weight": 100,
        "grid_search_pen_thresh": -0.05,
        "grid_search_classes_pen_weight": {
            3:  10,  # chair
            5:  10,  # table
            10: 10,  # sofa
            11: 10,  # bed
            19: 10   # stool
        },
        "lr": 0.003,
        "opt_steps": 200,
        "opt_contact_weight": 100,
        "opt_pen_thresh": 0.0,
        "opt_classes_pen_weight": {
            3:  1,   # chair
            5:  100, # table
            10: 10,  # sofa
            11: 10,  # bed
            19: 1    # stool
        }
    }
}
