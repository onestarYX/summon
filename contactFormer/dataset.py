import os
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import data_utils as du
from tqdm import tqdm

'''
Return a segment of a motion sequence in PROXD dataset.
'''
class ProxSegDataset(Dataset):
    def __init__(self, data_dir, fix_orientation=False, no_obj_classes=8, train_seg_len=32,
                 ds_weights_path="./support_files/downsampled_weights.npy", jump_step=1, step_multiplier=1, **kwargs):
        self.data_dir = data_dir
        self.semantics_dir = os.path.join(data_dir, "semantics")
        self.vertices_can_dir = os.path.join(data_dir, "vertices_can")
        self.seq_names = [f.split('cf')[0] for f in os.listdir(self.semantics_dir)]

        self.vertices_can_dict = dict()
        self.semantics_dict = dict()
        self.total_frames = 0
        for seq_name in self.seq_names:
            self.vertices_can_dict[seq_name] = torch.tensor(np.load(os.path.join(self.vertices_can_dir, seq_name + "verts_can.npy")), dtype=torch.float32)
            self.semantics_dict[seq_name] = torch.tensor(np.load(os.path.join(self.semantics_dir, seq_name + "cfs.npy")), dtype=torch.float32)
            self.total_frames += self.vertices_can_dict[seq_name].size(0)

        self.fix_orientation = fix_orientation
        self.no_obj_classes = no_obj_classes
        self.train_seg_len = train_seg_len
        self.ds_weights_path = ds_weights_path
        self.ds_weights = None
        self.associated_joints = None
        if fix_orientation:
            self.ds_weights = torch.tensor(np.load(self.ds_weights_path))
            self.associated_joints = torch.argmax(self.ds_weights, dim=1)

        self.jump_step = jump_step
        self.step_multiplier = step_multiplier

    def __len__(self):
        return self.step_multiplier * self.total_frames // self.train_seg_len

    def __getitem__(self, idx):
        seq_idx = torch.randint(len(self.seq_names), size=(1,))
        seq_name = self.seq_names[seq_idx]
        verts_can = self.vertices_can_dict[seq_name]
        contacts_s = self.semantics_dict[seq_name]

        contacts_s = torch.zeros(*contacts_s.shape, self.no_obj_classes, dtype=torch.float32).scatter_(-1,
                                                                                    contacts_s.unsqueeze(-1).type(torch.long),
                                                                                    1.)
        if self.jump_step > 1:
            idx_start = torch.randint(verts_can.shape[0] - 1 - self.train_seg_len * self.jump_step, size=(1,))
            idx_end = idx_start + self.train_seg_len * self.jump_step
            ret_verts_can = verts_can[idx_start:idx_end:self.jump_step]
            ret_contacts_s = contacts_s[idx_start:idx_end:self.jump_step]
        else:
            idx_start = torch.randint(verts_can.shape[0] - 1 - self.train_seg_len, size=(1,))
            idx_end = idx_start + self.train_seg_len
            ret_verts_can = verts_can[idx_start:idx_end]
            ret_contacts_s = contacts_s[idx_start:idx_end]

        if self.fix_orientation:
            ret_verts_can = du.normalize_orientation(ret_verts_can, self.associated_joints, torch.device("cpu"))

        return ret_verts_can, ret_contacts_s


'''
Support dataset for legacy variant of ContactFormer.
'''
class ProxSegDataset_seq(Dataset):
    def __init__(self, data_dir, fix_orientation=False, no_obj_classes=8, train_seg_len=32, num_seg=8,
                 ds_weights_path="./support_files/downsampled_weights.npy", step_multiplier=1, stride=32,
                 jump_step=1, **kwargs):
        self.data_dir = data_dir
        self.contacts_s_dir = os.path.join(data_dir, "semantics")
        self.vertices_can_dir = os.path.join(data_dir, "vertices_can")
        self.seq_names = [f.split('cfs')[0] for f in os.listdir(self.contacts_s_dir)]

        self.vertices_can_dict = dict()
        self.contacts_s_dict = dict()
        self.total_frames = 0
        for seq_name in self.seq_names:
            self.vertices_can_dict[seq_name] = torch.tensor(np.load(os.path.join(self.vertices_can_dir, seq_name + "verts_can.npy")), dtype=torch.float32)
            self.contacts_s_dict[seq_name] = torch.tensor(np.load(os.path.join(self.contacts_s_dir, seq_name + "cfs.npy")), dtype=torch.float32)
            self.total_frames += self.vertices_can_dict[seq_name].size(0)

        self.fix_orientation = fix_orientation
        self.no_obj_classes = no_obj_classes
        self.train_seg_len = train_seg_len
        self.ds_weights_path = ds_weights_path
        self.ds_weights = None
        self.associated_joints = None
        if fix_orientation:
            self.ds_weights = torch.tensor(np.load(self.ds_weights_path))
            self.associated_joints = torch.argmax(self.ds_weights, dim=1)
        self.num_seg = num_seg
        self.step_multiplier = step_multiplier
        self.stride = stride
        self.jump_step = jump_step

    def __len__(self):
        return self.step_multiplier * self.total_frames // (self.train_seg_len * self.num_seg)

    def __getitem__(self, idx):
        while(True):
            seq_idx = torch.randint(len(self.seq_names), size=(1,))
            seq_name = self.seq_names[seq_idx]
            verts_can = self.vertices_can_dict[seq_name]

            max_idx_start = verts_can.shape[0] - 1 - \
                            (self.train_seg_len + (self.num_seg - 1) * self.stride) * self.jump_step
            if max_idx_start > 0:
                idx_start = torch.randint(max_idx_start, size=(1,))
                break
            # print(f"reject {seq_name}")

        contacts_s = self.contacts_s_dict[seq_name]

        contacts_s = torch.zeros(*contacts_s.shape, self.no_obj_classes, dtype=torch.float32).scatter_(-1,
                                                                                    contacts_s.unsqueeze(-1).type(torch.long),
                                                                                    1.)

        idx_end = idx_start + self.train_seg_len * self.jump_step
        ret_verts_can = []
        ret_contacts_s = []
        for i in range(self.num_seg):
            cur_verts_can = verts_can[idx_start:idx_end:self.jump_step]
            cur_contacts_s = contacts_s[idx_start:idx_end:self.jump_step]
            if self.fix_orientation:
                cur_verts_can = du.normalize_orientation(cur_verts_can, self.associated_joints, torch.device("cpu"))
            ret_verts_can.append(cur_verts_can)
            ret_contacts_s.append(cur_contacts_s)
            idx_start += self.stride * self.jump_step
            idx_end += self.stride * self.jump_step

        ret_verts_can = torch.stack(ret_verts_can, dim=0)
        ret_contacts_s = torch.stack(ret_contacts_s, dim=0)

        return ret_verts_can, ret_contacts_s


'''
Support dataset for legacy variant of ContactFormer.
'''
class ProxSegDataset_var(Dataset):    # when jump_step=8, for a whole seq, dataset's max_frame is 165, max num_seg is 29
    def __init__(self, data_dir, fix_orientation=False, no_obj_classes=8, max_frame=128, num_seg=10, dist_eps=0.7,
                 ds_weights_path="./support_files/downsampled_weights.npy", jump_step=8, step_multiplier=1, **kwargs):
        self.data_dir = data_dir
        self.contacts_s_dir = os.path.join(data_dir, "semantics")
        self.vertices_can_dir = os.path.join(data_dir, "vertices_can")
        self.vertices_dir = os.path.join(data_dir, "vertices")
        self.seq_names = [f.split('cfs')[0] for f in os.listdir(self.contacts_s_dir)]

        self.vertices_can_dict = dict()
        self.vertices_dict = dict()
        self.contacts_s_dict = dict()
        self.max_frame = max_frame
        self.num_seg = num_seg
        self.dist_eps = dist_eps

        self.total_frames = 0
        for seq_name in self.seq_names:
            self.vertices_can_dict[seq_name] = torch.tensor(np.load(os.path.join(self.vertices_can_dir, seq_name + "verts_can.npy")), dtype=torch.float32)
            self.contacts_s_dict[seq_name] = torch.tensor(np.load(os.path.join(self.contacts_s_dir, seq_name + "cfs.npy")), dtype=torch.float32)
            self.vertices_dict[seq_name] = torch.tensor(np.load(os.path.join(self.vertices_dir, seq_name + "verts.npy")), dtype=torch.float32)
            self.total_frames += self.vertices_can_dict[seq_name].size(0)

        self.fix_orientation = fix_orientation
        self.no_obj_classes = no_obj_classes
        self.ds_weights_path = ds_weights_path
        self.ds_weights = None
        self.associated_joints = None
        if fix_orientation:
            self.ds_weights = torch.tensor(np.load(self.ds_weights_path))
            self.associated_joints = torch.argmax(self.ds_weights, dim=1)

        self.jump_step = jump_step
        self.step_multiplier = step_multiplier

    def __len__(self):
        return self.step_multiplier * self.total_frames // (self.max_frame * self.num_seg)

    def __getitem__(self, idx):
        seq_idx = torch.randint(len(self.seq_names), size=(1,))
        seq_name = self.seq_names[seq_idx]
        verts_can = self.vertices_can_dict[seq_name]
        contacts_s = self.contacts_s_dict[seq_name]
        verts = self.vertices_dict[seq_name]

        verts_center = torch.mean(verts[:, :, :2], dim=1)
        contacts_s = torch.zeros(*contacts_s.shape, self.no_obj_classes, dtype=torch.float32)\
            .scatter_(-1, contacts_s.unsqueeze(-1).type(torch.long), 1.)

        ret_verts_can = []
        ret_contacts_s = []
        ret_masks = []
        start_idx = torch.randint(verts_can.shape[0] // 2, size=(1,))
        for i in range(self.num_seg):
            if start_idx >= verts.shape[0]:
                mask = torch.zeros(self.max_frame)
                ret_masks.append(mask)
                cur_verts_can = torch.zeros(self.max_frame, *verts_can.shape[1:])
                cur_contacts_s = torch.zeros(self.max_frame, *contacts_s.shape[1:])
                ret_verts_can.append(cur_verts_can)
                ret_contacts_s.append(cur_contacts_s)
                continue

            cur_center = verts_center[start_idx]
            remaining_centers = verts_center[start_idx::self.jump_step]
            if remaining_centers.shape[0] == 0:
                mask = torch.zeros(self.max_frame)
                ret_masks.append(mask)
                cur_verts_can = torch.zeros(self.max_frame, *verts_can.shape[1:])
                cur_contacts_s = torch.zeros(self.max_frame, *contacts_s.shape[1:])
                ret_verts_can.append(cur_verts_can)
                ret_contacts_s.append(cur_contacts_s)
                continue

            remaining_centers -= cur_center
            dist = torch.linalg.norm(remaining_centers, dim=1)
            dist = (dist > self.dist_eps).to(torch.int32)
            if dist.sum() == 0: # TODO: correct this
                mask = torch.zeros(self.max_frame)
                ret_masks.append(mask)
                cur_verts_can = torch.zeros(self.max_frame, *verts_can.shape[1:])
                cur_contacts_s = torch.zeros(self.max_frame, *contacts_s.shape[1:])
                ret_verts_can.append(cur_verts_can)
                ret_contacts_s.append(cur_contacts_s)
                continue
            end_idx = torch.argmax(dist)
            end_idx = start_idx + end_idx * self.jump_step
            cur_verts_can = verts_can[start_idx:end_idx:self.jump_step]
            cur_contacts_s = contacts_s[start_idx:end_idx:self.jump_step]
            cur_seg_len = cur_verts_can.shape[0]
            if cur_seg_len > self.max_frame:
                cur_seg_len = self.max_frame
                cur_verts_can = cur_verts_can[:cur_seg_len]
                cur_contacts_s = cur_contacts_s[:cur_seg_len]

            if self.fix_orientation:
                cur_verts_can = du.normalize_orientation(cur_verts_can, self.associated_joints, torch.device("cpu"))

            # masking
            mask = torch.zeros(self.max_frame)
            mask[:cur_seg_len] = 1
            ret_masks.append(mask)
            cur_verts_can_pad = torch.zeros(self.max_frame - cur_seg_len, *cur_verts_can.shape[1:])
            cur_verts_can = torch.cat([cur_verts_can, cur_verts_can_pad], dim=0)
            cur_contacts_s_pad = torch.zeros(self.max_frame - cur_seg_len, *cur_contacts_s.shape[1:])
            cur_contacts_s = torch.cat([cur_contacts_s, cur_contacts_s_pad], dim=0)
            ret_verts_can.append(cur_verts_can)
            ret_contacts_s.append(cur_contacts_s)

            start_idx += cur_seg_len * self.jump_step

        ret_verts_can = torch.stack(ret_verts_can, dim=0)
        ret_contacts_s = torch.stack(ret_contacts_s, dim=0)
        ret_masks = torch.stack(ret_masks, dim=0)
        return ret_verts_can, ret_contacts_s, ret_masks

'''
Downsampled version of PROXD dataset (by applying frame skipping).
Used for training/testing final version of ContactFormer.
'''
class ProxDataset_ds(Dataset):    # when jump_step=8, for a whole seq, dataset's max_frame is 165, max num_seg is 29
    def __init__(self, data_dir, fix_orientation=False, no_obj_classes=8, max_frame=220,
                 ds_weights_path="./support_files/downsampled_weights.npy", jump_step=8, step_multiplier=1, **kwargs):
        '''
            data_dir: directory that stores processed PROXD dataset.
            fix_orientation: flag that specifies whether we always make the first pose in a motion sequence facing
                             towards a canonical direction.
            no_obj_classes: number of contact object classes.
            max_frame: the maximum motion sequence length which the model accepts (after applying frame skipping).
            ds_weights_path: the saved downsampling matrix for downsampling body vertices.
            jump_step: for every jump_step frames, we only select the first frame for some sequence.
            step_multiplier: a dummy parameter used to control the number of examples seen in each epoch (You can
                             ignore it if you don't know how to adjust it).
        '''
        self.data_dir = data_dir
        self.contacts_s_dir = os.path.join(data_dir, "semantics")
        self.vertices_can_dir = os.path.join(data_dir, "vertices_can")
        self.vertices_dir = os.path.join(data_dir, "vertices")
        self.seq_names = [f.split('cfs')[0] for f in os.listdir(self.contacts_s_dir)]

        self.vertices_can_dict = dict()
        self.vertices_dict = dict()
        self.contacts_s_dict = dict()
        self.max_frame = max_frame

        self.total_frames = 0
        for seq_name in self.seq_names:
            self.vertices_can_dict[seq_name] = torch.tensor(np.load(os.path.join(self.vertices_can_dir, seq_name + "verts_can.npy")), dtype=torch.float32)
            self.contacts_s_dict[seq_name] = torch.tensor(np.load(os.path.join(self.contacts_s_dir, seq_name + "cfs.npy")), dtype=torch.float32)
            self.vertices_dict[seq_name] = torch.tensor(np.load(os.path.join(self.vertices_dir, seq_name + "verts.npy")), dtype=torch.float32)
            self.total_frames += self.vertices_can_dict[seq_name].shape[0]

        self.fix_orientation = fix_orientation
        self.no_obj_classes = no_obj_classes
        self.ds_weights_path = ds_weights_path
        self.ds_weights = None
        self.associated_joints = None
        if fix_orientation:
            self.ds_weights = torch.tensor(np.load(self.ds_weights_path))
            self.associated_joints = torch.argmax(self.ds_weights, dim=1)

        self.jump_step = jump_step
        self.step_multiplier = step_multiplier

    def __len__(self):
        return self.step_multiplier * self.total_frames // self.max_frame

    def __getitem__(self, idx):
        seq_idx = torch.randint(len(self.seq_names), size=(1,))
        seq_name = self.seq_names[seq_idx]
        verts_can = self.vertices_can_dict[seq_name]
        contacts_s = self.contacts_s_dict[seq_name]
        verts = self.vertices_dict[seq_name]

        contacts_s = torch.zeros(*contacts_s.shape, self.no_obj_classes, dtype=torch.float32)\
            .scatter_(-1, contacts_s.unsqueeze(-1).type(torch.long), 1.)

        if self.max_frame * self.jump_step > verts.shape[0]:
            start_idx = torch.randint(self.jump_step, size=(1,))
            end_idx = verts.shape[0]
        else:
            start_idx = torch.randint(verts.shape[0] - self.max_frame * self.jump_step, size=(1,))
            end_idx = start_idx + self.max_frame * self.jump_step

        ret_verts_can = verts_can[start_idx:end_idx:self.jump_step]
        if self.fix_orientation:
            ret_verts_can = du.normalize_orientation(ret_verts_can, self.associated_joints, torch.device("cpu"))
        ret_contacts_s = contacts_s[start_idx:end_idx:self.jump_step]

        # masking
        seg_len = ret_verts_can.shape[0]
        mask = torch.zeros(self.max_frame)
        mask[:seg_len] = 1
        ret_verts_can_pad = torch.zeros(self.max_frame - seg_len, *ret_verts_can.shape[1:])
        ret_verts_can = torch.cat([ret_verts_can, ret_verts_can_pad], dim=0)
        ret_contacts_s_pad = torch.zeros(self.max_frame - seg_len, *ret_contacts_s.shape[1:])
        ret_contacts_s = torch.cat([ret_contacts_s, ret_contacts_s_pad], dim=0)

        return ret_verts_can, ret_contacts_s, mask


if __name__ == "__main__":
    train_dataset = ProxDataset_ds("../data/new_posa_temp_train", fix_orientation=True, max_frame=220, jump_step=8)
    train_data_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    verts_can, contacts_s, mask = next(iter(train_data_loader))
    print(verts_can.shape)
    print(contacts_s.shape)
    print(mask.shape)


    start = time.time()
    for _ in tqdm(train_data_loader):
        pass
    end = time.time()
    print(f"Enumerate: {end - start}")

    # start = time.time()
    # for _ in range(count):
    #     _, _, _ = next(iter(train_data_loader))
    # end = time.time()
    # print(f"Iterator: {end - start}")