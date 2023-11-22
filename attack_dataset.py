import random
import os

import numpy as np
import torch.utils.data as data

from dataset import spect_loader

def make_attack_pairs(manifest_path, n_pairs, message_file):
    data_root = '/root/autodl-tmp/TIMIT'
    pairs = []

    with open(manifest_path, 'r') as file:
        lines = [line.strip() for line in file.readlines()]

    for _ in range(n_pairs):
        sampled_file = random.sample(lines, 1)[0]
        carrier_file = os.path.join(data_root, sampled_file)
        pairs.append((carrier_file, message_file))
    
    return pairs


class TimitAttackDataset(data.Dataset):
    def __init__(self, manifest_path, trim_start, num_samples, n_pairs, message_file):
        random.seed(0)
        self.data_root = "/root/autodl-tmp/TIMIT"
        label_dict_path = "/root/autodl-tmp/TIMIT/processed/TIMIT_labels.npy"
        self.label_dict = np.load(label_dict_path, allow_pickle=True).item()
        self.spect_pairs = make_attack_pairs(manifest_path, n_pairs, message_file)
        # self.root = root
        self.loader = spect_loader
        self.trim_start = int(trim_start)
        self.num_samples = num_samples

    def __getitem__(self, index):
        carrier_file, msg_file = self.spect_pairs[index]

        carrier_spect, carrier_phase = self.loader(carrier_file, self.trim_start, True, self.num_samples)
        msg_spect, _ = self.loader(msg_file, self.trim_start, True, self.num_samples)
        msg_file_rel = os.path.relpath(msg_file, self.data_root)
        msg_id = self.label_dict[msg_file_rel]
        return carrier_spect, carrier_phase, msg_spect, msg_id

    def __len__(self):
        return len(self.spect_pairs)