import random
from typing import Tuple, Union, List

from boltons import fileutils
import numpy as np
import torch
import torch.utils.data as data
import soundfile

import hparams
from stft.stft import STFT

def spect_loader(path:str, trim_start:int, return_phase=False, num_samples=16000, crop=True) -> Union[torch.Tensor, 
                                                                                                      Tuple[torch.Tensor, torch.Tensor]]:
    y, sr = soundfile.read(path)

    if crop:
        y = y[trim_start: trim_start + num_samples]  # trim 'trim_start' from start and crop 1 sec
        y = np.hstack((y, np.zeros((num_samples - len(y)))))

        assert y.shape == (num_samples,)

    stft = STFT(hparams.N_FFT, hparams.HOP_LENGTH)
    y = torch.FloatTensor(y).unsqueeze(0)
    assert y.shape == (1, num_samples)
    spect, phase = stft.transform(y)

    if return_phase:
        return spect, phase
    
    return spect


class BaseDataset(data.Dataset):
    def __init__(self, root,
                       n_pairs=100000,
                       transform=None,
                       trim_start=0,
                       num_samples=16000,
                       test=False):
        random.seed(0)
        self.spect_pairs = self.make_pairs_dataset(root, n_pairs)
        self.root = root
        self.transform = transform
        self.loader = spect_loader
        self.trim_start = int(trim_start)
        self.num_samples = num_samples
        self.test = test

    def __getitem__(self, index) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
                                          Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:

        carrier_file, msg_file = self.spect_pairs[index]

        carrier_spect, carrier_phase = self.loader(carrier_file,
                                                   self.trim_start,
                                                   return_phase=True,
                                                   num_samples=self.num_samples)
        
        msg_spect, msg_phase = self.loader(msg_file,
                                           self.trim_start,
                                           return_phase=True,
                                           num_samples=self.num_samples)

        if self.transform is not None:
            carrier_spect = self.transform(carrier_spect)
            carrier_phase= self.transform(carrier_phase)
            msg_spect = self.transform(msg_spect)

        if self.test:
            return carrier_spect, carrier_phase, msg_spect, msg_phase
        else:
            return carrier_spect, carrier_phase, msg_spect

    def __len__(self):
        return len(self.spect_pairs)


class TimitDataset(BaseDataset):
    def __init__(self, root,
                       n_pairs=100000,
                       transform=None,
                       trim_start=0,
                       num_samples=16000,
                       test=False):
        super(TimitDataset, self).__init__(root,
                                           n_pairs,
                                           transform,
                                           trim_start,
                                           num_samples,
                                           test)

    def make_pairs_dataset(self, path: str, n_pairs: int) -> List[Tuple[str, str]]:
        pairs = []
        wav_files = list(fileutils.iter_find_files(path, "*.wav"))

        for i in range(n_pairs):
            sampled_files = random.sample(wav_files, 2)
            carrier_file, hidden_message_file = sampled_files
            pairs.append((carrier_file, hidden_message_file))
        return pairs

