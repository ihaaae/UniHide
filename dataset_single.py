import random
from typing import List, Tuple, Union

import numpy as np
import soundfile
import torch
import torch.utils.data as data
from boltons import fileutils

import hparams
from stft.stft import STFT


def spect_loader(path:str, trim_start:int, return_phase=False, num_samples=16000, crop=True) -> Union[torch.Tensor, 
                                                                                                      Tuple[torch.Tensor, torch.Tensor]]:
    y, _ = soundfile.read(path)

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

def make_dataset(path: str, n_pairs: int) -> List[str]:
    wav_files = list(fileutils.iter_find_files(path, "*.wav"))

    sampled_files = random.sample(wav_files, n_pairs)
    assert len(sampled_files) == n_pairs

    return sampled_files

class TimitDataset(data.Dataset):
    def __init__(self, root,
                       n_pairs=100000,
                       transform=None,
                       trim_start=0,
                       num_samples=16000):
        random.seed(0)
        self.spect_pairs = make_dataset(root, n_pairs)
        self.transform = transform
        self.loader = spect_loader
        self.trim_start = int(trim_start)
        self.num_samples = num_samples

    def __getitem__(self, index) -> torch.Tensor:
        carrier_file = self.spect_pairs[index]

        carrier_spect = self.loader(carrier_file,
                                    self.trim_start,
                                    num_samples=self.num_samples)

        if self.transform is not None:
            carrier_spect = self.transform(carrier_spect)

        return carrier_spect

    def __len__(self):
        return len(self.spect_pairs)
