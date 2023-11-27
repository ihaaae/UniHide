import torch
from hparams import N_FFT, HOP_LENGTH
from stft.stft import STFT

def fake_batch(wav_data: torch.Tensor) -> torch.Tensor:
    assert wav_data.shape == (48000,)
    
    batch_data = []
    for i in range(32):
        batch_data.append(wav_data[i * 1440: i * 1440 + 3200])
    
    batch_tensor = torch.stack(batch_data)
    assert batch_tensor.shape == (32, 3200)
    
    return batch_tensor

def convert(spect: torch.Tensor, phase:torch.Tensor) -> torch.Tensor:
    assert spect.shape == (1, 129, 378)
    assert phase.shape == (1, 129, 378)
    stft = STFT(N_FFT, HOP_LENGTH)
    out = stft.inverse(spect, phase).squeeze(0).squeeze(0)
    assert out.shape == (48000,)
    
    return out