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


import torch
import numpy as np

from srs_model import SincClassifier

def preprocess(wav_data):
    norm_factor = np.abs(wav_data).max()
    wav_data = wav_data/norm_factor
    return wav_data, norm_factor

timit_path = "/root/HideAndSpeak-master/SRS/SincNet_TIMIT/model_raw.pkl"

def get_pretrained_models():
    CNN_arch = {'cnn_input_dim': 3200, 
                'cnn_N_filt': [80, 60, 60], 
                'cnn_len_filt': [251, 5, 5], 
                'cnn_max_pool_len': [3, 3, 3], 
                'cnn_use_laynorm_inp': True, 
                'cnn_use_batchnorm_inp': False, 
                'cnn_use_laynorm': [True, True, True], 
                'cnn_use_batchnorm': [False, False, False], 
                'cnn_act': ['leaky_relu', 'leaky_relu', 'leaky_relu'], 
                'cnn_drop': [0.0, 0.0, 0.0], 
                'fs': 16000}

    DNN_arch = {'fc_input_dim': 6420, 
                'fc_lay': [2048, 2048, 2048], 
                'fc_drop': [0.0, 0.0, 0.0], 
                'fc_use_batchnorm': [True, True, True], 
                'fc_use_laynorm': [False, False, False], 
                'fc_use_laynorm_inp': True, 
                'fc_use_batchnorm_inp': False, 
                'fc_act': ['leaky_relu', 'leaky_relu', 'leaky_relu']}

    Timit_Classifier = {'fc_input_dim': 2048, 
                      'fc_lay': [462], 
                      'fc_drop': [0.0], 
                      'fc_use_batchnorm': [False], 
                      'fc_use_laynorm': [False], 
                      'fc_use_laynorm_inp': False, 
                      'fc_use_batchnorm_inp': False, 
                      'fc_act': ['none']}
    
    Timit_model = SincClassifier(CNN_arch, DNN_arch, Timit_Classifier)
    print("load model from:" + timit_path)
    timit_checkpoint_load = torch.load(timit_path)
    Timit_model.load_raw_state_dict(timit_checkpoint_load)
    Timit_model = Timit_model.cuda().eval()
    # freeze the model
    for p in Timit_model.parameters():
        p.requires_grad = False
        
    return Timit_model