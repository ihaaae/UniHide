from os.path import join, basename

from loguru import logger
import soundfile as sf
import numpy as np

from hparams import *
from stft.stft import STFT
from dataset import spect_loader
from griffin_lim import griffin_lim


def SNR(pred:np.array, label:np.array):
     assert pred.shape == label.shape, "the shape of pred and label must be the same"
     pred, label = (pred+1)/2, (label+1)/2
     if len(pred.shape)>1:
         sigma_s_square = np.mean(label*label, axis=1)
         sigma_e_square = np.mean((pred-label)*(pred-label), axis=1)
         snr = 10*np.log10((sigma_s_square/max(sigma_e_square, 1e-9)))
         snr = snr.mean()
     else:
         sigma_s_square = np.mean(label*label)
         # print('sigma_s_square:', sigma_s_square)
         sigma_e_square = np.mean((pred-label)*(pred-label))
         # print('sigma_e_square:', sigma_e_square)
         # print(sigma_s_square/max(sigma_e_square, 1e-9))
         snr = 10*np.log10((sigma_s_square/max(sigma_e_square, 1e-9)))
     return snr

def convert(solver, carrier_wav_path: str, msg_wav_paths: str, trg_dir: str, epoch: int, trim_start: int, num_samples: int):
    if solver.mode != 'test':
        logger.warning("generating audio not in test mode!")

    _, sr = sf.read(carrier_wav_path)
    carrier_basename = basename(carrier_wav_path).split(".")[0]
    msg_basenames = [basename(msg_wav_path).split(".")[0] for msg_wav_path in msg_wav_paths]

    spect_carrier, phase_carrier = spect_loader(carrier_wav_path, trim_start, return_phase=True, num_samples=num_samples)
    spect_carrier, phase_carrier = spect_carrier.unsqueeze(0), phase_carrier.unsqueeze(0)
    assert spect_carrier.shape == (1, 1, 129, 378)
    assert phase_carrier.shape == (1, 1, 129, 378)
    magphase_msg = [spect_loader(path, trim_start, return_phase=True, num_samples=num_samples) for path in msg_wav_paths]
    spects_msg, phases_msg = [D[0].unsqueeze(0) for D in magphase_msg], [D[1].unsqueeze(0) for D in magphase_msg]
    assert spects_msg[0].shape == (1, 1, 129, 378)
    assert phases_msg[0].shape == (1, 1, 129, 378)
    assert len(spects_msg) == 1

    spect_carrier = spect_carrier.to('cuda')
    spects_msg = [spect_msg.to('cuda') for spect_msg in spects_msg]
    spect_carrier_reconst, spects_msg_reconst = solver.forward(spect_carrier, phase_carrier, spects_msg)
    spect_carrier_reconst = spect_carrier_reconst.cpu().squeeze(0)
    spects_msg_reconst = [spect_msg_reconst.cpu().squeeze(0) for spect_msg_reconst in spects_msg_reconst]
    assert spect_carrier_reconst.shape == (1, 129, 378)
    assert spects_msg_reconst[0].shape == (1, 129, 378)

    stft = STFT(N_FFT, HOP_LENGTH)
    out_carrier = stft.inverse(spect_carrier_reconst, phase_carrier.squeeze(0)).squeeze(0).squeeze(0).detach().numpy()
    orig_out_carrier = stft.inverse(spect_carrier.cpu().squeeze(0), phase_carrier.squeeze(0)).squeeze(0).squeeze(0).detach().numpy()
    assert len(out_carrier.shape) == 1
    assert len(orig_out_carrier.shape) == 1
    
    print()
    print(f'out_carriers dimension: {out_carrier.shape}, original out carriers dimension: {orig_out_carrier.shape}')
    print(f'snr of audio: {SNR(out_carrier, orig_out_carrier)}')
    print()

    outs_msg = [stft.inverse(spect_msg_reconst, phase_msg.squeeze(0)).squeeze(0).squeeze(0).detach().numpy() for spect_msg_reconst, phase_msg in zip(spects_msg_reconst, phases_msg)]
    orig_outs_msg = [stft.inverse(spect_msg.cpu().squeeze(0), phase_msg.squeeze(0)).squeeze(0).squeeze(0).detach().numpy() for spect_msg, phase_msg in zip(spects_msg, phases_msg)]
    outs_msg_gl = [griffin_lim(m.cpu(), n_iter=50)[0, 0].detach().numpy() for m in spects_msg_reconst]
    assert len(outs_msg[0].shape) == 1
    assert len(orig_outs_msg[0].shape) == 1
    assert len(outs_msg_gl[0].shape) == 1

    sf.write(join(trg_dir, f'{epoch}_{carrier_basename}_carrier_embedded.wav'), out_carrier, sr)
    sf.write(join(trg_dir, f'{epoch}_{carrier_basename}_carrier_orig.wav'), orig_out_carrier, sr)
    for i in range(len(outs_msg)):
        sf.write(join(trg_dir, f'{epoch}_{msg_basenames[i]}_msg_recovered_orig_phase.wav'), outs_msg[i], sr)
        sf.write(join(trg_dir, f'{epoch}_{msg_basenames[i]}_msg_orig.wav'), orig_outs_msg[i], sr)
        sf.write(join(trg_dir, f'{epoch}_{msg_basenames[i]}_msg_recovered_gl_phase.wav'), outs_msg_gl[i], sr)
