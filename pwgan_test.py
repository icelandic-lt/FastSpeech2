'''
Testing the Parallel WaveGan Vocoder. To try it out, generate mel spectrogram
features using the feature network first.
'''

import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from scipy.io import wavfile
import h5py
import librosa

from parallel_wavegan.utils import download_pretrained_model
from parallel_wavegan.utils import load_model

h5fname = 'stats.h5'
h5_file = h5py.File(h5fname,'r')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def interpolate_vocoder_input(scale_factor, spec):
    """Interpolate spectrogram by the scale factor.
    It is mainly used to match the sampling rates of
    the tts and vocoder models.
    Args:
        scale_factor (float): scale factor to interpolate the spectrogram
        spec (np.array): spectrogram to be interpolated
    Returns:
        torch.tensor: interpolated spectrogram.
    """
    print(" > before interpolation :", spec.shape)
    spec = torch.tensor(spec).unsqueeze(0).unsqueeze(0)  # pylint: disable=not-callable
    spec = torch.nn.functional.interpolate(spec,
                                           scale_factor=scale_factor,
                                           recompute_scale_factor=True,
                                           mode='bilinear',
                                           align_corners=False).squeeze(0)
    print(" > after interpolation :", spec.shape)
    return spec


vocoder_tag = 'libritts_parallel_wavegan.v1.long'
vocoder = load_model(download_pretrained_model(vocoder_tag)).to(device).eval()
vocoder.remove_weight_norm()
for i in range(330001,330033):
    mel_fname = f'./mels/step_{i}_mel.npy'
    sample = np.load(mel_fname)
    mel_tn = np.log10(np.exp(sample))
    mu = np.array(h5_file['mean'][:])
    var = np.array(h5_file['scale'][:])
    sigma = np.sqrt(var)
    dl10n = ((mel_tn - mu.reshape(1, -1, 1)) / sigma.reshape(1, -1, 1)).squeeze(0)

    torch_sample = torch.from_numpy(dl10n.T).to(device)
    #print(torch_sample.shape)
    #torch_sample = interpolate_vocoder_input([1, 24000/22050], torch_sample)
    #torch_sample = torch_sample.unsqueeze(0)
    #print(torch_sample.shape)
    with torch.no_grad():
        wav = vocoder.inference(torch_sample)
    wav = wav.view(-1).cpu().numpy()
    plt.plot(wav)
    plt.savefig('test.png')
    wavfile.write(f'./mel_out/test_{i}.wav', 24000, wav)