import numpy as np
import os
import tgt
from scipy.io.wavfile import read
import pyworld as pw
import torch
import audio as Audio
from utils import get_alignment
import hparams as hp
import librosa


def prepare_align(in_dir):
    pass  # Already has tokens


def build_from_path(in_dir, out_dir):
    index = 1
    train = list()
    val = list()
    test = list()
    f0_max = energy_max = 0
    f0_min = energy_min = 1000000
    n_frames = 0
    text_index = 2

    lines = list()
    if hp.data_split:
        assert "train" in hp.data_split
        assert "val" in hp.data_split
        text_index = 1
        for key, value in hp.data_split.items():
            with open(value, encoding="utf-8") as f:
                for line in f:
                    lines.append((key, line))
    else:
        with open(os.path.join(in_dir, 'index.tsv'), encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                basename = parts[0]
                if int(basename[-2:]) < 92:
                    lines.append("train", line)
                if int(basename[-2:]) < 98:
                    lines.append("val", line)
                else:
                    lines.append("test", line)

    for set_name, line in lines:
        parts = line.strip().split('\t')
        basename = parts[0]
        text = parts[text_index]
            
        ret = process_utterance(in_dir, out_dir, basename)
        if ret is None:
            continue
        else:
            info, f_max, f_min, e_max, e_min, n = ret
        
        if set_name == "train":
            train.append(info)

            # Only gather stats for training data
            f0_max = max(f0_max, f_max)
            f0_min = min(f0_min, f_min)
            energy_max = max(energy_max, e_max)
            energy_min = min(energy_min, e_min)
            n_frames += n
        elif set_name == "val":
            val.append(info)
        elif set_name == "test":
            test.append(info)

        if index % 100 == 0:
            print("Done %d" % index, flush=True)
        index = index + 1
    
    with open(os.path.join(out_dir, 'stat.txt'), 'w', encoding='utf-8') as f:
        strs = ['Total time: {} hours'.format(n_frames*hp.hop_length/hp.sampling_rate/3600),
                'Total frames: {}'.format(n_frames),
                'Min F0: {}'.format(f0_min),
                'Max F0: {}'.format(f0_max),
                'Min energy: {}'.format(energy_min),
                'Max energy: {}'.format(energy_max)]
        for s in strs:
            print(s)
            f.write(s+'\n')
    
    return [r for r in train if r is not None], [r for r in val if r is not None], [r for r in test if r is not none]


def process_utterance(in_dir, out_dir, basename):
    wav_path = os.path.join(in_dir, 'audio', '{}.wav'.format(basename))
    tg_path = os.path.join(out_dir, 'TextGrid', '{}.TextGrid'.format(basename)) 
    
    # Get alignments
    try:
        textgrid = tgt.io.read_textgrid(tg_path)
    except FileNotFoundError:
        print("TextGrid not found", flush=True)
        return None

    phone, duration, start, end = get_alignment(textgrid.get_tier_by_name('phones'))
    text = '{'+ '}{'.join(phone) + '}' # '{A}{B}{$}{C}', $ represents silent phones
    text = text.replace('{$}', ' ')    # '{A}{B} {C}'
    text = text.replace('}{', ' ')     # '{A B} {C}'

    if start >= end:
        return None

    # Read and trim wav files
    sr, wav = read(wav_path)
    if len(wav.shape) > 1:
        wav = wav[:,0]
    if sr != hp.sampling_rate:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=hp.sampling_rate)
    wav = wav[int(hp.sampling_rate*start):int(hp.sampling_rate*end)].astype(np.float32)
    #y, sr = librosa.load(wav_path, sr=hp.sampling_rate)
    #wav = y[int(hp.sampling_rate*start):int(hp.sampling_rate*end)].astype(np.float32)

    # Compute fundamental frequency
    f0, _ = pw.dio(wav.astype(np.float64), hp.sampling_rate, frame_period=hp.hop_length/hp.sampling_rate*1000)
    f0 = f0[:sum(duration)]
    if len([f for f in f0 if f != 0]) < 1:
        return None
    
    try:
        # Compute mel-scale spectrogram and energy
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(torch.FloatTensor(wav))
        mel_spectrogram = mel_spectrogram.numpy().astype(np.float32)[:, :sum(duration)]
        energy = energy.numpy().astype(np.float32)[:sum(duration)]
        if mel_spectrogram.shape[1] >= hp.max_seq_len:
            return None
    except AssertionError:
        return None

    # Save alignment
    ali_filename = '{}-ali-{}.npy'.format(hp.dataset, basename)
    np.save(os.path.join(out_dir, 'alignment', ali_filename), duration, allow_pickle=False)

    # Save fundamental prequency
    f0_filename = '{}-f0-{}.npy'.format(hp.dataset, basename)
    np.save(os.path.join(out_dir, 'f0', f0_filename), f0, allow_pickle=False)

    # Save energy
    energy_filename = '{}-energy-{}.npy'.format(hp.dataset, basename)
    np.save(os.path.join(out_dir, 'energy', energy_filename), energy, allow_pickle=False)

    # Save spectrogram
    mel_filename = '{}-mel-{}.npy'.format(hp.dataset, basename)
    np.save(os.path.join(out_dir, 'mel', mel_filename), mel_spectrogram.T, allow_pickle=False)
    
    return '|'.join([basename, text]), max(f0), min([f for f in f0 if f != 0]), max(energy), min(energy), mel_spectrogram.shape[1]
