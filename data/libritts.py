import numpy as np
import os
import tgt
from tqdm import tqdm
import pyworld as pw
import torch
import audio as Audio
from utils import get_alignment
from text import _clean_text
import librosa
import hparams as hp
import random
from pathlib import Path


def get_speaker_table():
    '''
    see data.talromur.get_speaker_table
    '''
    spk_table = {}
    spk_id = 0

    spks = os.listdir(os.path.join(hp.preprocessed_path, 'TextGrid'))
    spks.sort()
    for spk in spks:
        spk_table[spk] = spk_id
        spk_id += 1
    inv_spk_table = {v:k for k, v in spk_table.items()}
    return spk_table, inv_spk_table


def prepare_align(in_dir):
    # already aligned
    pass


def build_from_path(in_dir, out_dir):
    train = list()
    val = list()
    f0_max = energy_max = 0
    f0_min = energy_min = 1000000
    n_frames = 0

    spkers = os.listdir(in_dir)
    spkers.sort()
    print("Total Speakers : {}".format(len(spkers)))

    if not os.path.exists(os.path.join(out_dir, "TextGrid")):
        raise FileNotFoundError("TextGrid not found in {}".format(out_dir))
    for spker in tqdm(spkers):
        spker_dir = os.path.join(in_dir, spker)
        file_paths = []
        for dirpath, dirnames, filenames in os.walk(spker_dir):
            for f in filenames:
                if f.endswith(".normalized.txt"):
                    subdir = Path(dirpath).relative_to(in_dir)
                    file_paths.append((subdir, f))

        for i, file_path in enumerate(file_paths):
            subdir = file_path[0]
            filename = file_path[1]
            basename = filename.replace(".normalized.txt", "")

            ret = process_utterance(in_dir, out_dir, subdir, basename)

            if ret is None:
                continue
            else:
                info, f_max, f_min, e_max, e_min, n = ret

            if i == 0:
                val.append(info)
            else:
                train.append(info)

            f0_max = max(f0_max, f_max)
            f0_min = min(f0_min, f_min)
            energy_max = max(energy_max, e_max)
            energy_min = min(energy_min, e_min)
            n_frames += n


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

    return [r for r in train if r is not None], [r for r in val if r is not None]


def process_utterance(in_dir, out_dir, dirname, basename):
    wav_path = os.path.join(in_dir , dirname   , '{}.wav'.format(basename))
    tg_path  = os.path.join(out_dir, 'TextGrid', dirname, '{}.TextGrid'.format(basename))

    if not os.path.exists(tg_path):
        return None

    textgrid = tgt.io.read_textgrid(tg_path)
    phone, duration, start, end = get_alignment(textgrid.get_tier_by_name('phones'))
    text = '{'+ '}{'.join(phone) + '}' # '{A}{B}{$}{C}', $ represents silent phones
    text = text.replace('{$}', ' ')    # '{A}{B} {C}'
    text = text.replace('}{', ' ')     # '{A B} {C}'

    if start >= end:
        return None

    wav, _ = librosa.load(wav_path, sr=hp.sampling_rate)
    wav = wav[int(hp.sampling_rate*start):int(hp.sampling_rate*end)].astype(np.float32)

    f0, _ = pw.dio(wav.astype(np.float64), hp.sampling_rate, frame_period=hp.hop_length/hp.sampling_rate*1000)
    f0 = f0[:sum(duration)]


    mel_spectrogram, energy = Audio.tools.get_mel_from_wav(torch.FloatTensor(wav))
    mel_spectrogram = mel_spectrogram.cpu().numpy().astype(np.float32)[:, :sum(duration)]
    energy = energy.cpu().numpy().astype(np.float32)[:sum(duration)]
    if mel_spectrogram.shape[1] >= hp.max_seq_len:
        return None


    try:
        assert(f0.shape[0] == energy.shape[0] == mel_spectrogram.shape[1])
    except AssertionError as e:
        print("duration problem: {}".format(wav_path))
        return None

    ali_filename = '{}-ali-{}.npy'.format(hp.dataset, basename)
    np.save(os.path.join(out_dir, 'alignment', ali_filename), duration, allow_pickle=False)

    f0_filename = '{}-f0-{}.npy'.format(hp.dataset, basename)
    np.save(os.path.join(out_dir, 'f0', f0_filename), f0, allow_pickle=False)

    energy_filename = '{}-energy-{}.npy'.format(hp.dataset, basename)
    np.save(os.path.join(out_dir, 'energy', energy_filename), energy, allow_pickle=False)

    mel_filename = '{}-mel-{}.npy'.format(hp.dataset, basename)
    np.save(os.path.join(out_dir, 'mel', mel_filename), mel_spectrogram.T, allow_pickle=False)
    try:
        return '|'.join([basename, text]), max(f0), min([f for f in f0 if f != 0]), max(energy), min(energy), mel_spectrogram.shape[1]
    except:
        return None