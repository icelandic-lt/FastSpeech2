import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import re
from string import punctuation
from g2p_en import G2p

from fastspeech2 import FastSpeech2
from text import text_to_sequence, sequence_to_text
import hparams as hp
import utils
import audio as Audio
from g2p_is import load_g2p, translate as g2p


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess(text):
    text = text.rstrip(punctuation)
    phone = g2p(text)
    phone = list(filter(lambda p: p != ' ', phone))
    phone = '{' + '}{'.join(phone) + '}'
    phone = re.sub(r'\{[^\w\s]?\}', '{sp}', phone)
    phone = phone.replace('}{', ' ')

    print('|' + phone + '|')
    sequence = np.array(text_to_sequence(phone, hp.text_cleaners))
    sequence = np.stack([sequence])

    return torch.from_numpy(sequence).long().to(device)


def read_sentences(fname):
    with open(fname) as fin:
        return [line.strip() for line in fin]


def get_FastSpeech2(num, full_path=None):
    if full_path:
        checkpoint_path = full_path
    else:
        checkpoint_path = os.path.join(hp.checkpoint_path, "checkpoint_{}.pth.tar".format(num))
    model = nn.DataParallel(FastSpeech2())
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model'])
    model.requires_grad = False
    model.eval()
    return model


def synthesize(model, waveglow, melgan, text, sentence, prefix='', duration_control=1.0, pitch_control=1.0, energy_control=1.0, output_dir=None):
    src_len = torch.from_numpy(np.array([text.shape[1]])).to(device)

    mel, mel_postnet, log_duration_output, f0_output, energy_output, _, _, mel_len = model(
        text, src_len, d_control=duration_control, p_control=pitch_control, e_control=energy_control)

    mel_torch = mel.transpose(1, 2).detach()
    mel_postnet_torch = mel_postnet.transpose(1, 2).detach()
    mel = mel[0].cpu().transpose(0, 1).detach()
    mel_postnet = mel_postnet[0].cpu().transpose(0, 1).detach()
    f0_output = f0_output[0].detach().cpu().numpy()
    energy_output = energy_output[0].detach().cpu().numpy()

    if not output_dir:
        output_dir = hp.test_path

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    gl_fname = '{}_griffin_lim.wav'.format(prefix)
    Audio.tools.inv_mel_spec(mel_postnet, os.path.join(
        hp.test_path, gl_fname))

    vocoder_fname = '{}_{}.wav'.format(prefix, hp.vocoder)
    if waveglow is not None:
        utils.waveglow_infer(mel_postnet_torch, waveglow, os.path.join(
            output_dir, vocoder_fname))
    if melgan is not None:
        utils.melgan_infer(mel_postnet_torch, melgan, os.path.join(
            output_dir, vocoder_fname))

    #utils.plot_data([(mel_postnet.numpy(), f0_output, energy_output)], [
    #                'Synthesized Spectrogram'], filename=os.path.join(hp.test_path, '{}.png'.format(prefix)))


if __name__ == "__main__":
    # Test
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-fs', type=str)
    parser.add_argument('--model-melgan', type=str)
    parser.add_argument('--model-g2p', type=str)
    parser.add_argument('--step', type=int, default=30000)
    parser.add_argument('--duration_control', type=float, default=1.0)
    parser.add_argument('--pitch_control', type=float, default=1.0)
    parser.add_argument('--energy_control', type=float, default=1.0)
    parser.add_argument('--sentences', type=str)
    parser.add_argument('--output-dir', type=str)
    args = parser.parse_args()

    if not args.model_g2p:
        raise argparse.ArgumentTypeError("G2P model missing")
    
    sentences = read_sentences(args.sentences)
    model = get_FastSpeech2(args.step, full_path=args.model_fs).to(device)

    melgan = waveglow = None
    if hp.vocoder == 'melgan':
        melgan = utils.get_melgan(full_path=args.model_melgan)
        melgan.to(device)
    elif hp.vocoder == 'waveglow':
        waveglow = utils.get_waveglow()
        waveglow.to(device)
    g2p_model = load_g2p(args.model_g2p)

    with torch.no_grad():
        for i, sentence in enumerate(sentences):
            # text = preprocess(sentence, g2p_model)
            # synthesize(model, waveglow, melgan, text, sentence, 's{}-step{}'.format(
            #     i+1, args.step), args.duration_control, args.pitch_control, args.energy_control)
            try:
                sent_id, text, _ = sentence.split("\t")
            except ValueError:
                # Possibly no id and only a sentence
                sent_id, text = f"{i:0>5}", sentence
            text = preprocess(text)
            synthesize(model, waveglow, melgan, text, sentence, '{}'.format(
                sent_id), args.duration_control, args.pitch_control, args.energy_control, args.output_dir)

