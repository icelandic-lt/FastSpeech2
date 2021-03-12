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

def preprocess(text, g2p_model):
    text = text.rstrip(punctuation)
    if hp.language == 'is':
        phone = g2p(text, g2p_model)
    elif hp.language == 'en':
        en_g2p = G2p()
        phone = en_g2p(text)
    phone = list(filter(lambda p: p != ' ', phone))
    phone = '{' + '}{'.join(phone) + '}'
    phone = re.sub(r'\{[^\w\s]?\}', '{sp}', phone)
    phone = phone.replace('}{', ' ')
    print('|' + phone + '|')
    sequence = np.array(text_to_sequence(phone, hp.text_cleaners))
    sequence = np.stack([sequence])

    return torch.from_numpy(sequence).long().to(device)


def get_FastSpeech2(num, full_path=None):
    if full_path:
        checkpoint_path = full_path
    else:
        checkpoint_path = os.path.join(hp.checkpoint_path, "checkpoint_{}.pth.tar".format(num))
    if hp.multi_speaker:
        model = nn.DataParallel(FastSpeech2(num_speakers=hp.num_speakers))
    else:
        model = nn.DataParallel(FastSpeech2())
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model'])
    model.requires_grad = False
    model.eval()
    return model


def synthesize(model, waveglow, melgan, text, sentence, prefix='',
        duration_control=1.0, pitch_control=1.0, energy_control=1.0,
        speaker_id=None, prosody_mel_path=None):

    sentence = sentence[:200]  # long filename will result in OS Error

    src_len = torch.from_numpy(np.array([text.shape[1]])).to(device)

    speaker_ids = None
    if speaker_id is not None:
        speaker_ids = torch.tensor(speaker_id).to(torch.int64).to(device)

    mel_prosody_target = None
    if prosody_mel_path is not None:
        '''
        prosody_mel_path is an ID of preprocessed mel spectrogram that is
        then fed into the prosody encoder.
        '''
        #mel_prosody_path = os.path.join(
        #    hp.preprocessed_path, "mel", "{}-mel-{}.npy".format(hp.dataset, prosody_mel_path))
        mel_prosody_target = np.load(prosody_mel_path)
        mel_prosody_target = torch.from_numpy(mel_prosody_target)
        mel_prosody_target = torch.unsqueeze(mel_prosody_target, dim=0)


    mel, mel_postnet, log_duration_output, f0_output, energy_output, _, _, mel_len = model(
        text, src_len, mel_prosody_target, d_control=duration_control, p_control=pitch_control, e_control=energy_control,
        speaker_ids=speaker_ids)

    mel_torch = mel.transpose(1, 2).detach()
    mel_postnet_torch = mel_postnet.transpose(1, 2).detach()
    mel = mel[0].cpu().transpose(0, 1).detach()
    mel_postnet = mel_postnet[0].cpu().transpose(0, 1).detach()
    f0_output = f0_output[0].detach().cpu().numpy()
    energy_output = energy_output[0].detach().cpu().numpy()

    np.save(f'{speaker_id}_mel_postnet.npy', mel_postnet)
    '''
    if not os.path.exists(hp.test_path):
        os.makedirs(hp.test_path)

    Audio.tools.inv_mel_spec(mel_postnet, os.path.join(
        hp.test_path, '{}_griffin_lim_{}.wav'.format(prefix, sentence)))

    sentence_id = sentence[:30]
    speaker_label = f'speaker#{speaker_id}' if speaker_id is not None else 'speaker'
    prosody_label = f'pros_src_{os.path.splitext(os.path.basename(prosody_mel_path))[0]}' if prosody_mel_path is not None else ''
    if waveglow is not None:
        utils.waveglow_infer(mel_postnet_torch, waveglow, os.path.join(
            hp.test_path, '{}_{}_{}_{}_{}.wav'.format(prefix, hp.vocoder, sentence_id,
            speaker_label, prosody_label)))
    if melgan is not None:
        utils.melgan_infer(mel_postnet_torch, melgan, os.path.join(
            hp.test_path, '{}_{}_{}_{}_{}.wav'.format(prefix, hp.vocoder, sentence_id,
            speaker_label, prosody_label)))

    #utils.plot_data([(mel_postnet.numpy(), f0_output, energy_output)], [
    #                'Synthesized Spectrogram'], filename=os.path.join(hp.test_path, '{}_{}.png'.format(prefix, sentence_id)))
    '''

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
    parser.add_argument('-sids', '--speaker_ids', nargs='+', default=None,
        type=lambda s: [int(item) for item in s.split(' ')],
        help='If using a multispeaker model, pass a list of valid speaker ids here')
    parser.add_argument('--prosody_mel_path', type=str, default=None)
    parser.add_argument('--sentence_path', type=str, default=None)
    args = parser.parse_args()

    if not args.model_g2p:
        raise argparse.ArgumentTypeError("G2P model missing")

    if args.sentence_path is None:
        sentences = ['Finally I can speak.']
    else:
        sentences = []
        with open(args.sentence_path, 'r') as f:
            for line in f: sentences.append(line.strip())

    model = get_FastSpeech2(args.step).to(device)
    melgan = waveglow = None
    if hp.vocoder == 'melgan':
        melgan = utils.get_melgan(full_path=args.model_melgan)
        melgan.to(device)
    elif hp.vocoder == 'waveglow':
        waveglow = utils.get_waveglow()
        waveglow.to(device)

    g2p_model = load_g2p(args.model_g2p)

    prosody_paths = []
    if args.prosody_mel_path is not None:
        if os.path.isdir(args.prosody_mel_path):
            for p in os.listdir(args.prosody_mel_path):
                prosody_paths.append(os.path.join(args.prosody_mel_path, p))
        else:
            prosody_paths.append(args.prosody_mel_path)
    else:
        prosody_paths = [None]
    for prosody_mel_path in prosody_paths:
        for i, sentence in enumerate(sentences):
            text = preprocess(sentence, g2p_model)
            if args.speaker_ids is None:
                synthesize(model, waveglow, melgan, text, sentence,
                    prefix='content-{}'.format(i+1), prosody_mel_path=prosody_mel_path)
            else:
                for speaker_id in args.speaker_ids:
                    synthesize(model, waveglow, melgan, text, sentence,
                        prefix='content-{}'.format(i+1), speaker_id=speaker_id,
                        prosody_mel_path=prosody_mel_path)

        with torch.no_grad():
            for i, sentence in enumerate(sentences):
                text = preprocess(sentence, g2p_model)
                if args.speaker_ids is None:
                    synthesize(model, waveglow, melgan, text, sentence, 'content-{}'.format(
                            i+1), args.duration_control, args.pitch_control, args.energy_control,
                            prosody_mel_path=prosody_mel_path)
                else:
                    for speaker_id in args.speaker_ids:
                        synthesize(model, waveglow, melgan, text, sentence, 'content-{}'.format(
                            i+1), args.duration_control, args.pitch_control, args.energy_control,
                            speaker_id=speaker_id, prosody_mel_path=prosody_mel_path)

