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
from g2p_is import translate as g2p

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess(text):
    text = text.rstrip(punctuation)
    phone = g2p(text)
    phone = list(filter(lambda p: p != ' ', phone))
    phone = '{'+ '}{'.join(phone) + '}'
    phone = re.sub(r'\{[^\w\s]?\}', '{sp}', phone)
    phone = phone.replace('}{', ' ')
    
    print('|' + phone + '|')    
    sequence = np.array(text_to_sequence(phone, hp.text_cleaners))
    sequence = np.stack([sequence])

    return torch.from_numpy(sequence).long().to(device)

def get_FastSpeech2(num):
    checkpoint_path = os.path.join(hp.checkpoint_path, "checkpoint_{}.pth.tar".format(num))
    model = nn.DataParallel(FastSpeech2())
    model.load_state_dict(torch.load(checkpoint_path)['model'])
    model.requires_grad = False
    model.eval()
    return model

def synthesize(model, waveglow, melgan, text, sentence, prefix=''):
    sentence = sentence[:200] # long filename will result in OS Error
    
    src_len = torch.from_numpy(np.array([text.shape[1]])).to(device)
        
    mel, mel_postnet, log_duration_output, f0_output, energy_output, _, _, mel_len = model(text, src_len)
    
    mel_torch = mel.transpose(1, 2).detach()
    mel_postnet_torch = mel_postnet.transpose(1, 2).detach()
    mel = mel[0].cpu().transpose(0, 1).detach()
    mel_postnet = mel_postnet[0].cpu().transpose(0, 1).detach()
    f0_output = f0_output[0].detach().cpu().numpy()
    energy_output = energy_output[0].detach().cpu().numpy()

    if not os.path.exists(hp.test_path):
        os.makedirs(hp.test_path)

    sentence_id = sentence[:30]
    Audio.tools.inv_mel_spec(mel_postnet, os.path.join(hp.test_path, '{}_griffin_lim_{}.wav'.format(prefix, sentence_id)))
    if waveglow is not None:
        utils.waveglow_infer(mel_postnet_torch, waveglow, os.path.join(hp.test_path, '{}_{}_{}.wav'.format(prefix, hp.vocoder, sentence_id)))
    if melgan is not None:
        utils.melgan_infer(mel_postnet_torch, melgan, os.path.join(hp.test_path, '{}_{}_{}.wav'.format(prefix, hp.vocoder, sentence_id)))
    
    utils.plot_data([(mel_postnet.numpy(), f0_output, energy_output)], ['Synthesized Spectrogram'], filename=os.path.join(hp.test_path, '{}_{}.png'.format(prefix, sentence_id)))


if __name__ == "__main__":
    # Test
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, default=30000)
    args = parser.parse_args()
    
    sentences = [
        "Advanced text to speech models such as Fast Speech can synthesize speech significantly faster than previous auto regressive models with comparable quality. The training of Fast Speech model relies on an auto regressive teacher model for duration prediction and knowledge distillation, which can ease the one to many mapping problem in T T S. However, Fast Speech has several disadvantages, 1, the teacher student distillation pipeline is complicated, 2, the duration extracted from the teacher model is not accurate enough, and the target mel spectrograms distilled from teacher model suffer from information loss due to data simplification, both of which limit the voice quality.",
        "Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition",
        "in being comparatively modern.",
        "For although the Chinese took impressions from wood blocks engraved in relief for centuries before the woodcutters of the Netherlands, by a similar process",
        "produced the block books, which were the immediate predecessors of the true printed book,",
        "the invention of movable metal letters in the middle of the fifteenth century may justly be considered as the invention of the art of printing.",
        "And it is worth mention in passing that, as an example of fine typography,",
        "the earliest book printed with movable types, the Gutenberg, or \"forty-two line Bible\" of about 1455,",
        "has never been surpassed.",
        "Printing, then, for our purpose, may be considered as the art of making books by means of movable types.",
        "Now, as all books not primarily intended as picture-books consist principally of types composed to form letterpress,"
        ]
    sentences = [
        "góðan dag ég kann að tala íslensku alveg hnökralaust eða svona næstum því",
        "hlýnun lengir líf fellibylja yfir landi",
        'Eg skal segja þér, kvað hann, hvað eg hefi hugsað. eg ætla að hafa til þings með mér kistur þær tvær, er Aðalsteinn konungur gaf mér, er hvortveggja er full af ensku silfri. Ætla eg að láta bera kisturnar til Lögbergs, þá er þar er fjölmennast; síðan ætla eg að sá silfrinu, og þykir mér undarlegt, ef allir skipta vel sín í milli; ætla eg, að þar myndi vera þá hrundningar, eða pústrar, eða bærist að um síðir, að allur þingheimurinn berðist.',
        "Vigdís Finnbogadóttir var fjórði forseti Íslands og gegndi hún embættinu frá 1980 til 1996. Hún var fyrsta konan i heiminum sem kosin var í lýðræðislegum kosningum til að gegna hlutverki þjóðhöfðingja.",
        "Í gær kvisaðist það út í Ósló að óvenjulíflegt væri á öldurhúsum nágrannasveitarfélaganna Asker og Bærum af þriðjudegi að vera auk þess sem norska ríkisútvarpið NRK greindi frá því að langar biðraðir væru fyrir utan líkamsræktarstöðvar bæja þessara, nokkuð sem íbúar þar höfðu sjaldan upplifað.",
        "Japanski bílsmiðurinn Honda hlaut í gær leyfi yfirvalda til að selja bíl með þriðja stigs sjálfaksturs tækni.",
        "Rómeó og Júlía er saga af sannri ást en um leið ástsýki og ungæðishætti. Í forgrunni verður mögnuð barátta ungrar konu gegn yfirþyrmandi feðraveldi. Fegurstu sögurnar geta sprottið upp úr hræðilegustu aðstæðunum.",
        "Diogo Jota, leikmaður Liverpool, er mjög ósáttur með EA Sports og sú einkunn sem þeir hafa gefið honum í FIFA 21 leiknum. EA Sports uppfærði ekki tölfræðina hans á þessu tímabili.",
    ]

    model = get_FastSpeech2(args.step).to(device)
    melgan = waveglow = None
    if hp.vocoder == 'melgan':
        melgan = utils.get_melgan()
        melgan.to(device)
    elif hp.vocoder == 'waveglow':
        waveglow = utils.get_waveglow()
        waveglow.to(device)
    
    for sentence in sentences:
        text = preprocess(sentence)
        synthesize(model, waveglow, melgan, text, sentence, prefix='step_{}'.format(args.step))
