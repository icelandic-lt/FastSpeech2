import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import argparse
import os
from dataset import Dataset
import hparams as hp
import utils
import audio as Audio
from synthesize import get_FastSpeech2


def _write_index_line(fout, model, group, fname, text, fid):
    fout.write("\t".join([model, group, fname, text, str(fid)]) + "\n")


def main(args):
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get dataset
    dataset = Dataset("test.txt")
    loader = DataLoader(dataset, batch_size=hp.batch_size**2, shuffle=False,
                        collate_fn=dataset.collate_fn, drop_last=False, num_workers=0)

    model = get_FastSpeech2(args.step, full_path=args.model_fs).to(device)

    # Load vocoder
    if hp.vocoder == 'melgan':
        melgan = utils.get_melgan(full_path=args.model_melgan)
    elif hp.vocoder == 'waveglow':
        waveglow = utils.get_waveglow()

    # Init logger
    log_path = hp.log_path
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        os.makedirs(os.path.join(log_path, 'test'))
    test_logger = SummaryWriter(os.path.join(log_path, 'test'))

    # Init synthesis directory
    test_path = hp.test_path
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    current_step = args.step
    findex = open(os.path.join(test_path, "index.tsv"), "w")

    # Testing
    print("Generate test audio")
    prefix = ""
    for i, batchs in enumerate(loader):
        for j, data_of_batch in enumerate(batchs):
            print("Start batch", j)
            fids = data_of_batch["id"]

            # Get Data
            text = torch.from_numpy(
                data_of_batch["text"]).long().to(device)
            mel_target = torch.from_numpy(
                data_of_batch["mel_target"]).float().to(device)
            D = torch.from_numpy(data_of_batch["D"]).long().to(device)
            log_D = torch.from_numpy(
                data_of_batch["log_D"]).float().to(device)
            f0 = torch.from_numpy(data_of_batch["f0"]).float().to(device)
            energy = torch.from_numpy(
                data_of_batch["energy"]).float().to(device)
            src_len = torch.from_numpy(
                data_of_batch["src_len"]).long().to(device)
            mel_len = torch.from_numpy(
                data_of_batch["mel_len"]).long().to(device)
            max_src_len = np.max(data_of_batch["src_len"]).astype(np.int32)
            max_mel_len = np.max(data_of_batch["mel_len"]).astype(np.int32)

            mel_output, mel_postnet_output, log_duration_output, f0_output, energy_output, src_mask, mel_mask, _ = model(text, src_len)

            for i in range(len(mel_len)):
                fid = fids[i]
                print("Generate audio for:", fid)
                length = mel_len[i].item()
                
                _mel_target_torch = mel_target[i, :length].detach(
                    ).unsqueeze(0).transpose(1, 2)
                
                _mel_target = mel_target[i, :length].detach(
                    ).cpu().transpose(0, 1)
                
                _mel_torch = mel_output[i, :length].detach(
                    ).unsqueeze(0).transpose(1, 2)
                
                _mel = mel_output[i, :length].detach().cpu().transpose(0, 1)
                
                _mel_postnet_torch = mel_postnet_output[i, :length].detach(
                    ).unsqueeze(0).transpose(1, 2)
                
                _mel_postnet = mel_postnet_output[i, :length].detach(
                    ).cpu().transpose(0, 1)

                fname = "{}{}_step_{}_gt_griffin_lim.wav".format(prefix, fid, current_step)
                Audio.tools.inv_mel_spec(_mel_target, os.path.join(hp.test_path, fname))
                _write_index_line(findex, "Griffin Lim", "vocoder", fname, "", fid)

                fname = "{}{}_step_{}_griffin_lim.wav".format(prefix, fid, current_step)
                Audio.tools.inv_mel_spec(_mel, os.path.join(hp.test_path, fname))
                _write_index_line(findex, "FastSpeech2 + GL", "tts", fname, "", fid)

                fname = "{}{}_step_{}_postnet_griffin_lim.wav".format(prefix, fid, current_step)
                Audio.tools.inv_mel_spec(_mel_postnet, os.path.join(hp.test_path, fname))
                _write_index_line(findex, "FastSpeech2 + PN + GL", "tts", fname, "", fid)

                if hp.vocoder == 'melgan':
                    fname = '{}{}_step_{}_ground-truth_{}.wav'.format(prefix, fid, current_step, hp.vocoder)
                    utils.melgan_infer(_mel_target_torch, melgan, os.path.join(
                        hp.test_path, fname))
                    _write_index_line(findex, "Melgan", "vocoder", fname, "", fid)

                    fname = '{}{}_step_{}_{}.wav'.format(prefix, fid, current_step, hp.vocoder)
                    utils.melgan_infer(_mel_torch, melgan, os.path.join(hp.test_path, fname))
                    _write_index_line(findex, "FastSpeech2 + Melgan", "tts", fname, "", fid)

                    fname = '{}{}_step_{}_postnet_{}.wav'.format(prefix, fid, current_step, hp.vocoder)
                    utils.melgan_infer(_mel_postnet_torch, melgan, os.path.join(
                        hp.test_path, fname))
                    _write_index_line(findex, "FastSpeech2 + PN + Melgan", "tts", fname, "", fid)

                elif hp.vocoder == 'waveglow':
                    utils.waveglow_infer(_mel_torch, waveglow, os.path.join(
                        hp.test_path, 'step_{}_{}.wav'.format(current_step, hp.vocoder)))
                    utils.waveglow_infer(_mel_postnet_torch, waveglow, os.path.join(
                        hp.test_path, 'step_{}_postnet_{}.wav'.format(current_step, hp.vocoder)))
                    utils.waveglow_infer(_mel_target_torch, waveglow, os.path.join(
                        hp.test_path, 'step_{}_ground-truth_{}.wav'.format(current_step, hp.vocoder)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, default=0)
    parser.add_argument('--model-fs', type=str)
    parser.add_argument('--model-melgan', type=str)

    args = parser.parse_args()

    main(args)
    print("Done")
