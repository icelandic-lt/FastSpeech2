import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import math
import os

import hparams
import audio as Audio
from utils import pad_1D, pad_2D, process_meta
from text import text_to_sequence, sequence_to_text

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Dataset(Dataset):
    def __init__(self, filename="train.txt", sort=True):
        self.basename, self.text = process_meta(
            os.path.join(hparams.preprocessed_path, filename))
        self.sort = sort

        if hparams.multi_speaker:
            if hparams.dataset.startswith('Talromur'):
                from data import talromur
                self.speaker_id_dirname = talromur.get_speaker_id_to_dirname()
                self.speaker_table, self.inv_speaker_table = talromur.get_speaker_table()
            elif hparams.dataset == 'LibriTTS':
                from data import libritts
                self.speaker_table, self.inv_speaker_table = libritts.get_speaker_table()


    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        phone = np.array(text_to_sequence(self.text[idx], []))
        preprocessed_path = hparams.preprocessed_path
        dataset = hparams.dataset
        if hparams.multi_speaker and hparams.dataset.startswith('Talromur'):
            # Instead of creating a new preprocessing directory with the data
            # from all Talromur collections, we just fetch the data from each
            # Talromur preprocess directory
            dataset = self.speaker_id_dirname[basename.split('-')[0]]
            preprocessed_path = os.path.join(
                os.path.dirname(preprocessed_path),
                dataset)

        mel_path = os.path.join(
            preprocessed_path, "mel", "{}-mel-{}.npy".format(dataset, basename))
        mel_target = np.load(mel_path)
        D_path = os.path.join(
            preprocessed_path, "alignment", "{}-ali-{}.npy".format(dataset, basename))
        D = np.load(D_path)
        f0_path = os.path.join(
            preprocessed_path, "f0", "{}-f0-{}.npy".format(dataset, basename))
        f0 = np.load(f0_path)
        energy_path = os.path.join(
            preprocessed_path, "energy", "{}-energy-{}.npy".format(dataset, basename))
        energy = np.load(energy_path)

        sample = {"id": basename,
                  "text": phone,
                  "mel_target": mel_target,
                  "D": D,
                  "f0": f0,
                  "energy": energy}

        return sample

    def reprocess(self, batch, cut_list):
        ids = [batch[ind]["id"] for ind in cut_list]
        if hparams.multi_speaker:
            if hparams.dataset.startswith('Talromur'):
                # fname format: <speaker_id>-<rec_id>.wav
                speaker_ids = [self.speaker_table[_id.split('-')[0]] for _id in ids]
            elif hparams.dataset == 'LibriTTS':
                speaker_ids = [self.speaker_table[_id.split('_')[0]] for _id in ids]

        texts = [batch[ind]["text"] for ind in cut_list]
        mel_targets = [batch[ind]["mel_target"] for ind in cut_list]
        Ds = [batch[ind]["D"] for ind in cut_list]
        f0s = [batch[ind]["f0"] for ind in cut_list]
        energies = [batch[ind]["energy"] for ind in cut_list]
        for text, D, id_ in zip(texts, Ds, ids):
            if len(text) != len(D):
                print(text, text.shape, D, D.shape, id_)
        length_text = np.array(list())
        for text in texts:
            length_text = np.append(length_text, text.shape[0])

        length_mel = np.array(list())
        for mel in mel_targets:
            length_mel = np.append(length_mel, mel.shape[0])

        texts = pad_1D(texts)
        Ds = pad_1D(Ds)
        mel_targets = pad_2D(mel_targets)
        f0s = pad_1D(f0s)
        energies = pad_1D(energies)
        log_Ds = np.log(Ds + hparams.log_offset)

        out = {"id": ids,
               "text": texts,
               "mel_target": mel_targets,
               "D": Ds,
               "log_D": log_Ds,
               "f0": f0s,
               "energy": energies,
               "src_len": length_text,
               "mel_len": length_mel}
        if hparams.multi_speaker:
            out.update({'spk_ids': speaker_ids})

        return out

    def collate_fn(self, batch):
        len_arr = np.array([d["text"].shape[0] for d in batch])
        index_arr = np.argsort(-len_arr)
        batchsize = len(batch)
        real_batchsize = int(math.sqrt(batchsize))

        cut_list = list()
        for i in range(real_batchsize):
            if self.sort:
                cut_list.append(
                    index_arr[i*real_batchsize:(i+1)*real_batchsize])
            else:
                cut_list.append(
                    np.arange(i*real_batchsize, (i+1)*real_batchsize))

        output = list()
        for i in range(real_batchsize):
            output.append(self.reprocess(batch, cut_list[i]))

        return output


if __name__ == "__main__":
    # Test
    dataset = Dataset('train.txt')
    training_loader = DataLoader(dataset, batch_size=10, shuffle=False, collate_fn=dataset.collate_fn,
                                 drop_last=True, num_workers=0)
    total_step = hparams.epochs * len(training_loader) * hparams.batch_size

    cnt = 0
    for i, batchs in enumerate(training_loader):
        for j, data_of_batch in enumerate(batchs):
            mel_target = torch.from_numpy(
                data_of_batch["mel_target"]).float().to(device)
            print(mel_target.shape)
            D = torch.from_numpy(data_of_batch["D"]).int().to(device)
            if mel_target.shape[1] == D.sum().item():
                cnt += 1
        if i == 0:
            break

    print(cnt, len(dataset))
