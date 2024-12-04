import random
import os
from pathlib import Path
import torch
import numpy as np
import pickle5 as pickle
import pandas as pd
import ast
import json

import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchaudio
from tqdm import tqdm
from transformers import AutoTokenizer
from config import podcast_wavlm_tokens, podcast_transcripts
from config import podcast_audio_folder, train_files, valid_files
from config import podcast_wavlm_feats#, podcast_roberta_feats
from config import podcast_wavlm_feats_6, podcast_wavlm_tokens_6
# from config import podcast_roberta_feats_all
from config import quantized_energy_folder, quantized_pitch_folder
from config import podcast_labels, podcast_text_labels
from config import podcast_roberta_feats_whisper
from config import podcast_roberta_feats_whisper_sup
from config import podcast_roberta_logits
from config import podcast_text_labels, podcast_pase_feats

class SpeechTextDataset(Dataset):
    def __init__(
        self,
        sample_rate: int = 16000,
        label_rate: int = 50,
        min_samples: int = 32000,
        max_samples: int = 80000,
        train: bool = True,
        tokenizer_name: str = "roberta-base",
        max_length: int = 1024,
        supervised: bool = True
    ):
        self.wavs_dir = podcast_audio_folder
        self.opensmile_folder = podcast_pase_feats
        self.wavlm_tokens = {}
        self.wavlm_tokens_6 = {}
        self.transcripts = {}
        with open(podcast_wavlm_tokens) as f:
            lines = f.readlines()
            for l in lines:
                d = ast.literal_eval(l)
                name, tokens = d["audio"], d["wavlm"]
                self.wavlm_tokens[name.split(os.sep)[-1]] = np.array(tokens.split(" ")).astype(int)
        with open(podcast_wavlm_tokens_6) as f:
            lines = f.readlines()
            for l in lines:
                d = ast.literal_eval(l)
                name, tokens = d["audio"], d["wavlm"]
                self.wavlm_tokens_6[name.split(os.sep)[-1]] = np.array(tokens.split(" ")).astype(int)
        self.transcripts = json.load(open(podcast_transcripts, "r"))
        names = list(self.wavlm_tokens.keys())
        for name in names:
            if name not in self.transcripts:
                continue
            if len(self.wavlm_tokens[name]) < 100:
                del self.transcripts[name]
                del self.wavlm_tokens[name]
                del self.wavlm_tokens_6[name]
        if train:
            self.files = list(pickle.load(open(train_files, "rb")))
            self.files = [x for x in self.files if x in self.transcripts]
        else:
            self.files = list(pickle.load(open(valid_files, "rb")))
            self.files = [x for x in self.files if x in self.transcripts]

        self.sample_rate = sample_rate
        self.label_rate = label_rate
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.train = train
        self.max_length = max_length
        self.wavlm_feats_folder = podcast_wavlm_feats
        self.wavlm_feats_folder_6 = podcast_wavlm_feats_6
        if not supervised:
            self.roberta_feats_folder = podcast_roberta_feats_whisper
        else:
            self.roberta_feats_folder = podcast_roberta_feats_whisper_sup
        self.roberta_logits_folder = podcast_roberta_logits
        # self.label_nums = {'C':0, 'O':9, 'A':0, 'N':1, 'D':0, 'H':2, 'S':3, 'F':4, 'X':8, 'U':5}
        self.label_nums = {0:0, 1:1, 2:2}
        # df = pd.read_csv(podcast_labels)
        # df = df[df["FileName"].isin (self.files)]
        # self.labels_dict_init = pd.Series(df.EmoClass.values,index=df.FileName).to_dict()
        self.labels_dict = json.load(open(podcast_text_labels, "r"))
        # self.labels_dict = {}
        # self.class_lists = {}
        # for k, v in self.labels_dict_init.items():
        #     if self.label_nums[v] <= 7:
        #         self.labels_dict[k] = self.label_nums[v]
        #         if self.label_nums[v] in self.class_lists:
        #             self.class_lists[self.label_nums[v]].append(k)
        #         else:
        #             self.class_lists[self.label_nums[v]] = [k]
        self.files = [x for x in self.files if x in self.labels_dict]
        self.files = [x for x in self.files if "MSP-PODCAST_2997_1007" not in x]
        # self.paired_dict = {}
        # self.unpaired_dict = {}

        # for i, f in enumerate(tqdm(self.files)):
        #     same_class = self.class_lists[self.labels_dict[f]]
        #     other_classes = [x for x in range(6) if x != self.labels_dict[f]]
        #     diff_class = random.choice(other_classes)
        #     diff_class = self.class_lists[diff_class]
        #     paired_aud = random.choice(same_class)
        #     unpaired_aud = random.choice(diff_class)
        #     self.paired_dict[f] = f
        #     self.unpaired_dict[f] = unpaired_aud

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        wav_name = self.files[index]
        wav_path = os.path.join(self.wavs_dir, wav_name)
        # units_path = self.units_dir / wav_path.relative_to(self.wavs_dir)
        # pitch_file = os.path.join(quantized_pitch_folder, wav_name.replace(".wav", ".npy"))
        # energy_file = os.path.join(quantized_energy_folder, wav_name.replace(".wav", ".npy"))
        wavlm_codes = self.wavlm_tokens[wav_name]
        # wavlm_codes_6 = self.wavlm_tokens_6[wav_name]
        # if os.path.exists(pitch_file) == False:
        #     pitch_contour = np.zeros(wavlm_codes.shape) + 3
        # else:
        #     pitch_contour = np.load(pitch_file)
        
        # if os.path.exists(energy_file) == False:
        #     energy_contour = np.zeros(wavlm_codes.shape) + 3
        # else:
        #     energy_contour = np.load(energy_file)
        wav, _ = torchaudio.load(wav_path)        
        # wav = F.pad(wav, ((400 - 320) // 2, (400 - 320) // 2))

        opensmile_feat = np.load(os.path.join(self.opensmile_folder, wav_name.replace(".wav", ".npy"))).T[::2, :]
        opensmile_feat_mean = np.mean(opensmile_feat, 0)
        opensmile_feat_std = np.std(opensmile_feat, 0) + 1.0
        opensmile_feat = (opensmile_feat-opensmile_feat_mean)/opensmile_feat_std

        # paired_wav_name = self.paired_dict[wav_name]
        # unpaired_wav_name = self.unpaired_dict[wav_name]
        # paired_roberta_feats = np.load(os.path.join(self.roberta_feats_all_folder, paired_wav_name.replace(".wav", "_text.npy")))

        roberta_feats = np.load(os.path.join(self.roberta_feats_folder, wav_name.replace(".wav", "_text.npy")))
        roberta_logits = np.load(os.path.join(self.roberta_logits_folder, wav_name.replace(".wav", "_text.npy")))
        # roberta_feats = torch.mean(roberta_feats, 1)
        # roberta_feats = roberta_feats[-1, :]
        # wavlm_feat = np.load(os.path.join(self.wavlm_feats_folder, wav_name.replace(".wav", ".npy")))
        # wavlm_feat_6 = np.load(os.path.join(self.wavlm_feats_folder_6, wav_name.replace(".wav", ".npy")))

        return wav, torch.tensor(roberta_feats), torch.tensor(roberta_logits), torch.from_numpy(opensmile_feat), torch.from_numpy(wavlm_codes).long()

    def collate(self, batch):
        wavs, roberta_feats, roberta_logits, opensmile_feats, wavlm_codes = zip(*batch)
        wavs = list(wavs)
        wavlm_codes = list(wavlm_codes)
        opensmile_feats = list(opensmile_feats)
        wav_lengths = [wav.size(-1) for wav in wavs]
        code_lengths = [code.size(-1) for code in wavlm_codes]
        max_wav_length = self.max_samples

        wav_frames = min(self.max_samples, max_wav_length)
        rate = self.label_rate / self.sample_rate
        code_frames = round(wav_frames * rate)

        collated_wavs, padding_masks = [], []
        collated_opensmile_feats = []

        for wav, code, os_feats in zip(wavs, wavlm_codes, opensmile_feats):
            if wav.size(-1) < wav_frames:
                wav_diff = -wav.size(-1) + wav_frames
                padded_wav = torch.zeros((1, wav_diff))
                wav = torch.cat((wav, padded_wav), dim=-1)
                collated_wavs.append(wav)
                code_diff = -code.size(-1) + code_frames
                padding_code = -100*torch.ones((code_diff))
                code = torch.cat((code, padding_code), dim=-1)
                os_feat_diff = -os_feats.size(0) + code_frames
                os_feats_padding = torch.zeros((os_feat_diff, 256))
                os_feats = torch.cat((os_feats, os_feats_padding), dim = 0)
                collated_opensmile_feats.append(os_feats)
                padding_mask = torch.zeros((1, code.size(-1)))
                padding_mask[:,-code_diff:] = 1
                padding_masks.append(padding_mask)
            else:
                wav_diff = wav.size(-1) - wav_frames
                wav_offset = random.randint(0, wav_diff)
                wav = wav[:, wav_offset : wav_offset + wav_frames]
                collated_wavs.append(wav)
                code_offset = round(wav_offset*rate)
                code = code[code_offset : code_offset + code_frames]
                os_feats = os_feats[code_offset:code_offset+code_frames, :]
                
                if code.size(-1) < code_frames:
                    code_diff = -code.size(-1) + code_frames
                    padding_mask = torch.zeros((1, code.size(-1)))
                    padding_mask[:,-code_diff:] = 1
                else:
                    padding_mask = torch.zeros((1, code.size(-1)))
                if os_feats.size(0) < code_frames:
                    os_feat_diff = -os_feats.size(0) + code_frames
                    os_feats_padding = torch.zeros((os_feat_diff, 256))
                    os_feats = torch.cat((os_feats, os_feats_padding), dim = 0)
                collated_opensmile_feats.append(os_feats)
                padding_masks.append(padding_mask)
                # print(pitch.shape)

        wavs = torch.stack(collated_wavs, dim=0)
        # padding_masks = torch.stack(padding_masks, dim=0)
        opensmile_feats = torch.stack(collated_opensmile_feats, dim=0)
        roberta_feats = torch.stack(roberta_feats, dim=0)
        roberta_logits = torch.stack(roberta_logits, dim=0)

        return wavs, opensmile_feats, roberta_feats, roberta_logits

# if __name__ == "__main__":
#     dataset = SpeechTextDataset(train=True)
#     train_loader = DataLoader(
#         dataset,
#         collate_fn=dataset.collate,
#         batch_size=32,
#         num_workers=1,
#         pin_memory=True,
#         shuffle=False,
#         drop_last=True,
#     )
#     for ind, batch in enumerate(tqdm(train_loader)):
#         wavs, opensmile_feats, roberta_feats, roberta_logits = batch
#         print(wavs.shape, opensmile_feats.shape)
#         print(roberta_feats.shape)