import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import json
from torch.optim import Adam, SGD
from sklearn.metrics import f1_score
import argparse
import librosa
from model_dual import SpeechTextModel
from transformers import WavLMModel, RobertaModel, WavLMConfig

#Logger set
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser(description="Fine tune")
parser.add_argument(
    "--seed",
    metavar="seed",
    type=int,
    default=0
)
parser.add_argument(
        "--flag",
        metavar="flag",
        default="train",
        type=str,
    )

parser.add_argument(
        "--dataset",
        metavar="dataset",
        type=str,
    )

parser.add_argument(
        "--alpha",
        metavar="alpha",
        type=float,
    )
parser.add_argument(
        "--model",
        metavar="model",
        type=str,
    )

args = parser.parse_args()

SEED = args.seed
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

#CUDA devices enabled
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.cuda.empty_cache()

class SpeechDataset(Dataset):
    def __init__(
        self,
        folder,
        labels_file
    ):
        self.transcripts = {}
        self.folder = folder
        self.labels = labels_file
        wav_files = list(self.labels.keys())
        wav_files_folder = os.listdir(folder)
        wav_files = [x for x in wav_files if ".wav" in x]
        wav_files = [x for x in wav_files if x in wav_files_folder]
        self.wav_files = wav_files
        self.sr = 16000
        self.duration = 5000

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, index):
        wav_name = self.wav_files[index]
        label = self.labels[wav_name]
        

        audio_file = os.path.join(self.folder, wav_name)      
        (sig, sr) = librosa.load(audio_file, sr=None)
        if sr != self.sr:
            sig = librosa.resample(sig, sr, self.sr)
        final_sig = sig

        aud = (sig, sr)
        reaud = (sig, self.sr)
        resig = sig
        sig_len = resig.shape[0]
        max_len = self.sr//1000 * self.duration
        if len(resig.shape) == 2:
            resig = np.mean(resig, axis = 1)
        if (sig_len > 320000):
            # Truncate the signal to the given length
            start = np.random.randint(0, sig_len-max_len)

            final_sig = resig[start:start+max_len]
        # else:
        #     final_sig = resig

        if (sig_len < 8000):
            pad_end_len = max_len - sig_len

            # # Pad with 0s
            # pad_begin = np.zeros((pad_begin_len))
            pad_end = np.zeros((pad_end_len))

            final_sig = np.float32(np.concatenate((resig, pad_end), 0))
            final_aud = (final_sig, self.sr)
        # # else:
        # #     final_sig = resig
        
        return final_sig, wav_name

    def collate(self, batch):
        final_sig, label = zip(*batch)
        final_sig = torch.tensor(np.array(final_sig))
        label = list(label)

        return final_sig, label[0]
    
class SpeechClassifier(nn.Module):
    def __init__(self, emo_model, use_ce, num_classes):
        super().__init__()
        self.use_ce = use_ce
        if use_ce == "True":
            self.fc_pred = nn.Linear(768, num_classes)
        self.fc_valence = nn.Linear(768, 3)
        self.emo_model = emo_model

    def forward(self, speech):
        all_feats, pooled_speech = self.emo_model(speech)
        # all_feats = torch.mean(all_feats, 2)
        if self.use_ce == "True":
            speech_pred = self.fc_pred(pooled_speech)
        else:
            speech_pred = None
        speech_pred_valence = self.fc_valence(pooled_speech)

        return all_feats, speech_pred_valence, speech_pred, pooled_speech


class EmotionClassifier(nn.Module):
    def __init__(self, joint_model, output_dim):
        super().__init__()
        self.joint_model = joint_model
        self.fc_text = nn.Linear(768*2, 768)
        self.act = nn.ReLU()
        self.fc_text_final = nn.Linear(768, output_dim)

    def forward(self, audio):
        x, pooled_audio, x_aud = self.joint_model.extract_audio_features(audio.to(device))
        # comb = torch.mean(x, 0).squeeze(0)
        # comb = torch.mean(comb, 1).squeeze(1)
        # pooled_text = torch.cat((comb, pooled_text), -1)
        # x_text = self.act(self.fc_text(pooled_text))
        # pred_text = self.fc_text_final(x_text)

        return x, pooled_audio, x_aud

def create_MELD_dataset(mode, bs=8):
    if mode == 'train':
        folder = "/home/soumyad/TAFFC/MELD_data/MELD_wav_train"
        transcript_file = "/home/soumyad/MMER/MELD/train_transcripts.json"
        f = open("/home/soumyad/TAFFC/MELD_data/train_wav_labels.json")
        labels = json.load(f)
        f.close()
    elif mode == 'val':
        folder = "/home/soumyad/TAFFC/MELD_data/MELD_wav_val"
        transcript_file = "/home/soumyad/MMER/MELD/val_transcripts.json"
        f = open("/home/soumyad/TAFFC/MELD_data/val_wav_labels.json")
        labels = json.load(f)
        f.close()
    else:
        folder = "/home/soumyad/TAFFC/MELD_data/MELD_wav_test"
        transcript_file = "/home/soumyad/MMER/MELD/test_transcripts.json"
        f = open("/home/soumyad/TAFFC/MELD_data/test_wav_labels.json")
        labels = json.load(f)
        f.close()
    dataset = SpeechDataset(folder, labels)
    loader = DataLoader(dataset,
                    batch_size=bs,
                    pin_memory=False,
                    shuffle=True,
                    drop_last=False,
                    collate_fn=dataset.collate)
    return loader
    
def create_MOSI_dataset(mode, bs=8):
    if mode == 'train':
        folder = "/home/soumyad/TAFFC/MOSI_data/train_wavs"
        transcript_file = "/home/soumyad/MMER/MOSI/train_transcripts.json"
        f = open("/home/soumyad/TAFFC/MOSI_data/train_wav_labels.json")
        labels = json.load(f)
        f.close()
    elif mode == 'val':
        folder = "/home/soumyad/TAFFC/MOSI_data/val_wavs"
        transcript_file = "/home/soumyad/MMER/MOSI/val_transcripts.json"
        f = open("/home/soumyad/TAFFC/MOSI_data/val_wav_labels.json")
        labels = json.load(f)
        f.close()
    else:
        folder = "/home/soumyad/TAFFC/MOSI_data/test_wavs"
        transcript_file = "/home/soumyad/MMER/MOSI/test_transcripts.json"
        f = open("/home/soumyad/TAFFC/MOSI_data/test_wav_labels.json")
        labels = json.load(f)
        f.close()
    dataset = SpeechDataset(folder, labels)
    loader = DataLoader(dataset,
                    batch_size=bs,
                    pin_memory=False,
                    shuffle=True,
                    drop_last=False,
                    collate_fn=dataset.collate)
    return loader

def create_IEMOCAP4_dataset(mode, bs=8):
    if mode == 'train':
        folder = "/data2/soumyad/IEMOCAP_data/wavs"
        transcript_file = "/home/soumyad/MMER/IEMOCAP/IEMOCAP4_transcripts.json"
        f = open("/data2/soumyad/IEMOCAP_data/train_labels.json")
        labels = json.load(f)
        f.close()
    elif mode == 'val':
        folder = "/data2/soumyad/IEMOCAP_data/wavs"
        transcript_file = "/home/soumyad/MMER/IEMOCAP/IEMOCAP4_transcripts.json"
        f = open("/data2/soumyad/IEMOCAP_data/val_labels.json")
        labels = json.load(f)
        f.close()
    else:
        folder = "/data2/soumyad/IEMOCAP_data/wavs"
        transcript_file = "/home/soumyad/MMER/IEMOCAP/IEMOCAP4_transcripts.json"
        f = open("/data2/soumyad/IEMOCAP_data/test_wav_labels.json")
        labels = json.load(f)
        f.close()
    dataset = SpeechDataset(folder, labels)
    loader = DataLoader(dataset,
                    batch_size=bs,
                    pin_memory=False,
                    shuffle=True,
                    drop_last=False,
                    collate_fn=dataset.collate)
    return loader

def create_IEMOCAP6_dataset(mode, bs=8):
    if mode == 'train':
        folder = "/home/soumyad/TAFFC/IEMOCAP_data/wavs"
        transcript_file = "/home/soumyad/MMER/IEMOCAP/IEMOCAP6_transcripts.json"
        f = open("/home/soumyad/TAFFC/IEMOCAP6_data/train_labels.json")
        labels = json.load(f)
        f.close()
    elif mode == 'val':
        folder = "/home/soumyad/TAFFC/IEMOCAP_data/wavs"
        transcript_file = "/home/soumyad/MMER/IEMOCAP/IEMOCAP6_transcripts.json"
        f = open("/home/soumyad/TAFFC/IEMOCAP6_data/val_labels.json")
        labels = json.load(f)
        f.close()
    else:
        folder = "/home/soumyad/TAFFC/IEMOCAP_data/wavs"
        transcript_file = "/home/soumyad/MMER/IEMOCAP/IEMOCAP6_transcripts.json"
        f = open("/home/soumyad/TAFFC/IEMOCAP6_data/test_wav_labels.json")
        labels = json.load(f)
        f.close()
    dataset = SpeechDataset(folder, labels)
    loader = DataLoader(dataset,
                    batch_size=bs,
                    pin_memory=False,
                    shuffle=True,
                    drop_last=False,
                    collate_fn=dataset.collate)
    return loader

def create_RAVDESS_dataset(mode, bs=8):
    s = 1
    if mode == 'train':
        folder = "/data2/soumyad/RAVDESS/wavs"
        f = open("/data2/soumyad/RAVDESS/train_dict" + str(s) + ".json")
        labels = json.load(f)
        f.close()
    elif mode == 'val':
        folder = "/data2/soumyad/RAVDESS/wavs"
        f = open("/data2/soumyad/RAVDESS/val_dict" + str(s) + ".json")
        labels = json.load(f)
        f.close()
    else:
        folder = "/data2/soumyad/RAVDESS/wavs"
        f = open("/data2/soumyad/RAVDESS/test_dict" + str(s) + ".json")
        labels = json.load(f)
        f.close()
    dataset = SpeechDataset(folder, labels)
    loader = DataLoader(dataset,
                    batch_size=bs,
                    pin_memory=False,
                    shuffle=True,
                    drop_last=False,
                    collate_fn=dataset.collate)
    return loader

def create_CAFE_dataset(mode, bs=8):
    s = 5
    if mode == 'train':
        folder = "/data2/soumyad/S2ST/pitch_transfer/cafe_wavs16k"
        f = open("/data2/soumyad/S2ST/pitch_transfer/french_train_labels" + str(s) + ".json")
        labels = json.load(f)
        f.close()
    elif mode == 'val':
        folder = "/data2/soumyad/S2ST/pitch_transfer/cafe_wavs16k"
        f = open("/data2/soumyad/S2ST/pitch_transfer/french_valid_labels" + str(s) + ".json")
        labels = json.load(f)
        f.close()
    else:
        folder = "/data2/soumyad/S2ST/pitch_transfer/cafe_wavs16k"
        f = open("/data2/soumyad/S2ST/pitch_transfer/french_test_labels" + str(s) + ".json")
        labels = json.load(f)
        f.close()
    dataset = SpeechDataset(folder, labels)
    loader = DataLoader(dataset,
                    batch_size=bs,
                    pin_memory=False,
                    shuffle=True,
                    drop_last=False,
                    collate_fn=dataset.collate)
    return loader

def create_EMODB_dataset(mode, bs=8):
    s = 3
    if mode == 'train':
        folder = "/data2/soumyad/emodb/wav"
        f = open("/data2/soumyad/emodb/train_dict" + str(s) + ".json")
        labels = json.load(f)
        f.close()
    elif mode == 'val':
        folder = "/data2/soumyad/emodb/wav"
        f = open("/data2/soumyad/emodb/val_dict" + str(s) + ".json")
        labels = json.load(f)
        f.close()
    else:
        folder = "/data2/soumyad/emodb/wav"
        f = open("/data2/soumyad/emodb/test_dict" + str(s) + ".json")
        labels = json.load(f)
        f.close()
    dataset = SpeechDataset(folder, labels)
    loader = DataLoader(dataset,
                    batch_size=bs,
                    pin_memory=False,
                    shuffle=True,
                    num_workers=8,
                    drop_last=False,
                    collate_fn=dataset.collate)
    return loader

def create_BN_dataset(mode, bs=8):
    if mode == 'train':
        folder = "/data2/soumyad/indic_datasets/bn/wavs"
        f = open("/data2/soumyad/indic_datasets/bn/train_labels.json")
        labels = json.load(f)
        f.close()
    elif mode == 'val':
        folder = "/data2/soumyad/indic_datasets/bn/wavs"
        f = open("/data2/soumyad/indic_datasets/bn/val_labels.json")
        labels = json.load(f)
        f.close()
    else:
        folder = "/data2/soumyad/indic_datasets/bn/wavs"
        f = open("/data2/soumyad/indic_datasets/bn/test_labels.json")
        labels = json.load(f)
        f.close()
    dataset = SpeechDataset(folder, labels)
    loader = DataLoader(dataset,
                    batch_size=bs,
                    pin_memory=False,
                    shuffle=True,
                    drop_last=False,
                    collate_fn=dataset.collate)
    return loader

def create_TM_dataset(mode, bs=8):
    if mode == 'train':
        folder = "/data2/soumyad/indic_datasets/public/wavs"
        f = open("/data2/soumyad/indic_datasets/public/train_labels.json")
        labels = json.load(f)
        f.close()
    elif mode == 'val':
        folder = "/data2/soumyad/indic_datasets/public/wavs"
        f = open("/data2/soumyad/indic_datasets/public/val_labels.json")
        labels = json.load(f)
        f.close()
    else:
        folder = "/data2/soumyad/indic_datasets/public/wavs"
        f = open("/data2/soumyad/indic_datasets/public/test_labels.json")
        labels = json.load(f)
        f.close()
    dataset = SpeechDataset(folder, labels)
    loader = DataLoader(dataset,
                    batch_size=bs,
                    pin_memory=False,
                    shuffle=True,
                    drop_last=False,
                    collate_fn=dataset.collate)
    return loader


def encode_dataset():
    # wavlm_config = WavLMConfig()
    # wavlm_model = WavLMModel.from_pretrained('microsoft/wavlm-base', config = wavlm_config)
    # roberta_model = RobertaModel.from_pretrained('roberta-base')
    # model = SpeechTextModel(wavlm_model, roberta_model, num_layers=6, common_model="roberta")
    # model = torch.load('/data2/soumyad/emo_pretraining/models_speechtext_unpaired_textlabels/model-70000.pth', map_location=device)
    # model = model.to(device)

    wavlm_config = WavLMConfig()
    wavlm_model = WavLMModel.from_pretrained('microsoft/wavlm-base')
    roberta_model = RobertaModel.from_pretrained('roberta-base')
    model = SpeechTextModel(wavlm_model, roberta_model, num_layers=6, common_model="roberta", use_conv="True", pool_fn="avg")
    emo_model = EmotionClassifier(model, 3)
    emo_model = torch.load(args.model, map_location=device)
    # checkpoint = torch.load("test/care_supervised.tar", map_location = device)
    # k = []
    # new_dict = checkpoint['model_state_dict']
    # for name, param in new_dict.items():
    #     if "joint_model.linear" in name:
    #         k.append(name)
    # for n in k:
    #     del new_dict[n]
    # emo_model.load_state_dict(new_dict, strict=False)
    # emo_model = emo_model.to(device)
    # emo_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    # final_model = torch.load('models_m1k0c0_128/model-20000.pth', map_location=device)
    # final_model = final_model.to(device)

    if args.dataset == "MELD":
        train_loader = create_MELD_dataset("train", 1)
        val_loader = create_MELD_dataset("val", 1)
        test_loader = create_MELD_dataset("test", 1)
        out_dir_emo = "/home/soumyad/feats_MELD_own"
        out_dir_os = "/home/soumyad/feats_MELD_os"
    elif args.dataset == "MOSI":
        train_loader = create_MOSI_dataset("train", 1)
        val_loader = create_MOSI_dataset("val", 1)
        test_loader = create_MOSI_dataset("test", 1)
        out_dir_emo = "/home/soumyad/feats_MOSI_own"
        out_dir_os = "/home/soumyad/feats_MOSI_os"
    elif args.dataset == "IEMOCAP4":
        train_loader = create_IEMOCAP4_dataset("train", 1)
        val_loader = create_IEMOCAP4_dataset("val", 1)
        test_loader = create_IEMOCAP4_dataset("test", 1)
        out_dir_emo = "/home/soumyad/feats_IEMOCAP4_own"
        out_dir_os = "/home/soumyad/feats_IEMOCAP4_os"
    elif args.dataset == "IEMOCAP6":
        train_loader = create_IEMOCAP6_dataset("train", 1)
        val_loader = create_IEMOCAP6_dataset("val", 1)
        test_loader = create_IEMOCAP6_dataset("test", 1)
        out_dir_emo = "/home/soumyad/feats_IEMOCAP6_own"
        out_dir_os = "/home/soumyad/feats_IEMOCAP6_os"
    elif args.dataset == "RAVDESS":
        train_loader = create_RAVDESS_dataset("train", 1)
        val_loader = create_RAVDESS_dataset("val", 1)
        test_loader = create_RAVDESS_dataset("test", 1)
        out_dir_emo = "/home/soumyad/feats_RAVDESS_own"
        out_dir_os = "/home/soumyad/feats_RAVDESS_os"
    elif args.dataset == "CAFE":
        train_loader = create_CAFE_dataset("train", 1)
        val_loader = create_CAFE_dataset("val", 1)
        test_loader = create_CAFE_dataset("test", 1)
        out_dir_emo = "/home/soumyad/feats_CAFE_own"
        out_dir_os = "/home/soumyad/feats_CAFE_os"
    elif args.dataset == "EMODB":
        train_loader = create_EMODB_dataset("train", 1)
        val_loader = create_EMODB_dataset("val", 1)
        test_loader = create_EMODB_dataset("test", 1)
        out_dir_emo = "/home/soumyad/feats_EMODB_own"
        out_dir_os = "/home/soumyad/feats_EMODB_os"
    elif args.dataset == "BN":
        train_loader = create_BN_dataset("train", 1)
        val_loader = create_BN_dataset("val", 1)
        test_loader = create_BN_dataset("test", 1)
        out_dir_emo = "/home/soumyad/feats_BN_own"
        out_dir_os = "/home/soumyad/feats_BN_os"
    elif args.dataset == "TM":
        train_loader = create_TM_dataset("train", 1)
        val_loader = create_TM_dataset("val", 1)
        test_loader = create_TM_dataset("test", 1)
        out_dir_emo = "/home/soumyad/feats_TM_own"
        out_dir_os = "/home/soumyad/feats_TM_os"
    else:
        print("Dataset not found")
    os.makedirs(out_dir_emo, exist_ok=True)
    os.makedirs(os.path.join(out_dir_emo, "train"), exist_ok=True)
    os.makedirs(os.path.join(out_dir_emo, "val"), exist_ok=True)
    os.makedirs(os.path.join(out_dir_emo, "test"), exist_ok=True)
    os.makedirs(os.path.join(out_dir_os, "train"), exist_ok=True)
    os.makedirs(os.path.join(out_dir_os, "val"), exist_ok=True)
    os.makedirs(os.path.join(out_dir_os, "test"), exist_ok=True)
    alpha = args.alpha
    print("***************************************************")
    print("Dataset:", args.dataset)
    print("Alpha:", args.alpha)
    print("Model:", args.model)
    print("***************************************************")
    with torch.no_grad():
        for i, data in enumerate(tqdm(train_loader)):
            aud, name = data
            
            with torch.no_grad():
                feat, pooled_audio, feat_aud = emo_model(aud.to(device))
            out_path_emo_aud = os.path.join(out_dir_emo, "train", name.replace(".wav", "_aud.npy"))
            out_path_os_aud = os.path.join(out_dir_os, "train", name.replace(".wav", "_aud.npy"))
            feat_aud = feat_aud.squeeze(1)
            pooled_aud = torch.mean(feat_aud[-1, :, :], 0)
            feat = feat.squeeze(1)
            # feat_aud = alpha*feat + (1-alpha)*feat_aud
            feat_aud = torch.cat((feat, feat_aud), -1)
            x_aud = torch.mean(feat_aud, 1)
            pooled_audio = pooled_audio.squeeze(0)
            
            np.save(out_path_emo_aud, x_aud.cpu().detach().numpy())
            np.save(out_path_os_aud, pooled_audio.cpu().detach().numpy())

        for i, data in enumerate(tqdm(val_loader)):
            aud, name = data
            with torch.no_grad():
                feat, pooled_audio, feat_aud = emo_model(aud.to(device))
            out_path_emo_aud = os.path.join(out_dir_emo, "val", name.replace(".wav", "_aud.npy"))
            out_path_os_aud = os.path.join(out_dir_os, "val", name.replace(".wav", "_aud.npy"))
            feat_aud = feat_aud.squeeze(1)
            pooled_aud = torch.mean(feat_aud[-1, :, :], 0)
            feat = feat.squeeze(1)
            # feat_aud = alpha*feat + (1-alpha)*feat_aud
            feat_aud = torch.cat((feat, feat_aud), -1)
            x_aud = torch.mean(feat_aud, 1)
            pooled_audio = pooled_audio.squeeze(0)
            
            np.save(out_path_emo_aud, x_aud.cpu().detach().numpy())
            np.save(out_path_os_aud, pooled_audio.cpu().detach().numpy())
        
        for i, data in enumerate(tqdm(test_loader)):
            aud, name = data
            with torch.no_grad():
                feat, pooled_audio, feat_aud = emo_model(aud.to(device))
            out_path_emo_aud = os.path.join(out_dir_emo, "test", name.replace(".wav", "_aud.npy"))
            out_path_os_aud = os.path.join(out_dir_os, "test", name.replace(".wav", "_aud.npy"))
            feat_aud = feat_aud.squeeze(1)
            pooled_aud = torch.mean(feat_aud[-1, :, :], 0)
            feat = feat.squeeze(1)
            # feat_aud = alpha*feat + (1-alpha)*feat_aud
            feat_aud = torch.cat((feat, feat_aud), -1)
            x_aud = torch.mean(feat_aud, 1)
            pooled_audio = pooled_audio.squeeze(0)
            
            np.save(out_path_emo_aud, x_aud.cpu().detach().numpy())
            np.save(out_path_os_aud, pooled_audio.cpu().detach().numpy())

# def get_pitch():
#     # wavlm_config = WavLMConfig()
#     # wavlm_model = WavLMModel.from_pretrained('microsoft/wavlm-base', config = wavlm_config)
#     # roberta_model = RobertaModel.from_pretrained('roberta-base')
#     # model = SpeechTextModel(wavlm_model, roberta_model, num_layers=6, common_model="roberta")
#     model = torch.load('/data2/soumyad/emo_pretraining/models_distil_regression10011/model-60000.pth', map_location=device)
#     model = model.to(device)

#     if args.dataset == "MELD":
#         train_loader = create_MELD_dataset("train", 1)
#         val_loader = create_MELD_dataset("val", 1)
#         test_loader = create_MELD_dataset("test", 1)
#         out_dir_emo = "/home/soumyad/pitch_MELD_own"
#         # out_dir_os = "/home/soumyad/feats_MELD_os"
#     elif args.dataset == "MOSI":
#         train_loader = create_MOSI_dataset("train", 1)
#         val_loader = create_MOSI_dataset("val", 1)
#         test_loader = create_MOSI_dataset("test", 1)
#         out_dir_emo = "/home/soumyad/pitch_MOSI_own"
#         # out_dir_os = "/home/soumyad/feats_MOSI_os"
#     elif args.dataset == "IEMOCAP4":
#         train_loader = create_IEMOCAP4_dataset("train", 1)
#         val_loader = create_IEMOCAP4_dataset("val", 1)
#         test_loader = create_IEMOCAP4_dataset("test", 1)
#         out_dir_emo = "/home/soumyad/pitch_IEMOCAP4_own"
#         # out_dir_os = "/home/soumyad/feats_IEMOCAP4_os"
#     elif args.dataset == "IEMOCAP6":
#         train_loader = create_IEMOCAP6_dataset("train", 1)
#         val_loader = create_IEMOCAP6_dataset("val", 1)
#         test_loader = create_IEMOCAP6_dataset("test", 1)
#         out_dir_emo = "/home/soumyad/pitch_IEMOCAP6_own"
#         # out_dir_os = "/home/soumyad/feats_IEMOCAP6_os"
#     else:
#         print("Dataset not found")
#     os.makedirs(out_dir_emo, exist_ok=True)
#     os.makedirs(os.path.join(out_dir_emo, "train"), exist_ok=True)
#     os.makedirs(os.path.join(out_dir_emo, "val"), exist_ok=True)
#     os.makedirs(os.path.join(out_dir_emo, "test"), exist_ok=True)
#     # os.makedirs(os.path.join(out_dir_os, "train"), exist_ok=True)
#     # os.makedirs(os.path.join(out_dir_os, "val"), exist_ok=True)
#     # os.makedirs(os.path.join(out_dir_os, "test"), exist_ok=True)
#     with torch.no_grad():
#         for i, data in enumerate(tqdm(train_loader)):
#             aud, name = data
            
#             with torch.no_grad():
#                 feat = model.extract_pitch(aud.to(device))
#             out_path_emo_aud = os.path.join(out_dir_emo, "train", name.replace(".wav", "_aud.npy"))
#             # out_path_os_aud = os.path.join(out_dir_os, "train", name.replace(".wav", "_aud.npy"))
#             x_aud = feat.squeeze(0)
#             # x_os = feat_os.squeeze(1)
            
#             np.save(out_path_emo_aud, x_aud.cpu().detach().numpy())
#             # np.save(out_path_os_aud, x_os.cpu().detach().numpy())

#         for i, data in enumerate(tqdm(val_loader)):
#             aud, name = data
            
#             with torch.no_grad():
#                 feat = model.extract_audio_features(aud.to(device))
#             out_path_emo_aud = os.path.join(out_dir_emo, "val", name.replace(".wav", "_aud.npy"))
#             # out_path_os_aud = os.path.join(out_dir_os, "val", name.replace(".wav", "_aud.npy"))
#             x_aud = feat.squeeze(1)
#             # x_os = feat_os.squeeze(1)
#             np.save(out_path_emo_aud, x_aud.cpu().detach().numpy())
#             # np.save(out_path_os_aud, x_os.cpu().detach().numpy())
        
#         for i, data in enumerate(tqdm(test_loader)):
#             aud, name = data
            
#             with torch.no_grad():
#                 feat = model.extract_audio_features(aud.to(device))
#             out_path_emo_aud = os.path.join(out_dir_emo, "test", name.replace(".wav", "_aud.npy"))
#             # out_path_os_aud = os.path.join(out_dir_os, "test", name.replace(".wav", "_aud.npy"))
#             x_aud = feat.squeeze(1)
#             # x_os = feat_os.squeeze(1)
#             np.save(out_path_emo_aud, x_aud.cpu().detach().numpy())
#             # np.save(out_path_os_aud, x_os.cpu().detach().numpy())

if __name__ == "__main__":
    encode_dataset()
    # get_pitch()
