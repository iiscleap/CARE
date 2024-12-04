import os
import torch
import torchaudio
import librosa
import logging
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import Adam, SGD
import torch.nn as nn
import random
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import random
import argparse
import time
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pickle

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

class MyDataset(Dataset):

    def __init__(self, folder, target_dict):
        self.folder = folder
        self.target = target_dict
        wav_files = os.listdir(self.folder)
        wav_files = [x.replace("_aud.npy", ".wav") for x in wav_files]
        wav_files = [x for x in wav_files if ".wav" in x]
        self.wav_files = wav_files
        self.feats = {}
        self.pooled_feats = {}
        for i, f in enumerate(tqdm(self.wav_files)):
            feat_path = os.path.join(self.folder, f.replace(".wav", "_aud.npy"))
            self.feats[f] = np.load(feat_path)
        for i, f in enumerate(tqdm(self.wav_files)):
            feat_path = os.path.join(self.folder.replace("own", "os"), f.replace(".wav", "_aud.npy"))
            self.pooled_feats[f] = np.load(feat_path)

    def __len__(self):
        return len(self.wav_files) 
        
    def __getitem__(self, audio_ind):
        start_time = time.time()
        feature_file = self.feats[self.wav_files[audio_ind]]
        pooled_feat = self.pooled_feats[self.wav_files[audio_ind]]
        class_id = self.target[self.wav_files[audio_ind].replace(".npy", ".wav")]    
        end_time = time.time() 

        return feature_file, pooled_feat, class_id, self.wav_files[audio_ind]

class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
        self.softmax = nn.functional.softmax

    def forward(self, batch_rep, att_mask=None):
        """
            N: batch size, T: sequence length, H: Hidden dimension
            input:
                batch_rep : size (N, T, H)
            attention_weight:
                att_w : size (N, T, 1)
            return:
                utter_rep: size (N, H)
        """
        att_logits = self.W(batch_rep).squeeze(-1)
        if att_mask is not None:
            att_logits = att_mask + att_logits
        att_w = self.softmax(att_logits, dim=-1).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep

class EmotionClassifier(nn.Module):
    def __init__(self,
                 hidden_dim,
                 output_dim):
        
        super().__init__()
        
        self.weights = nn.Parameter(torch.rand(13, 1))
        self.out = nn.Linear(hidden_dim, output_dim)
        self.fc = nn.Linear(768*3, 768)
        self.fc_1 = nn.Linear(768, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.layer_norm = nn.LayerNorm(768)
        
    def forward(self, feat, pooled_feats):
        feat = feat.permute(0, 2, 1)
        weights_normalized = nn.functional.softmax(self.weights, dim=0)
        feats_final = torch.matmul(feat, weights_normalized.squeeze())
        feat = feats_final#[:, :768]
        feat = torch.cat((feat, pooled_feats), -1)
        output = self.fc_1((self.dropout(self.fc(feat))))
        out = self.out(self.dropout(output))
        # output = self.out(self.dropout(self.relu(self.fc_1(pooled_feats))))
        return out, output

def compute_accuracy(output, labels):
    #Function for calculating accuracy
    pred = torch.argmax(output, dim = 1)
    correct_pred = (pred == labels).float()
    tot_correct = correct_pred.sum()

    return tot_correct

def compute_loss(output, labels):
    #Function for calculating loss

    ce_loss = nn.CrossEntropyLoss(reduction='none')(output, labels.squeeze(-1).long())
    pt = torch.exp(-ce_loss)
    loss = ((1-pt)**0 * ce_loss).mean()
    return loss

def create_MELD_dataset(mode, bs=8):
    if mode == 'train':
        folder = "/home/soumyad/feats_MELD_own/train"
        f = open("/home/soumyad/TAFFC/MELD_data/train_wav_labels.json")
        labels = json.load(f)
        f.close()
    elif mode == 'val':
        folder = "/home/soumyad/feats_MELD_own/val"
        f = open("/home/soumyad/TAFFC/MELD_data/val_wav_labels.json")
        labels = json.load(f)
        f.close()
    else:
        folder = "/home/soumyad/feats_MELD_own/test"
        f = open("/home/soumyad/TAFFC/MELD_data/test_wav_labels.json")
        labels = json.load(f)
        f.close()
    dataset = MyDataset(folder, labels)
    loader = DataLoader(dataset,
                    batch_size=bs,
                    pin_memory=False,
                    shuffle=True,
                    num_workers=8,
                    drop_last=False)
    return loader
    
def create_MOSI_dataset(mode, bs=8):
    if mode == 'train':
        folder = "/home/soumyad/feats_MOSI_own/train"
        f = open("/home/soumyad/TAFFC/MOSI_data/train_wav_labels.json")
        labels = json.load(f)
        f.close()
    elif mode == 'val':
        folder = "/home/soumyad/feats_MOSI_own/val"
        f = open("/home/soumyad/TAFFC/MOSI_data/val_wav_labels.json")
        labels = json.load(f)
        f.close()
    else:
        folder = "/home/soumyad/feats_MOSI_own/test"
        f = open("/home/soumyad/TAFFC/MOSI_data/test_wav_labels.json")
        labels = json.load(f)
        f.close()
    dataset = MyDataset(folder, labels)
    loader = DataLoader(dataset,
                    batch_size=bs,
                    pin_memory=False,
                    shuffle=True,
                    drop_last=False)
    return loader

def create_IEMOCAP4_dataset(mode, bs=8):
    if mode == 'train':
        folder = "/home/soumyad/feats_IEMOCAP4_own/train"
        f = open("/data2/soumyad/IEMOCAP_data/train_labels.json")
        labels = json.load(f)
        f.close()
    elif mode == 'val':
        folder = "/home/soumyad/feats_IEMOCAP4_own/val"
        f = open("/data2/soumyad/IEMOCAP_data/val_labels.json")
        labels = json.load(f)
        f.close()
    else:
        folder = "/home/soumyad/feats_IEMOCAP4_own/test"
        f = open("/data2/soumyad/IEMOCAP_data/test_wav_labels.json")
        labels = json.load(f)
        f.close()
    dataset = MyDataset(folder, labels)
    loader = DataLoader(dataset,
                    batch_size=bs,
                    pin_memory=False,
                    shuffle=True,
                    drop_last=False)
    return loader

def create_IEMOCAP6_dataset(mode, bs=8):
    if mode == 'train':
        folder = "/home/soumyad/feats_IEMOCAP6_own/train"
        f = open("/home/soumyad/TAFFC/IEMOCAP6_data/train_labels.json")
        labels = json.load(f)
        f.close()
    elif mode == 'val':
        folder = "/home/soumyad/feats_IEMOCAP6_own/val"
        f = open("/home/soumyad/TAFFC/IEMOCAP6_data/val_labels.json")
        labels = json.load(f)
        f.close()
    else:
        folder = "/home/soumyad/feats_IEMOCAP6_own/test"
        f = open("/home/soumyad/TAFFC/IEMOCAP6_data/test_wav_labels.json")
        labels = json.load(f)
        f.close()
    dataset = MyDataset(folder, labels)
    loader = DataLoader(dataset,
                    batch_size=bs,
                    pin_memory=False,
                    shuffle=True,
                    num_workers=8,
                    drop_last=False)
    return loader

def create_RAVDESS_dataset(mode, bs=8):
    if mode == 'train':
        folder = "/home/soumyad/feats_RAVDESS_own/train"
        f = open("/data2/soumyad/RAVDESS/train_dict1.json")
        labels = json.load(f)
        f.close()
    elif mode == 'val':
        folder = "/home/soumyad/feats_RAVDESS_own/val"
        f = open("/data2/soumyad/RAVDESS/val_dict1.json")
        labels = json.load(f)
        f.close()
    else:
        folder = "/home/soumyad/feats_RAVDESS_own/test"
        f = open("/data2/soumyad/RAVDESS/test_dict1.json")
        labels = json.load(f)
        f.close()
    dataset = MyDataset(folder, labels)
    loader = DataLoader(dataset,
                    batch_size=bs,
                    pin_memory=False,
                    shuffle=True,
                    num_workers=8,
                    drop_last=False)
    return loader

def create_CAFE_dataset(mode, bs=8):
    if mode == 'train':
        folder = "/home/soumyad/feats_CAFE_own/train"
        f = open("/data2/soumyad/S2ST/pitch_transfer/french_train_labels5.json")
        labels = json.load(f)
        f.close()
    elif mode == 'val':
        folder = "/home/soumyad/feats_CAFE_own/val"
        f = open("/data2/soumyad/S2ST/pitch_transfer/french_valid_labels5.json")
        labels = json.load(f)
        f.close()
    else:
        folder = "/home/soumyad/feats_CAFE_own/test"
        f = open("/data2/soumyad/S2ST/pitch_transfer/french_test_labels5.json")
        labels = json.load(f)
        f.close()
    dataset = MyDataset(folder, labels)
    loader = DataLoader(dataset,
                    batch_size=bs,
                    pin_memory=False,
                    shuffle=True,
                    num_workers=8,
                    drop_last=False)
    return loader

def create_EMODB_dataset(mode, bs=8):
    if mode == 'train':
        folder = "/home/soumyad/feats_EMODB_own/train"
        f = open("/data2/soumyad/emodb/train_dict3.json")
        labels = json.load(f)
        f.close()
    elif mode == 'val':
        folder = "/home/soumyad/feats_EMODB_own/val"
        f = open("/data2/soumyad/emodb/val_dict3.json")
        labels = json.load(f)
        f.close()
    else:
        folder = "/home/soumyad/feats_EMODB_own/test"
        f = open("/data2/soumyad/emodb/test_dict3.json")
        labels = json.load(f)
        f.close()
    dataset = MyDataset(folder, labels)
    loader = DataLoader(dataset,
                    batch_size=bs,
                    pin_memory=False,
                    shuffle=True,
                    num_workers=8,
                    drop_last=False)
    return loader

def create_BN_dataset(mode, bs=8):
    if mode == 'train':
        folder = "/home/soumyad/feats_BN_own/train"
        f = open("/data2/soumyad/indic_datasets/bn/train_labels.json")
        labels = json.load(f)
        f.close()
    elif mode == 'val':
        folder = "/home/soumyad/feats_BN_own/val"
        f = open("/data2/soumyad/indic_datasets/bn/val_labels.json")
        labels = json.load(f)
        f.close()
    else:
        folder = "/home/soumyad/feats_BN_own/test"
        f = open("/data2/soumyad/indic_datasets/bn/test_labels.json")
        labels = json.load(f)
        f.close()
    dataset = MyDataset(folder, labels)
    loader = DataLoader(dataset,
                    batch_size=bs,
                    pin_memory=False,
                    shuffle=True,
                    num_workers=8,
                    drop_last=False)
    return loader

def create_TM_dataset(mode, bs=8):
    if mode == 'train':
        folder = "/home/soumyad/feats_TM_own/train"
        f = open("/data2/soumyad/indic_datasets/public/train_labels.json")
        labels = json.load(f)
        f.close()
    elif mode == 'val':
        folder = "/home/soumyad/feats_TM_own/val"
        f = open("/data2/soumyad/indic_datasets/public/val_labels.json")
        labels = json.load(f)
        f.close()
    else:
        folder = "/home/soumyad/feats_TM_own/test"
        f = open("/data2/soumyad/indic_datasets/public/test_labels.json")
        labels = json.load(f)
        f.close()
    dataset = MyDataset(folder, labels)
    loader = DataLoader(dataset,
                    batch_size=bs,
                    pin_memory=False,
                    shuffle=True,
                    num_workers=8,
                    drop_last=False)
    return loader

def train():

    if args.dataset == "MELD":
        train_loader = create_MELD_dataset("train", 32)
        val_loader = create_MELD_dataset("val", 32)
        num_classes = 7
    elif args.dataset == "MOSI":
        train_loader = create_MOSI_dataset("train", 32)
        val_loader = create_MOSI_dataset("val", 32)
        num_classes = 2
    elif args.dataset == "IEMOCAP4":
        train_loader = create_IEMOCAP4_dataset("train", 32)
        val_loader = create_IEMOCAP4_dataset("val", 32)
        num_classes = 4
    elif args.dataset == "IEMOCAP6":
        train_loader = create_IEMOCAP6_dataset("train", 32)
        val_loader = create_IEMOCAP6_dataset("val", 32)
        num_classes = 6
    elif args.dataset == "RAVDESS":
        train_loader = create_RAVDESS_dataset("train", 32)
        val_loader = create_RAVDESS_dataset("val", 32)
        num_classes = 7
    elif args.dataset == "CAFE":
        train_loader = create_CAFE_dataset("train", 32)
        val_loader = create_CAFE_dataset("val", 32)
        num_classes = 7
    elif args.dataset == "EMODB":
        train_loader = create_EMODB_dataset("train", 32)
        val_loader = create_EMODB_dataset("val", 32)
        num_classes = 7
    elif args.dataset == "BN":
        train_loader = create_BN_dataset("train", 32)
        val_loader = create_BN_dataset("val", 32)
        num_classes = 7
    elif args.dataset == "TM":
        train_loader = create_TM_dataset("train", 32)
        val_loader = create_TM_dataset("val", 32)
        num_classes = 7
    else:
        print("Dataset not found")


    model = EmotionClassifier(256, num_classes)
    # checkpoint = torch.load('model_frozen_own.tar', map_location = device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    lr = 1e-4
    optimizer = Adam([{'params':model.parameters(), 'lr':lr}])
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=20)
    final_val_loss = 0

    for e in range(50):
        model.train()
        tot_loss, tot_correct = 0.0, 0.0
        val_loss, val_acc = 0.0, 0.0
        val_correct = 0.0
        train_size = 0
        val_size = 0
        pred_tr = []
        gt_tr = []
        pred_val = []
        gt_val = []
        for i, data in enumerate(tqdm(train_loader)):
            train_size += data[0].shape[0]
            model.zero_grad()
            # Get the input features and target labels, and put them on the GPU
            feats, pooled_feats, labels = data[0].to(device), data[1].to(device), data[2].to(device)
            final_out, _ = model(feats, pooled_feats)
            loss = compute_loss(final_out, labels)
            optimizer.zero_grad()
            loss.backward()
            tot_loss += loss.detach().item()
            optimizer.step()
            pred = torch.argmax(final_out, dim = 1)
            pred = pred.detach().cpu().numpy()
            pred = list(pred)
            pred_tr.extend(pred)
            labels = labels.detach().cpu().numpy()
            labels = list(labels)
            gt_tr.extend(labels)
            optimizer.zero_grad()
        # scheduler.step()
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(tqdm(val_loader)):
                val_size += data[0].shape[0]
                feats, pooled_feats, labels = data[0].to(device), data[1].to(device), data[2].to(device)
                val_out, _ = model(feats.to(device), pooled_feats)
                loss = compute_loss(val_out, labels)
                val_loss += loss.item()
                pred = torch.argmax(val_out, dim = 1)
                pred = pred.detach().cpu().numpy()
                pred = list(pred)
                pred_val.extend(pred)
                labels = labels.detach().cpu().numpy()
                labels = list(labels)
                gt_val.extend(labels)
        val_f1 = f1_score(gt_val, pred_val, average='weighted')
        if val_f1 > final_val_loss:
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),},
                        'model_frozen_own.tar')
            final_val_loss = val_f1
        train_loss = tot_loss/len(train_loader)
        train_f1 = f1_score(gt_tr, pred_tr, average='weighted')
        val_loss_log = val_loss/len(val_loader)
        val_f1 = f1_score(gt_val, pred_val, average='weighted')
        e_log = e + 1
        logger.info(f"Epoch {e_log}, \
                    Training Loss {train_loss},\
                    Training Accuracy {train_f1}")
        logger.info(f"Epoch {e_log}, \
                    Validation Loss {val_loss_log},\
                    Validation Accuracy {val_f1}")
        
def test():

    if args.dataset == "MELD":
        test_loader = create_MELD_dataset("test", 1)
        num_classes = 7
    elif args.dataset == "MOSI":
        test_loader = create_MOSI_dataset("test", 1)
        num_classes = 2
    elif args.dataset == "IEMOCAP4":
        test_loader = create_IEMOCAP4_dataset("test", 1)
        num_classes = 4
    elif args.dataset == "IEMOCAP6":
        test_loader = create_IEMOCAP6_dataset("test", 1)
        num_classes = 6
    elif args.dataset == "BN":
        test_loader = create_BN_dataset("test", 1)
        num_classes = 7
    elif args.dataset == "TM":
        test_loader = create_TM_dataset("test", 1)
        num_classes = 7
    else:
        print("Dataset not found")
    model = EmotionClassifier(256, num_classes)
    checkpoint = torch.load('model_frozen_own.tar', map_location = device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    # print(model.layers_comb.weight)
    model.eval()
    pred_test, gt_test = [], []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            feats, pooled_feats, labels = data[0].to(device), data[1].to(device), data[2].to(device)
            test_out, _ = model(feats, pooled_feats)
            pred = torch.argmax(test_out, dim = 1)
            pred = pred.detach().cpu().numpy()
            pred = list(pred)
            pred_test.extend(pred)
            labels = labels.detach().cpu().numpy()
            labels = list(labels)
            gt_test.extend(labels)
    test_f1 = f1_score(gt_test, pred_test, average='weighted')
    logger.info(f"Test Accuracy {test_f1}")

def get_feats():
    if args.dataset == "MELD":
        train_loader = create_MELD_dataset("train", 1)
        val_loader = create_MELD_dataset("val", 1)
        test_loader = create_MELD_dataset("test", 1)
        num_classes = 7
        out_file = "audio_features_meld.pkl"
        logits_file = "audio_logits_meld.pkl"
    elif args.dataset == "MOSI":
        train_loader = create_MOSI_dataset("train", 1)
        val_loader = create_MOSI_dataset("val", 1)
        test_loader = create_MOSI_dataset("test", 1)
        num_classes = 2
        out_file = "audio_features_mosi.pkl"
        logits_file = "audio_logits_mosi.pkl"
    elif args.dataset == "IEMOCAP4":
        train_loader = create_IEMOCAP4_dataset("train", 1)
        val_loader = create_IEMOCAP4_dataset("val", 1)
        test_loader = create_IEMOCAP4_dataset("test", 1)
        out_file = "audio_features_iemocap4.pkl"
        logits_file = "audio_logits_iemocap4.pkl"
        num_classes = 4
    elif args.dataset == "IEMOCAP6":
        train_loader = create_IEMOCAP6_dataset("train", 1)
        val_loader = create_IEMOCAP6_dataset("val", 1)
        test_loader = create_IEMOCAP6_dataset("test", 1)
        out_file = "audio_features_iemocap6.pkl"
        logits_file = "audio_logits_iemocap6.pkl"
        num_classes = 6
    else:
        print("Dataset not found")
    
    model = EmotionClassifier(1024, num_classes)
    checkpoint = torch.load('model_frozen_own.tar', map_location = device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    train_dict, val_dict, test_dict = dict(), dict(), dict()
    test_logits = dict()
    with torch.no_grad():
        for i, data in enumerate(tqdm(train_loader)):
            feats, pooled_feats, labels, name = data
            
            with torch.no_grad():
                _, out_feats = model(feats.to(device), pooled_feats.to(device))
            train_dict[name[0]] = out_feats.squeeze(0).cpu().detach().numpy()
        
        with open(out_file.replace(".pkl", "_train.pkl"), 'wb') as handle:
            pickle.dump(train_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


        for i, data in enumerate(tqdm(val_loader)):
            feats, pooled_feats, labels, name = data
            
            with torch.no_grad():
                _, out_feats = model(feats.to(device), pooled_feats.to(device))
            val_dict[name[0]] = out_feats.squeeze(0).cpu().detach().numpy()
        
        with open(out_file.replace(".pkl", "_valid.pkl"), 'wb') as handle:
            pickle.dump(val_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        for i, data in enumerate(tqdm(test_loader)):
            feats, pooled_feats, labels, name = data
            
            with torch.no_grad():
                out, out_feats = model(feats.to(device), pooled_feats.to(device))
            test_dict[name[0]] = out_feats.squeeze(0).cpu().detach().numpy()
            test_logits[name[0]] = out.squeeze(0).cpu().detach().numpy()
        with open(out_file.replace(".pkl", "_test.pkl"), 'wb') as handle:
            pickle.dump(test_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(logits_file, 'wb') as handle:
            pickle.dump(test_logits, handle, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    train()
    test()
    # get_feats()
