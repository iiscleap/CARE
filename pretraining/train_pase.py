import os
import torch
import logging
import numpy as np
import json
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.cuda.amp as amp
import random
from tqdm import tqdm
import random
import torch.nn.functional as F
from transformers import WavLMModel, RobertaModel, WavLMConfig
from transformers import AutoFeatureExtractor
from dataset_pase import SpeechTextDataset
from model_pase import SpeechTextModel
import argparse


# torch.set_printoptions(profile="full")
#Logger set

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
torch.autograd.set_detect_anomaly(True)
#CUDA devices enabled
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.cuda.empty_cache()


# LOG_INTERVAL = 5
# VALIDATION_INTERVAL = 1000
# CHECKPOINT_INTERVAL = 5000


parser = argparse.ArgumentParser(description="Train wavlm soft content encoder.")
parser.add_argument(
    "checkpoint_dir",
    metavar="checkpoint_dir",
    help="path to save models.",
    type=str,
)
parser.add_argument(
        "--alpha",
        help="weight for the masked loss.",
        default=1.0,
        type=float,
    )

parser.add_argument(
        "--common_model",
        help="what to use as fusion module",
        default="roberta",
        type=str,
    )

parser.add_argument(
        "--batch_size",
        help="batch size",
        type=int,
    )

parser.add_argument(
        "--num_layers",
        help="number of layers to use",
        type=int,
        default=6
    )

parser.add_argument(
        "--use_conv",
        help="whether to use conv layers",
        type=str,
        default="True"
    )

parser.add_argument(
        "--use_pretrained",
        help="whether to use pretrained wavlm",
        type=str,
        default="True"
    )

parser.add_argument(
        "--pool_fn",
        help="type of pooling function",
        type=str,
        default="avg"
    )

parser.add_argument(
        "--energy_weights",
        help="whether to use energy_weights",
        type=str,
        default="False"
    )
parser.add_argument(
        "--pitch_weights",
        help="whether to use pitch_weights",
        type=str,
        default="False"
    )

parser.add_argument(
        "--supervised",
        help="whether to use supervised roberta",
        type=str,
        default="False"
    )




args = parser.parse_args()


BATCH_SIZE = args.batch_size
LEARNING_RATE = 1e-5
BETAS = (0.9, 0.99)
EPS = 1e-06
WEIGHT_DECAY = 0
MAX_NORM = 10
STEPS = 800000

class EmotionClassifier(nn.Module):
    def __init__(self, joint_model, output_dim):
        super().__init__()
        self.joint_model = joint_model
        self.fc_text = nn.Linear(768*2, 768)
        self.act = nn.ReLU()
        self.fc_text_final = nn.Linear(768, output_dim)

    def forward(self, audio):
        speech_feats, pooled_audio, _, _ = self.joint_model(audio.to(device))
        # comb = torch.mean(x, 0).squeeze(0)
        # comb = torch.mean(comb, 1).squeeze(1)
        # pooled_text = torch.cat((comb, pooled_text), -1)
        # x_text = self.act(self.fc_text(pooled_text))
        # pred_text = self.fc_text_final(x_text)

        return speech_feats, pooled_audio

def zero_grad_hook(grad):
    # Zero out the gradients for the common part
    return torch.zeros_like(grad)

def train(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    if args.supervised == "True":
        train_dataset = SpeechTextDataset(train=True, supervised=True)
        valid_dataset = SpeechTextDataset(train=False, supervised=True)
    else:
        train_dataset = SpeechTextDataset(train=True, supervised=False)
        valid_dataset = SpeechTextDataset(train=False, supervised=False)
    train_loader = DataLoader(
        train_dataset,
        collate_fn=train_dataset.collate,
        batch_size=BATCH_SIZE,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        collate_fn=valid_dataset.collate,
        batch_size=BATCH_SIZE,
        pin_memory=True,
        shuffle=False,
        drop_last=True,
    )
    wavlm_config = WavLMConfig()
    if args.use_pretrained == "True":
        wavlm_model = WavLMModel.from_pretrained('microsoft/wavlm-base', config = wavlm_config)
    else:
        wavlm_model = WavLMModel(wavlm_config)
    roberta_model = RobertaModel.from_pretrained('roberta-base')
    
    model = SpeechTextModel(wavlm_model, roberta_model, num_layers=args.num_layers, common_model=args.common_model, use_conv=args.use_conv, pool_fn=args.pool_fn)
    emo_model = EmotionClassifier(model, 3)

    if args.supervised == "True":
        checkpoint = torch.load("text_whisper_supervised/text_supervised.tar", map_location = device)
        emo_model.load_state_dict(checkpoint['model_state_dict'])
    # emo_model = torch.load("models_dual_l1/model-45000.pth", map_location=device)
    emo_model = emo_model.to(device)
    dual_params, common_params = [], []
    for name, param in emo_model.named_parameters():
        if "upblock" in name or "downblock" in name:
            param.requires_grad = True
            common_params.append(param)
        elif "audio_model" in name:
            param.requires_grad = True
            common_params.append(param)
        elif "dual_model" in name:
            param.requires_grad = True
            dual_params.append(param)
        elif "joint_model.linear" in name:
            param.requires_grad = True
            dual_params.append(param)
        # elif "common_model" in name:
        #     param.requires_grad = True
        #     common_params.append(param)
        else:
            print(name)
            param.requires_grad = False
    dual_optimizer = AdamW(
            dual_params,
            lr=LEARNING_RATE,
            betas=BETAS,
            eps=EPS,
            weight_decay=WEIGHT_DECAY,
        )
    common_optimizer = AdamW(
            common_params,
            lr=LEARNING_RATE,
            betas=BETAS,
            eps=EPS,
            weight_decay=WEIGHT_DECAY,
        )

    n_epochs = STEPS // len(train_loader) + 1
    wt = args.alpha
    num_steps = 0
    train_audio_masked_loss = 0
    train_audio_unmasked_loss = 0
    train_speechtext_loss = 0
    train_opensmile_loss = 0
    best_valid_loss = 99999999999
    # logger.info("**" * 40)
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"CUDNN version: {torch.backends.cudnn.version()}")
    logger.info(f"CUDNN enabled: {torch.backends.cudnn.enabled}")
    logger.info(f"CUDNN deterministic: {torch.backends.cudnn.deterministic}")
    logger.info(f"CUDNN benchmark: {torch.backends.cudnn.benchmark}")
    logger.info(f"# of GPUS: {torch.cuda.device_count()}")
    logger.info(f"batch size: {BATCH_SIZE}")
    logger.info(f"iterations per epoch: {len(train_loader)}")
    logger.info(f"# of epochs: {n_epochs}")
    logger.info("**" * 40 + "\n")
    accum_steps = 1
    for epoch in range(n_epochs + 1):
        emo_model.train()
        for i, data in enumerate(train_loader):
            num_steps += 1
            wavs, opensmile_feats, roberta_feats, roberta_logits = data
            out = emo_model(wavs.to(device))
            opensmile_feats = opensmile_feats[:, :249, :].to(device)
            speech_feats, pooled_audio = out
            
            
            if len(roberta_feats.shape) == 3:
                roberta_feats = roberta_feats[:, -1, :].to(device)
                roberta_feats = roberta_feats.squeeze(1)
            else:
                roberta_feats = roberta_feats.to(device)

            distil_loss = nn.MSELoss()(roberta_feats, pooled_audio)
            opensmile_loss = nn.MSELoss()(speech_feats, opensmile_feats)         

            train_speechtext_loss += distil_loss.item()
            train_opensmile_loss += opensmile_loss.item()

            distil_loss = distil_loss/accum_steps
            opensmile_loss = opensmile_loss/accum_steps
            distil_loss.backward(retain_graph = True)
            
            hook_handles = []
            for param in common_params:
                handle = param.register_hook(zero_grad_hook)
                hook_handles.append(handle)


            opensmile_loss.backward()

            for handle in hook_handles:
                handle.remove()            

            if ((i + 1) % accum_steps == 0) or (i + 1 == len(train_loader)):
                common_optimizer.step()
                dual_optimizer.step()
                common_optimizer.zero_grad()
                dual_optimizer.zero_grad()


            if num_steps % 10000 == 0:
                torch.save(emo_model, os.path.join(args.checkpoint_dir, "model-"+str(num_steps)+".pth"))
                train_speechtext_loss_n = train_speechtext_loss/num_steps
                train_opensmile_loss_n = train_opensmile_loss/num_steps
                logger.info("*"*40)
                logger.info(f"Step: {num_steps}")
                # logger.info(f"Audio Masked Loss: {train_audio_masked_loss_n}")
                # logger.info(f"Audio Unmasked Loss: {train_audio_unmasked_loss_n}")
                logger.info(f"Speech Text Distillation Loss: {train_speechtext_loss_n}")
                logger.info(f"Speech Opensmile Loss: {train_opensmile_loss_n}")
                logger.info("*"*40)
            
            if num_steps % 5000 == 0:
                with torch.no_grad():
                    emo_model.eval()
                    valid_audio_loss = 0
                    valid_speechtext_loss = 0
                    valid_opensmile_loss = 0
                    valid_steps = 0
                    for i, data in enumerate(valid_loader):
                        valid_steps += 1
                        wavs, opensmile_feats, roberta_feats, roberta_logits = data
                        out = emo_model(wavs.to(device))
                        opensmile_feats = opensmile_feats[:, :249, :].to(device)
                        speech_feats, pooled_audio = out
                        if len(roberta_feats.shape) == 3:
                            roberta_feats = roberta_feats[:, -1, :].to(device)
                            roberta_feats = roberta_feats.squeeze(1)
                        else:
                            roberta_feats = roberta_feats.to(device)
                        distil_loss = nn.MSELoss()(roberta_feats, pooled_audio)
                        opensmile_loss = nn.MSELoss()(speech_feats, opensmile_feats)   
            

                        audio_loss = distil_loss + opensmile_loss
                        valid_speechtext_loss += distil_loss.item()
                        valid_opensmile_loss += opensmile_loss.item()

                    total_valid_loss = valid_speechtext_loss + valid_opensmile_loss
                    total_valid_loss = total_valid_loss / valid_steps
                    if total_valid_loss < best_valid_loss:
                        torch.save(emo_model, os.path.join(args.checkpoint_dir, "model-"+str(num_steps)+".pth"))
                        # valid_audio_loss_n = valid_audio_loss/valid_steps   
                        valid_speechtext_loss_n = valid_speechtext_loss/valid_steps  
                        valid_opensmile_loss_n = valid_opensmile_loss/valid_steps                 
                        logger.info("*"*40)
                        logger.info(f"Step: {num_steps}")
                        # logger.info(f"Val Audio Loss: {valid_audio_loss_n}")
                        logger.info(f"Val Distillation Loss: {valid_speechtext_loss_n}")
                        logger.info(f"Val Opensmile Loss: {valid_opensmile_loss_n}")
                        logger.info("*"*40)
                        best_valid_loss = total_valid_loss
            emo_model.train()

if __name__ == "__main__":
    train(args)
