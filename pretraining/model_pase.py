import copy
import random
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn import LayerNorm
from transformers import AutoTokenizer
from typing import List, Optional, Tuple
from transformers import WavLMModel, RobertaModel
import numpy as np
    
class SpeechTextModel(nn.Module):
    def __init__(self, wavlm_model,
                 roberta_model,
                 mlm_proba=0.15,
                 num_layers=6,
                 common_model="roberta",
                 use_conv=False,
                 pool_fn="avg"):
        super().__init__()
        self.common_model_name = common_model
        self.audio_feature_extractor = nn.Sequential(*wavlm_model.feature_extractor.conv_layers)
        self.feature_projection_audio = wavlm_model.feature_projection
        self.pos_conv = wavlm_model.encoder.pos_conv_embed
        self.wavlm_layer_norm = wavlm_model.encoder.layer_norm
        self.wavlm_dropout = nn.Dropout(0.1, inplace=False)
        self.audio_model = nn.ModuleList([wavlm_model.encoder.layers[i] for i in range(num_layers)])
        self.dual_model = nn.ModuleList([wavlm_model.encoder.layers[i] for i in range(num_layers, 12)])
        self.position_bias = None
        self.linear = nn.Linear(768, 256)
        
        self.roberta_embeddings = roberta_model.embeddings
        self.text_model = nn.ModuleList([roberta_model.encoder.layer[i] for i in range(num_layers)])
        self.mlm_probability = mlm_proba
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        
        if common_model == "roberta":
            self.common_model = nn.ModuleList([roberta_model.encoder.layer[i] for i in range(6, 12)])
        self.use_conv = use_conv
        if use_conv == "True":
            self.downblock = nn.ModuleList([])
            self.upblock = nn.ModuleList([])
            for i in range(6):
                self.downblock.append(nn.Conv1d(in_channels=768,
                                          out_channels=768,
                                          kernel_size=5,
                                          stride=3))
                self.upblock.append(nn.ConvTranspose1d(in_channels=768,
                                                 out_channels=768,
                                                 kernel_size=5,
                                                 stride=3,
                                                 output_padding=1))         


        # self.decoder = nn.Linear(768, 50265)
        self.act = nn.GELU()
        if pool_fn == "atten":
            self.pool_fn = "atten"
            self.attenpool = SelfAttentionPooling(768)
        else:
            self.pool_fn = "avg"

        self.mask_length: int = 10     # mask length
        self.mask_prob: float = 0.65     # probability of replacing a token with mask
        self.mask_selection: str = "static"     # how to choose mask length
        self.mask_other: float = 0     # secondary mask argument (used for more complex distributions), see help in compute_mask_indicesh
        self.no_mask_overlap: bool = False     # whether to allow masks to overlap
        self.mask_min_space: int = 1     # min space between spans (if no overlap is enabled)

    def extract_audio_features(self, audio):
        out_layers, out_layers_aud = [], []
        audio = audio.unsqueeze(1)
        audio_features = self.audio_feature_extractor(audio)
        audio_features = audio_features.permute(0, 2, 1)
        audio_features = self.feature_projection_audio(audio_features)[0]
        audio_features = audio_features.permute(0, 2, 1)
        audio_features_conv = self.pos_conv(audio_features.transpose(1,2))
        audio_features_conv = audio_features_conv.transpose(1, 2)
        audio_features = audio_features + audio_features_conv
        audio_features = self.wavlm_layer_norm(audio_features.transpose(1,2))
        audio_features = self.wavlm_dropout(audio_features)
        x = audio_features
        speech_feats = x
        out_layers.append(speech_feats)
        out_layers_aud.append(speech_feats)
        for i, layer in enumerate(self.audio_model):
            if i != 0:
                speech_feats, position_bias = layer(speech_feats, position_bias=position_bias)
            else:
                speech_feats, position_bias = layer(speech_feats)
            out_layers.append(speech_feats)
            out_layers_aud.append(speech_feats)
        inp = speech_feats

        speech_only_feats = speech_feats
        for i, layer in enumerate(self.dual_model):
            speech_only_feats, position_bias = layer(speech_only_feats, position_bias=position_bias)
            out_layers_aud.append(speech_only_feats)
        speech_only_feats = self.linear(speech_only_feats)
        fusion_out_aud = torch.stack(out_layers_aud, dim = 0)
        if self.common_model_name =="roberta":
            x = inp
            for i, layer in enumerate(self.common_model):
                if self.use_conv == "True":
                    x = x.permute(0, 2, 1)
                    x = self.downblock[i](x)
                    x = x.permute(0, 2, 1)
                x = layer(x)[0] 
                if self.use_conv == "True":
                    x = x.permute(0, 2, 1)
                    x = self.upblock[i](x)
                    x = x.permute(0, 2, 1)
                if x.shape[1] < inp.shape[1]:
                    x = torch.cat((x, torch.zeros(x.shape[0],inp.shape[1]-x.shape[1] , x.shape[-1]).to(x.device)), 1)
                else:
                    x = x[:, :inp.shape[1], :]
                out_layers.append(x)
            fusion_out = torch.stack(out_layers, dim = 0)

        fusion_feats_audio = x
        if self.pool_fn == "avg":
            pooled_audio = torch.mean(fusion_feats_audio, 1).squeeze(1)
        else:
            pooled_audio = self.attenpool(fusion_feats_audio)
        return fusion_out, pooled_audio, fusion_out_aud

    def extract_text_features(self, text):
        text_inputs = text['input_ids']
        attention_mask = text['attention_mask']
        embedding_output = self.roberta_embeddings(
            input_ids=text['input_ids'],
        )
        out_layers = []
        text_feats = embedding_output
        out_layers.append(embedding_output)

        for layer in self.text_model:
            text_feats = layer(text_feats)[0] 
            out_layers.append(text_feats)

        inp = text_feats
        if self.common_model_name =="roberta":
            x = inp
            for layer in self.common_model:
                x = layer(x)[0] 
                out_layers.append(x)
            fusion_out = torch.stack(out_layers, dim = 0)
        else:
            out_layers = torch.stack(out_layers, dim = 0)
            fusion_out = self.common_model(inp)
            fusion_out = torch.cat((out_layers, fusion_out), dim=0)
        if self.pool_fn == "avg":
            pooled_text = torch.mean(x, 1).squeeze(1)
        else:
            pooled_text = self.attenpool(x)
        return fusion_out, pooled_text
        

    def forward(self, audio, padding_mask=None, mask=False, mode="speech", weights=None):
        out_layers, out_layers_aud = [], []
        if mode == "speech" or mode == "speech_text":
            audio_features = self.audio_feature_extractor(audio)
            audio_features = audio_features.permute(0, 2, 1)
            audio_features = self.feature_projection_audio(audio_features)[0]
            audio_features = audio_features.permute(0, 2, 1)
            audio_features_conv = self.pos_conv(audio_features.transpose(1,2))
            audio_features_conv = audio_features_conv.transpose(1, 2)
            audio_features = audio_features + audio_features_conv
            audio_features = self.wavlm_layer_norm(audio_features.transpose(1,2))
            audio_features = self.wavlm_dropout(audio_features)
            if mask:
                x, mask_indices = self.apply_mask(
                    audio_features, padding_mask
                )
            else:
                x = audio_features
                mask_indices = None
            speech_feats = x
            out_layers.append(speech_feats)
            out_layers_aud.append(speech_feats)
            for i, layer in enumerate(self.audio_model):
                if i != 0:
                    speech_feats, position_bias = layer(speech_feats, position_bias=position_bias)
                else:
                    speech_feats, position_bias = layer(speech_feats)
                out_layers.append(speech_feats)
                out_layers_aud.append(speech_feats)
            inp = speech_feats
            out_6 = inp
            speech_only_feats = speech_feats
            for i, layer in enumerate(self.dual_model):
                speech_only_feats, position_bias = layer(speech_only_feats, position_bias=position_bias)
                out_layers_aud.append(speech_only_feats)
            speech_only_feats_os = self.linear(speech_only_feats)
        if self.common_model_name =="roberta":
            x = inp
            for i, layer in enumerate(self.common_model):
                if self.use_conv == "True" and mode =="speech":
                    x = x.permute(0, 2, 1)
                    x = self.downblock[i](x)
                    x = x.permute(0, 2, 1)
                x = layer(x)[0] 
                if self.use_conv == "True" and mode =="speech":
                    x = x.permute(0, 2, 1)
                    x = self.upblock[i](x)
                    x = x.permute(0, 2, 1)
                out_layers.append(x)
            fusion_out = x
        else:
            fusion_out = self.common_model(inp)[-1]

        if mode == "speech":
            fusion_feats_audio = fusion_out
            if self.pool_fn == "avg":
                if weights == None:
                    pooled_audio = torch.mean(fusion_feats_audio, 1).squeeze(1)
                else:
                    pooled_audio = torch.matmul(weights, fusion_feats_audio)
            else:
                pooled_audio = self.attenpool(fusion_feats_audio)
        fusion_out = torch.stack(out_layers, dim = 0)
        fusion_out_aud = torch.stack(out_layers_aud, dim = 0)
            

        return speech_only_feats_os, pooled_audio, fusion_out, fusion_out_aud


