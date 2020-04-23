import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from util.layers import CustomConv1d, HighwayConv1d
from datasets.dataset import KSSDataset, TTSDataLoader
from .config import DIMENSIONS as D

from .audio_decoder import AudioDecoder
from .text_encoder import TextEncoder
from .audio_encoder import AudioEncoder

class TTSModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.audio_encoder = AudioEncoder()
        self.text_encoder = TextEncoder()
        self.audio_decoder = AudioDecoder()
        
    def mix_input(self, text_encoded_att, text_encoded_chr, audio_encoded):
        attention = torch.matmul(text_encoded_att.permute(0, 2, 1), audio_encoded)
        attention = torch.softmax(attention / D['latent'] ** 0.5, axis=1)
        mixed_input = torch.matmul(text_encoded_chr, attention)
#         input_to_decoder = torch.cat([mixed_input, audio_encoded], axis=1)
        input_to_decoder = mixed_input
        return input_to_decoder, attention
        
    def forward(self, inputs):
        audio_input, text_input = inputs
        audio_encoded = self.audio_encoder(audio_input)
        text_encoded_att, text_encoded_chr = self.text_encoder(text_input)
        input_to_decoder, attention = self.mix_input(text_encoded_att, text_encoded_chr, audio_encoded)
        audio_decoded = self.audio_decoder(input_to_decoder)
        return audio_decoded
    
    def __calc_losses(self, batch):
        (audio_input, text_input), audio_target = batch
        audio_encoded = self.audio_encoder(audio_input)
        text_encoded_att, text_encoded_chr = self.text_encoder(text_input)
        input_to_decoder, attention = self.mix_input(text_encoded_att, text_encoded_chr, audio_encoded)
        audio_decoded = self.audio_decoder(input_to_decoder)
        N = attention.size(1)
        T = attention.size(2)
        ts = torch.arange(T, dtype=torch.double).repeat(N).view(N, T)
        ns = torch.arange(N, dtype=torch.double).repeat(T).view(T, N).T
        attention_guide = 1 - torch.exp(-(ns / N - ts / T) ** 2 / (2 * 0.2 ** 2))
        att_loss = torch.mean(attention * attention_guide.to(attention.device))
        bce_loss = nn.BCELoss()(audio_decoded, audio_target)
        l1_loss = nn.L1Loss()(audio_decoded, audio_target)
        return att_loss, bce_loss, l1_loss
    
    def training_step(self, batch, batch_idx):
        att_loss, bce_loss, l1_loss = self.__calc_losses(batch)
        loss = att_loss + bce_loss + l1_loss
        return {'loss': loss, 'att': att_loss, 'bce': bce_loss, 'l1': l1_loss}
    
    def validation_step(self, batch, batch_idx):
        att_loss, bce_loss, l1_loss = self.__calc_losses(batch)
        loss = att_loss + bce_loss + l1_loss
        return {'val_loss': loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_att = torch.stack([x['att'] for x in outputs]).mean()
        avg_bce = torch.stack([x['bce'] for x in outputs]).mean()
        avg_l1 = torch.stack([x['l1'] for x in outputs]).mean()
        
        logs = {'loss': avg_loss, 'att': avg_att, 'bce': avg_bce, 'l1': avg_l1}
        results = {'log': logs}
        return results
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        logs = {'val_loss': avg_loss}
        results = {'log': logs}
        return results
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
    
    def prepare_data(self):
        self.train_set = KSSDataset(train=True)
        self.val_set = KSSDataset(train=False)
    
    def train_dataloader(self):
        return TTSDataLoader(self.train_set, batch_size=32)
    
    def val_dataloader(self):
        return TTSDataLoader(self.val_set, batch_size=32)