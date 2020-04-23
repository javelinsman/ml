from util.g2p import phoneme_types
from util.g2p import grapheme_to_phoneme, convert_to_indices
from util.layers import CustomConv1d, HighwayConv1d
from tts.dataset import KSSDataset, TTSDataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

D = {
    'latent': 256,
    'F': 80,
    'F_': 513,
    'c': 512,
    'embedding': 128
}

class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        d = D['latent']
        self.layers = nn.ModuleList([
            CustomConv1d(D['F'], d, 1, dilation=1), nn.ReLU(),
            CustomConv1d(d, d, 1, dilation=1), nn.ReLU(),
            CustomConv1d(d, d, 1, dilation=1),

            HighwayConv1d(d, 3, 1, padding_mode='causal'),
            HighwayConv1d(d, 3, 3, padding_mode='causal'),
            HighwayConv1d(d, 3, 9, padding_mode='causal'),
            HighwayConv1d(d, 3, 27, padding_mode='causal'),
            HighwayConv1d(d, 3, 1, padding_mode='causal'),
            HighwayConv1d(d, 3, 3, padding_mode='causal'),
            HighwayConv1d(d, 3, 9, padding_mode='causal'),
            HighwayConv1d(d, 3, 27, padding_mode='causal'),
            HighwayConv1d(d, 3, 3, padding_mode='causal'),
            HighwayConv1d(d, 3, 3, padding_mode='causal'),
        ])
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        for layer in self.layers[:1]:
            x = layer(x)
        return x
            
class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        d = D['latent']
        e = D['embedding']
        
        self.embedding = nn.Embedding(len(phoneme_types), e)
        self.layers = nn.ModuleList([
            
            CustomConv1d(e, 2 * d, 1, 1), nn.ReLU(),
            CustomConv1d(2 * d, 2 * d, 1, 1),

            HighwayConv1d(2 * d, 3, 1),
            HighwayConv1d(2 * d, 3, 3),
            HighwayConv1d(2 * d, 3, 9),
            HighwayConv1d(2 * d, 3, 27),
            HighwayConv1d(2 * d, 3, 1),
            HighwayConv1d(2 * d, 3, 3),
            HighwayConv1d(2 * d, 3, 9),
            HighwayConv1d(2 * d, 3, 27),
            HighwayConv1d(2 * d, 3, 1),
            HighwayConv1d(2 * d, 3, 1),
            HighwayConv1d(2 * d, 1, 1),
            HighwayConv1d(2 * d, 1, 1),
        ])
    
    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        for layer in self.layers[:1]:
            x = layer(x)
        return x[:,:D['latent'],:], x[:,D['latent']:,:]
    


class AudioDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        d = D['latent']
        self.layers = nn.ModuleList([
#             CustomConv1d(2 * d, d, 1, 1, 'causal'),
            CustomConv1d(d, d, 1, 1, 'causal'),
            HighwayConv1d(d, 3, 1, padding_mode='causal'),
            HighwayConv1d(d, 3, 3, padding_mode='causal'),
            HighwayConv1d(d, 3, 9, padding_mode='causal'),
            HighwayConv1d(d, 3, 27, padding_mode='causal'),
            HighwayConv1d(d, 3, 1, padding_mode='causal'),
            HighwayConv1d(d, 3, 1, padding_mode='causal'),
            CustomConv1d(d, d, 1, 1, 'causal'), nn.ReLU(),
            CustomConv1d(d, d, 1, 1, 'causal'), nn.ReLU(),
            CustomConv1d(d, d, 1, 1, 'causal'), nn.ReLU(),
            CustomConv1d(d, D['F'], 1, 1, 'causal'), nn.Sigmoid(),
        ])
        
    def forward(self, x):
        for layer in [self.layers[0], self.layers[-2], self.layers[-1]]:
            x = layer(x)
        return x.permute(0, 2, 1)

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
        pass
#         self.train_set = KSSDataset(train=True)
#         self.val_set = KSSDataset(train=False)
    
    def train_dataloader(self):
        return train_loader
#         return TTSDataLoader(self.train_set, batch_size=32)
    
    def val_dataloader(self):
        return val_loader
#         return TTSDataLoader(self.val_set, batch_size=32)