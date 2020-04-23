import os
import random
import torch
import torch.nn.functional as F
import librosa
import numpy as np

from datasets.dataset import KSSDataset, TTSDataLoader, TTSData
from util.layers import CustomConv1d, HighwayConv1d
from util.preprocess import calc_spectrograms


train_set = KSSDataset(train=True)
val_set = KSSDataset(train=False)
train_loader = TTSDataLoader(train_set, batch_size=32)
val_loader = TTSDataLoader(val_set, batch_size=32)


def test_mel_shift():
    batch = random.choice(train_loader)
    (mels_left, _), mels_right = batch
    assert torch.all(mels_left[:,:,1:] == mels_right[:,:,:-1]).item() == True
    assert torch.all(mels_left[:,:,0] == 0).item() == True

def test_mel_sync():
    wav_path = os.path.join(
        os.path.dirname(__file__),
        '../datasets/data/kss/1/1_0001.wav'
    )
    mel_calc = calc_spectrograms(*librosa.load(wav_path))['mel_norm']
    mel_cache = TTSData(wav_path, '').melspectrogram
    assert np.linalg.norm(mel_calc - mel_cache) < 1e-6

def test_custom_conv1d():
    batch = F.pad(torch.rand(100, 5, 3), (1, 0))
    conv_same = CustomConv1d(5, 10, 5, 1, 'same')
    conv_causal = CustomConv1d(5, 10, 5, 1, 'causal')
    assert torch.any(conv_same(batch)[:,:,0] - conv_same.conv.bias > 0).item() == True
    assert torch.any(conv_causal(batch)[:,:,0] - conv_causal.conv.bias > 0).item() == False
