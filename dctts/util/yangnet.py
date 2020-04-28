import torch
import torch.nn as nn
from torch.nn.utils import weight_norm as norm
from util.g2p import phoneme_types
import numpy as np
import util.yangyang as mm

class ConfigArgs:
    data_path = '/home/yangyangii/ssd/data/LJSpeech-1.1'
    mel_dir, mag_dir = 'd_mels', 'd_mags'
    ga_dir = 'guides' # guided attention
    meta = 'metadata.csv'
    meta_train = 'meta-train.csv'
    meta_eval = 'meta-eval.csv'
    testset = 'test_sents.txt'
    logdir = 'logs'
    sampledir = 'samples'
    prepro = True
    mem_mode= True
    ga_mode = True
    log_mode = True
    save_term = 1000
    n_workers = 8
    n_gpu = 2
    global_step = 0

    sr = 22050 # sampling rate
    preemph = 0.97 # pre-emphasize
    n_fft = 2048
    n_mags = n_fft//2 + 1
    n_mels = 80
    frame_shift = 0.0125
    frame_length = 0.05
    hop_length = int(sr*frame_shift)
    win_length = int(sr*frame_length)
    gl_iter = 50 # Griffin-Lim iteration
    max_db = 100
    ref_db = 20
    power = 1.5
    r = 4  # reduction factor. mel/4
    g = 0.2

    batch_size = 32
    test_batch = 50 # for test
    max_step = 200000
    lr = 0.001
    lr_decay_step = 50000 # actually not decayed per this step
    Ce = 128  # for text embedding and encoding
    Cx = 256 # for text embedding and encoding
    Cy = 256 # for audio encoding
    Cs = 512 # for SSRN
    drop_rate = 0.05

    max_Tx = 188
    max_Ty = 250

    vocab = u'''PE !',-.?abcdefghijklmnopqrstuvwxyz'''

args = ConfigArgs

class TextEncoder(nn.Module):
    """
    Text Encoder
        T: (N, Cx, Tx) Text embedding (variable length)
    Returns:
        K: (N, Cx, Tx) Text Encoding for Key
        V: (N, Cx, Tx) Text Encoding for Value
    """
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.hc_blocks = nn.ModuleList([norm(mm.Conv1d(args.Ce, args.Cx*2, 1, padding='same', activation_fn=torch.relu))])  # filter up to split into K, V
        self.hc_blocks.extend([norm(mm.Conv1d(args.Cx*2, args.Cx*2, 1, padding='same', activation_fn=None))])
        self.hc_blocks.extend([norm(mm.HighwayConv1d(args.Cx*2, args.Cx*2, 3, dilation=3**i, padding='same'))
                               for _ in range(2) for i in range(4)])
        self.hc_blocks.extend([norm(mm.HighwayConv1d(args.Cx*2, args.Cx*2, 3, dilation=1, padding='same'))
                               for i in range(2)])
        self.hc_blocks.extend([norm(mm.HighwayConv1d(args.Cx*2, args.Cx*2, 1, dilation=1, padding='same'))
                               for i in range(2)])
        self.embedding = nn.Embedding(len(phoneme_types), args.Ce)

    def forward(self, L):
        y = self.embedding(L).permute(0, 2, 1)
        for i in range(len(self.hc_blocks)):
            y = self.hc_blocks[i](y)
        K, V = y.chunk(2, dim=1)  # half size for axis Cx
        return K, V

class AudioEncoder(nn.Module):
    """
    Text Encoder
        prev_audio: (N, n_mels, Ty/r) Mel-spectrogram (variable length)
    Returns:
        Q: (N, Cx, Ty/r) Audio Encoding for Query
    """

    def __init__(self):
        super(AudioEncoder, self).__init__()
        self.hc_blocks = nn.ModuleList([norm(mm.CausalConv1d(args.n_mels, args.Cx, 1, activation_fn=torch.relu))])
        self.hc_blocks.extend([norm(mm.CausalConv1d(args.Cx, args.Cx, 1, activation_fn=torch.relu))
                               for _ in range(2)])
        self.hc_blocks.extend([norm(mm.CausalHighwayConv1d(args.Cx, args.Cx, 3, dilation=3**i)) # i is in [[0,1,2,3],[0,1,2,3]]
                               for _ in range(2) for i in range(4)])
        self.hc_blocks.extend([norm(mm.CausalHighwayConv1d(args.Cx, args.Cx, 3, dilation=3))
                               for i in range(2)])
        # self.hc_blocks.extend([mm.CausalConv1d(args.Cy, args.Cx, 1, dilation=1, activation_fn=torch.relu)]) # down #filters to dotproduct K, V

    def forward(self, S):
        Q = S
        for i in range(len(self.hc_blocks)):
            Q = self.hc_blocks[i](Q)
        return Q

class DotProductAttention(nn.Module):
    """
    Dot Product Attention
    Args:
        K: (N, Cx, Tx)
        V: (N, Cx, Tx)
        Q: (N, Cx, Ty)
    Returns:
        R: (N, Cx, Ty)
        A: (N, Tx, Ty) alignments
    """

    def __init__(self):
        super(DotProductAttention, self).__init__()

    def forward(self, K, V, Q):
        A = torch.softmax((torch.bmm(K.transpose(1, 2), Q)/np.sqrt(args.Cx)), dim=1) # K.T.dot(Q) -> (N, Tx, Ty)
        R = torch.bmm(V, A) # (N, Cx, Ty)
        return R, A

class AudioDecoder(nn.Module):
    """
    Dot Product Attention
    Args:
        R_: (N, Cx*2, Ty)
    Returns:
        O: (N, n_mels, Ty)
    """
    def __init__(self):
        super(AudioDecoder, self).__init__()
        self.hc_blocks = nn.ModuleList([norm(mm.CausalConv1d(args.Cx*2, args.Cy, 1, activation_fn=torch.relu))])
        # self.hc_blocks = nn.ModuleList([norm(mm.CausalConv1d(args.Cx, args.Cy, 1, activation_fn=torch.relu))])
        self.hc_blocks.extend([norm(mm.CausalHighwayConv1d(args.Cy, args.Cy, 3, dilation=3**i))
                               for i in range(4)])
        self.hc_blocks.extend([norm(mm.CausalHighwayConv1d(args.Cy, args.Cy, 3, dilation=1))
                               for _ in range(2)])
        self.hc_blocks.extend([norm(mm.CausalConv1d(args.Cy, args.Cy, 1, dilation=1, activation_fn=torch.relu))
                               for _ in range(3)])
        self.hc_blocks.extend([norm(mm.CausalConv1d(args.Cy, args.n_mels, 1, dilation=1))]) # down #filters to dotproduct K, V

    def forward(self, R_):
        Y = R_
        for i in range(len(self.hc_blocks)):
            Y = self.hc_blocks[i](Y)
        return torch.sigmoid(Y)

class Text2Mel(nn.Module):
    """
    Text2Mel
    Args:
        L: (N, Tx) text
        S: (N, Ty/r, n_mels) previous audio
    Returns:
        Y: (N, Ty/r, n_mels)
    """
    def __init__(self):
        super(Text2Mel, self).__init__()
        self.name = 'Text2Mel'
        self.embed = nn.Embedding(len(args.vocab), args.Ce, padding_idx=0)
        self.TextEnc = TextEncoder()
        self.AudioEnc = AudioEncoder()
        self.Attention = DotProductAttention()
        self.AudioDec = AudioDecoder()
    
    def forward(self, S, L):
        L = self.embed(L).transpose(1,2) # -> (N, Cx, Tx) for conv1d
        S = S.transpose(1,2) # (N, n_mels, Ty/r) for conv1d
        K, V = self.TextEnc(L) # (N, Cx, Tx) respectively
        Q = self.AudioEnc(S) # -> (N, Cx, Ty/r)
        R, A = self.Attention(K, V, Q) # -> (N, Cx, Ty/r)
        R_ = torch.cat((R, Q), 1) # -> (N, Cx*2, Ty/r)
        Y = self.AudioDec(R_) # -> (N, n_mels, Ty/r)
        return Y.transpose(1, 2), A # (N, Ty/r, n_mels)