import os
import json
import numpy as np
from collections import defaultdict
from util.preprocess import calc_spectrograms
from util.g2p import grapheme_to_phoneme, convert_to_indices
from sklearn.model_selection import train_test_split
import pathlib
import librosa
import torch

class TTSData:
    def __init__(self, path_to_wav, sentence):
        self.path_to_wav = path_to_wav
        self.sentence = sentence
        self._melspectrogram = None
        self._char_sequence = None
        self._char_sequence_len = None
        self._melspectrogram = None

    def __repr__(self):
        return f'TTSData(sentence={self.sentence})'

    def calc_melspectrogram(self):
        waveform, sampling_rate = librosa.load(self.path_to_wav)
        return calc_spectrograms(waveform, sampling_rate)['mel_norm']

    @property
    def melspectrogram(self):
        if self._melspectrogram is None:
            cache_path = pathlib.Path(os.path.join(
                os.path.dirname(__file__), 'data/cache/',
                self.path_to_wav.replace('/', '-').replace('.', '')
            ))
            if cache_path.exists():
                with open(cache_path) as f:
                    mel = np.array(json.loads(f.read()), dtype=np.float32)
            else:
                mel = self.calc_melspectrogram()
                cache_path.parent.mkdir(exist_ok=True, parents=True)
                with open(cache_path, 'w') as f:
                    f.write(json.dumps(mel.tolist()))
            self._melspectrogram = mel
        return self._melspectrogram

    @property
    def char_sequence(self):
        if self._char_sequence is None:
            self._char_sequence = \
                convert_to_indices(grapheme_to_phoneme(self.sentence))
        return self._char_sequence

    @property
    def char_sequence_len(self):
        if self._char_sequence_len is None:
            self._char_sequence_len = len(self.char_sequence)
        return self._char_sequence_len
        
class KSSDataset:
    def __init__(self, train=True, section=2):
        self.section = section
        self.train = train
        self.seeds = self.__make_seeds()
        self.data = [TTSData(*seed) for seed in self.seeds]
        self.data.sort(key=lambda data: data.char_sequence_len)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

    def __make_seeds(self):
        base_path = os.path.join(os.path.dirname(__file__), 'data')
        transcript_path = os.path.join(base_path, 'transcript.v.1.4.txt')
        seeds = []
        with open(transcript_path, encoding='utf-8') as f:
            for line in f:
                args = line.split('|')
                if self.section != 'all' and not args[0].startswith(f'{self.section}/'):
                    continue
                seeds.append((
                    os.path.join(base_path, 'kss', args[0]),
                    args[2]  # sentence
                ))
        train_seeds, test_seeds = train_test_split(seeds, test_size=0.2,
                                                   random_state=100)
        if self.train:
            return train_seeds
        else:
            return test_seeds

class TTSDataLoader:
    def __init__(self, dataset, batch_size=64):
        self.dataset = dataset
        self.batch_size = batch_size

    def __getitem__(self, index):
        l = index * self.batch_size
        r = (index + 1) * self.batch_size
        return self.__postprocess(self.dataset[l:r])

    def __len__(self):
        n = len(self.dataset)
        return 1 + (n - 1) // self.batch_size

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __postprocess(self, batch):
        max_n = np.max([data.char_sequence.shape[0] for data in batch])
        padded_sentences = np.array([
            np.pad(data.char_sequence,
                   ((0, max_n - data.char_sequence.shape[0]),))
            for data in batch
        ])

        max_t = np.max([data.melspectrogram.shape[1] for data in batch])
        padded_melspectrograms = np.array([
            np.pad(data.melspectrogram,
                   ((0, 0), (1, max_t - data.melspectrogram.shape[1])))
            for data in batch
        ])
        mel_left = padded_melspectrograms[:,:,:max_t]
        mel_right = padded_melspectrograms[:,:,1:]
        return [torch.tensor(mel_left), torch.tensor(padded_sentences)], torch.tensor(mel_right)



        

# def make_melspectogram_pairs(batch):
#     melspectrograms = [
#         calc_spectrograms(data['waveform'], data['sampling_rate'])['mel_norm']
#         for data in batch
#     ]
#     longest = np.max([melspectrogram.shape[1] for melspectrogram in melspectrograms])


# def make_padded_char_sequences(batch):
#     sentences = [data['text'] for data in batch]
#     sentences = [
#         convert_to_indices(graphemes_to_phonemes(sentence))
#         for sentence in sentences
#     ]

# def process(batch):
    