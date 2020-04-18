import os
import numpy as np
from .KoG2P.g2p import runKoG2P

path_to_rulebook = os.path.join(
    os.path.dirname(__file__),
    'KoG2P/rulebook.txt'
)

phoneme_types = [
    'empty', '--', 'p0', 'ph', 'pp', 't0', 'th', 'tt', 'k0', 'kh',
    'kk', 's0', 'ss', 'h0', 'c0', 'ch', 'cc', 'mm', 'nn', 'rr', 'pf',
    'ph', 'tf', 'th', 'kf', 'kh', 'kk', 's0', 'ss', 'h0', 'c0', 'ch',
    'mf', 'nf', 'ng', 'll', 'ks', 'nc', 'nh', 'lk', 'lm', 'lb', 'ls',
    'lt', 'lp', 'lh', 'ps', 'ii', 'ee', 'qq', 'aa', 'xx', 'vv', 'uu',
    'oo', 'ye', 'yq', 'ya', 'yv', 'yu', 'yo', 'wi', 'wo', 'wq', 'we',
    'wa', 'wv', 'xi'
]

def grapheme_to_phoneme(text):
    words = text.split()
    phonemes_list = [''.join(runKoG2P(word, path_to_rulebook)) for word in words]
    phonemes_string = ' -- '.join(phonemes_list)
    return phonemes_string.split()

def convert_to_indices(phonemes):
    return np.array([
        phoneme_types.index(phoneme) for phoneme in phonemes
    ])