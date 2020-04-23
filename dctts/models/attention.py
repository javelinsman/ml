import torch
from .config import DIMENSIONS as D

def mix_input(text_encoded_att, text_encoded_chr, audio_encoded):
    attention = torch.matmul(text_encoded_att.permute(0, 2, 1), audio_encoded)
    attention = torch.softmax(attention / D['latent'] ** 0.5, axis=1)
    mixed_input = torch.matmul(text_encoded_chr, attention)
#         input_to_decoder = torch.cat([mixed_input, audio_encoded], axis=1)
    input_to_decoder = mixed_input
    return input_to_decoder, attention
        