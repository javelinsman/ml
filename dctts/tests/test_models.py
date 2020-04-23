import torch
from models.audio_encoder import AudioEncoder
from models.text_encoder import TextEncoder
from models.audio_decoder import AudioDecoder
from models.tts_model import TTSModel
from models.config import DIMENSIONS as D

F = D['F']
d = D['latent']
e = D['embedding']

def test_audio_encoder():
    audio_encoder = AudioEncoder()
    assert audio_encoder(torch.rand(32, F, 17)).size() == (32, d, 17)

def test_text_encoder():
    text_encoder = TextEncoder()
    t_att, t_chr = text_encoder(torch.randint(10, (32, 28)))
    assert t_att.size() == (32, d, 28)
    assert t_chr.size() == (32, d, 28)

def test_tts_model():
    tts_model = TTSModel()
    audio_input = torch.rand(32, F, 227)
    text_input = torch.randint(10, (32, 28))
    audio_output = tts_model([audio_input, text_input])
    assert audio_output.size() == (32, F, 227)

