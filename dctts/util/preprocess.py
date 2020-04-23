import librosa
import numpy as np

def calc_spectrograms(waveform, sampling_rate, gamma=0.6):
    Z_raw = np.abs(librosa.stft(
        waveform, n_fft=1024, window='hann', win_length=1024, hop_length=256
    ))
    Z = (Z_raw / Z_raw.max()) ** gamma
    S_raw = librosa.feature.melspectrogram(S=Z, sr=sampling_rate, n_mels=80)
    S = (S_raw / S_raw.max()) ** gamma
    return {
        'stft': Z_raw,
        'stft_norm': Z,
        'mel': S_raw,
        'mel_norm': S
    }