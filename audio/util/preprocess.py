import librosa
import numpy as np

def calc_spectrograms(waveform, sampling_rate, gamma=2):
    Z_raw = np.abs(librosa.stft(waveform))
    Z = (Z_raw / Z_raw.max()) ** gamma
    S_raw = librosa.feature.melspectrogram(S=Z, sr=sampling_rate)
    S = (S_raw / S_raw.max()) ** gamma
    return {
        'stft': Z_raw,
        'stft_norm': Z,
        'mel': S_raw,
        'mel_norm': S
    }