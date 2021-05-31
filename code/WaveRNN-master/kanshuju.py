import torch
import numpy as np
import librosa
import pyworld
import matplotlib.pyplot as plt

def load_wav(path):
    return librosa.load(path, sr=16000)[0]

def stft(y):
    return librosa.stft(y=y, n_fft=2048, hop_length=80, win_length=1100)

def amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))

def db_to_amp(x):
    return np.power(10.0, x * 0.05)

def linear_to_mel(spectrogram):
    return librosa.feature.melspectrogram(
        S=spectrogram, sr=16000, n_fft=2048, n_mels=36, fmin=40)

def melspectrogram(y):
    D = stft(y)
    # print(D.shape)
    S = linear_to_mel(np.abs(D))
    print(S.shape)
    return S

def world_decompose(wav, fs=16000, frame_period = 5.0):
    # Decompose speech signal into f0, spectral envelope and aperiodicity using WORLD
    wav = wav.astype(np.float64)
    f0, timeaxis = pyworld.harvest(wav, fs, frame_period = frame_period, f0_floor = 71.0, f0_ceil = 800.0)    # f0_floor是基频的下限  f0_ceil是基频的上限
    #   frame_period是连续帧之间的间隔
    sp = pyworld.cheaptrick(wav, f0, timeaxis, fs)
    ap = pyworld.d4c(wav, f0, timeaxis, fs)
    return f0     # 返回基频  时间  频谱包络  非周期性

# path = "/home/NewDisk/yelinhui/yelinhui/Wavenet/WaveRNN-master/data/data_resample/wavs/LJ001-0004.wav"
# y = load_wav(path)
#
# f0 = world_decompose(y)
#
# lenth = len(f0)
# x = np.arange(lenth)
# plt.plot(x, f0, color = "b")
# plt.show()
a = "/LJ002-0147.npy"
a = np.load(a)

b = "/quant/LJ002-0147.npy"
b = np.load(b)
print(len(b))
print(np.sum(a-b))