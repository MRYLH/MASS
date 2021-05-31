import os
from glob import glob
from python_speech_features import fbank, delta
import librosa
import numpy as np
import pandas as pd

import ffmpeg

import pickle
import sys
from multiprocessing import Pool

# import silence_detector
# import constants as c
# from constants import SAMPLE_RATE
from shenwenshibie import silence_detector
import shenwenshibie.constants as c
from shenwenshibie.constants import SAMPLE_RATE
from time import time


def find_files(directory, pattern='**/*.wav'):
    """Recursively finds all files matching the pattern."""
    return glob(os.path.join(directory, pattern), recursive=True)

def data_catalog(dataset_dir=c.DATASET_DIR, pattern='*.npy'):
    libri = pd.DataFrame()
    libri['filename'] = find_files(dataset_dir, pattern=pattern)
    libri['filename'] = libri['filename'].apply(lambda x: x.replace('\\', '/'))  # normalize windows paths
    libri['speaker_id'] = libri['filename'].apply(lambda x: x.split('/')[-1].split('-')[0])
    num_speakers = len(libri['speaker_id'].unique())
    print('Found {} files with {} different speakers.'.format(str(len(libri)).zfill(7), str(num_speakers).zfill(5)))
    # print(libri.head(10))
    return libri

def normalize_frames(m,epsilon=1e-12):
    return [(v - np.mean(v)) / max(np.std(v),epsilon) for v in m]

def extract_features(signal=np.random.uniform(size=48000), target_sample_rate=SAMPLE_RATE):
    filter_banks, energies = fbank(signal, samplerate=target_sample_rate, nfilt=64, winlen=0.025)   #filter_bank (num_frames , 64),energies (num_frames ,)
    #delta_1 = delta(filter_banks, N=1)
    #delta_2 = delta(delta_1, N=1)

    filter_banks = normalize_frames(filter_banks)
    #delta_1 = normalize_frames(delta_1)
    #delta_2 = normalize_frames(delta_2)

    #frames_features = np.hstack([filter_banks, delta_1, delta_2])    # (num_frames , 192)
    frames_features = filter_banks     # (num_frames , 64)
    num_frames = len(frames_features)
    return np.reshape(np.array(frames_features),(num_frames, 64, 1))   #(num_frames,64, 1)


def read_audio(filename, sample_rate=SAMPLE_RATE):
    audio, sr = librosa.load(filename, sr=16000, mono=True)
    audio = VAD(audio.flatten())
    start_sec, end_sec = c.TRUNCATE_SOUND_SECONDS
    start_frame = int(start_sec * SAMPLE_RATE)
    end_frame = int(end_sec * SAMPLE_RATE)

    if len(audio) < (end_frame - start_frame):
        au = [0] * (end_frame - start_frame)
        for i in range(len(audio)):
            au[i] = audio[i]
        audio = np.array(au)
    return audio

def VAD(audio):
    chunk_size = int(SAMPLE_RATE*0.05) # 50ms
    index = 0
    sil_detector = silence_detector.SilenceDetector(15)
    nonsil_audio=[]
    while index + chunk_size < len(audio):
        if not sil_detector.is_silence(audio[index: index+chunk_size]):
            nonsil_audio.extend(audio[index: index + chunk_size])
        index += chunk_size

    return np.array(nonsil_audio)

# def upsample_wav(file, rate):
#     tfm = sox.Transformer()
#     tfm.rate(rate)
#     out_path = file.split('.wav')[0] + "_hr.wav"
#     tfm.build(file, out_path)
#     return out_path

# file = "/home/cjy/Documents/ylh/fakeaudio/TTS--transformer/1wav/1/test_0.wav"
# upsample_wav(file,16000)



wav_dir=c.WAV_DIR

out_dir=c.DATASET_DIR
libri = data_catalog(wav_dir, pattern='**/*.wav')
print(libri)

for i in range(len(libri)):
    filename = libri[i:i + 1]['filename'].values[0]
    print(filename)
    target_filename = out_dir + filename.split("/")[-1].split('.')[0] + '.npy'
    print(target_filename)
    raw_audio = read_audio(filename)
    feature = extract_features(raw_audio, target_sample_rate=SAMPLE_RATE)
    np.save(target_filename, feature)