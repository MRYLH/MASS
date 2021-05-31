import librosa
import numpy as np
import os, sys

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import argparse
# import pyworld
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from utils import *
from tqdm import tqdm
from collections import defaultdict
from collections import namedtuple
from sklearn.model_selection import train_test_split
import glob
from os.path import join, basename
import subprocess


def resample(spk, origin_wavpath, target_wavpath):
    wavfiles = [i for i in os.listdir(join(origin_wavpath, spk)) if
                i.endswith(".wav")]

    for wav in wavfiles:
        folder_to = join(target_wavpath, spk)
        os.makedirs(folder_to, exist_ok=True)
        wav_to = join(folder_to, wav)
        wav_from = join(origin_wavpath, spk, wav)
        subprocess.call(['sox', wav_from, "-r", "16000", wav_to])
    return 0


def resample_to_16k(origin_wavpath, target_wavpath, num_workers=1):
    os.makedirs(target_wavpath, exist_ok=True)
    emotion_folders = os.listdir(
        origin_wavpath)
    print("f> Using {num_workers} workers!")
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    for emotion in emotion_folders:
        futures.append(executor.submit(partial(resample, emotion, origin_wavpath, target_wavpath)))
    result_list = [future.result() for future in tqdm(futures)]
    print(result_list)


def split_data(paths):
    indices = np.arange(len(paths))
    test_size = 0.005
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=1234)
    train_paths = list(np.array(paths)[train_indices])
    test_paths = list(np.array(paths)[test_indices])
    return train_paths, test_paths


def get_emotion_world_feats(wave_path, output_path, sample_rate=16000):
    wave_directs = os.listdir(wave_path)
    for wave_direct in wave_directs:
        wave_direct_path = wave_path+"/"+wave_direct
        paths = glob.glob(join(wave_direct_path, '*.wav'))

        coded_sps = []
        for wav_file in tqdm(paths):
            f0, _, sp, _, coded_sp = world_encode_wav(wav_file, fs=sample_rate)
            coded_sps.append(coded_sp)

        coded_sps_mean, coded_sps_std = coded_sp_statistics(coded_sps)

        for wave_file in tqdm(paths):
            wav_name = basename(wave_file)
            f0, timeaxis, sp, ap, coded_sp = world_encode_wav(wave_file, fs=sample_rate)
            f0 = f0.reshape(-1, 1)
            normed_coded_sp = normalize_coded_sp(coded_sp, coded_sps_mean, coded_sps_std)

            feature = np.concatenate((normed_coded_sp, f0), axis=1)
            np.save(join(output_path, wav_name.replace('.wav', '.npy')), feature, allow_pickle=False)

    return 0


if __name__ == '__main__':
    path = ""
    output_path = ""
    get_emotion_world_feats(wave_path=path, output_path = output_path)