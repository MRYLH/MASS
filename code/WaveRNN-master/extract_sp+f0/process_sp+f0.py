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


def get_emotion_world_feats(emotion_fold_path, mc_dir_train, mc_dir_test, sample_rate=16000):
    paths = glob.glob(join(emotion_fold_path, '*.wav'))
    train_paths, test_paths = split_data(paths)



    a = np.load("/home/yelinhui/yelinhui/quanbu/fakeaudio/wavenet/WaveRNN-master/data/sp+f0/wavs_stats.npz")
    coded_sps_mean = a["coded_sps_mean"]
    coded_sps_std = a["coded_sps_std"]

    log_f0s_mean = a["log_f0s_mean"]
    log_f0s_std = a["log_f0s_std"]

    for wav_file in tqdm(train_paths):
        wav_nam = basename(wav_file)
        f0, timeaxis, sp, ap, coded_sp = world_encode_wav(wav_file, fs=sample_rate)
        f0 = f0.reshape(-1, 1)

        normed_coded_sp = normalize_coded_sp(coded_sp, coded_sps_mean,
                                             coded_sps_std)


        feature = np.concatenate((normed_coded_sp, f0), axis=1)
        np.save(join(mc_dir_train, wav_nam.replace('.wav', '.npy')), feature, allow_pickle=False)

    for wav_file in tqdm(test_paths):
        wav_nam = basename(wav_file)
        f0, timeaxis, sp, ap, coded_sp = world_encode_wav(wav_file, fs=sample_rate)
        f0 = f0.reshape(-1, 1)
        normed_coded_sp = normalize_coded_sp(coded_sp, coded_sps_mean,
                                             coded_sps_std)


        feature = np.concatenate((normed_coded_sp, f0), axis=1)
        np.save(join(mc_dir_test, wav_nam.replace('.wav', '.npy')), feature, allow_pickle=False)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    sample_rate_default = 16000
    origin_wavpath_default = "/"
    target_wavpath_default = "/"
    mc_dir_train_default = '/'
    mc_dir_test_default = '/'

    parser.add_argument("--sample_rate", type=int, default=16000, help="Sample rate.")  # 采样率
    parser.add_argument("--origin_wavpath", type=str, default=origin_wavpath_default,
                        help="The original wav path to resample.")  # 原语音数据集路径
    parser.add_argument("--target_wavpath", type=str, default=target_wavpath_default,
                        help="The original wav path to resample.")  # 采样为16k后的语音数据集路径
    parser.add_argument("--mc_dir_train", type=str, default=mc_dir_train_default,
                        help="The directory to store the training features.")  # 训练的mcep训练集路径
    parser.add_argument("--mc_dir_test", type=str, default=mc_dir_test_default,
                        help="The directory to store the testing features.")  # 测试的mcep路径
    parser.add_argument("--num_workers", type=int, default=None, help="The number of cpus to use.")

    argv = parser.parse_args()

    sample_rate = argv.sample_rate
    origin_wavpath = argv.origin_wavpath
    target_wavpath = argv.target_wavpath
    mc_dir_train = argv.mc_dir_train
    mc_dir_test = argv.mc_dir_test
    num_workers = argv.num_workers if argv.num_workers is not None else cpu_count()


    emotion_used = ['wavs']


    os.makedirs(mc_dir_train, exist_ok=True)
    os.makedirs(mc_dir_test, exist_ok=True)

    num_workers = 1  # cpu_count()
    print("number of workers: ", num_workers)
    executor = ProcessPoolExecutor(max_workers=num_workers)

    work_dir = target_wavpath

    futures = []
    for emotion in emotion_used:
        emotion_used_path = os.path.join(work_dir, emotion)
        print(emotion_used_path)
        futures.append(executor.submit(
            partial(get_emotion_world_feats, emotion_used_path, mc_dir_train, mc_dir_test, sample_rate)))
    result_list = [future.result() for future in tqdm(futures)]
    print(result_list)
    sys.exit(0)

