import librosa
import hparams as hp
import os
import numpy as np
import glob
import os
from os.path import join,basename
from utils.display import *
from utils.dsp import *
from utils import hparams as hp
from multiprocessing import Pool, cpu_count
from utils.paths import Paths
import pickle
import argparse
from utils.text.recipes import ljspeech
from utils.files import get_files
from pathlib import Path


def load_wav(path):
    return librosa.load(path, sr=16000)[0]

def valid_n_workers(num):
    n = int(num)
    if n < 1:
        raise argparse.ArgumentTypeError('%r must be an integer greater than 0' % num)
    return n
parser = argparse.ArgumentParser(description='Preprocessing for WaveRNN and Tacotron')
parser.add_argument('--path', '-p', help='directly point to dataset path (overrides hparams.wav_path')
parser.add_argument('--extension', '-e', metavar='EXT', default='.wav', help='file extension to search for in dataset folder')
parser.add_argument('--num_workers', '-w', metavar='N', type=valid_n_workers, default=cpu_count()-1, help='The number of worker threads to use for preprocessing')
parser.add_argument('--hp_file', metavar='FILE', default='hparams.py', help='The file to use for the hyperparameters')
args = parser.parse_args()

hp.configure(args.hp_file)  # Load hparams from file
if args.path is None:
    args.path = hp.wav_path

extension = args.extension
path = args.path

wave_path_source = "/data_vcc_dataset"
wave_directs = os.listdir(wave_path_source)

for wave_direct in wave_directs:
    wave_direct_path = wave_path_source+"/"+wave_direct
    paths = glob.glob(join(wave_direct_path, '*.wav'))

    for wave_path in paths:
        wave_name = basename(wave_path)[:-4]
        y = load_wav(wave_path)
        peak = np.abs(y).max()
        if hp.peak_norm or peak > 1.0:
            y /= peak
        if hp.voc_mode == 'RAW':
            quant = encode_mu_law(y, mu=2**hp.bits) if hp.mu_law else float_2_label(y, bits=hp.bits)
        elif hp.voc_mode == 'MOL':
            quant = float_2_label(y, bits=16)
        np.save("/quant_speaker_to_fine_tuning/{}.npy".format(wave_name), quant.astype(np.int64))