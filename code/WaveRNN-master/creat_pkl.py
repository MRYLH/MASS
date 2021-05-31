import numpy as np
import pickle
from os.path import join, basename
import os


path = ""
wavfiles = [i for i in os.listdir(path) if
                i.endswith(".npy")]

dataset = []
for file in wavfiles:
    wav_path = join(path, file)
    wav_length = len(np.load(wav_path))
    wave_name = basename(file)[:-4]
    dataset += [(wave_name, wav_length)]


path_pkl = ""
with open(path_pkl+"/"+'dataset_sp_f0_speaker_finr_tuning.pkl', 'wb') as f:
    pickle.dump(dataset, f)
# for i, (item_id, length) in enumerate(pool.imap_unordered(process_wav, wav_files), 1):
#     dataset += [(item_id, length)]
#     bar = progbar(i, len(wav_files))
#     message = f'{bar} {i}/{len(wav_files)} '
#     stream(message)

