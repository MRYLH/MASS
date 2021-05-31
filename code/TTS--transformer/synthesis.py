import sys
# sys.setdefaultencoding('utf-8')
import torch as t
from utils import spectrogram2wav
from scipy.io.wavfile import write
import hyperparams as hp
from text import text_to_sequence
import numpy as np
from network import ModelPostNet, Model
from collections import OrderedDict
from tqdm import tqdm
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def load_checkpointpostnet(step, model_name="transformer"):
    state_dict = t.load('/home/yelinhui/yelinhui/quanbu/fakeaudio/TTS--transformer/checkpoint/checkpoint_postnet_90000.pth.tar')
    new_state_dict = OrderedDict()
    for k, value in state_dict['model'].items():
        key = k[7:]
        new_state_dict[key] = value

    return new_state_dict

def load_checkpointtransformer(step, model_name="transformer"):
    state_dict = t.load('/home/yelinhui/yelinhui/quanbu/fakeaudio/TTS--transformer/checkpoint/checkpoint_transformer_160000.pth.tar')
    new_state_dict = OrderedDict()
    for k, value in state_dict['model'].items():
        key = k[7:]
        new_state_dict[key] = value

    return new_state_dict

def synthesis(text, args):
    m = Model()
    m_post = ModelPostNet()

    m.load_state_dict(load_checkpointtransformer(args.restore_step1, "transformer"))      # 载入前端
    m_post.load_state_dict(load_checkpointpostnet(args.restore_step2, "postnet"))         # 载入后处理网络

    text = np.asarray(text_to_sequence(text, [hp.cleaners]))
    text = t.LongTensor(text).unsqueeze(0)
    text = text.cuda()
    mel_input = t.zeros([1, 1, 80]).cuda()

    pos_text = t.arange(1, text.size(1)+1).unsqueeze(0)   # 生成[1,2,3........text.size(1)]的张量
    # torch.size=[1,text.size(1)+1] torch.arange(start,end)
    pos_text = pos_text.cuda()

    m = m.cuda()
    m_post = m_post.cuda()
    m.train(False)
    m_post.train(False)
    
    pbar = tqdm(range(args.max_len))
    with t.no_grad():
        for i in pbar:
            pos_mel = t.arange(1, mel_input.size(1)+1).unsqueeze(0).cuda()
            mel_pred, postnet_pred, attn, stop_token, _, attn_dec = m.forward(text, mel_input, pos_text, pos_mel)
            mel_input = t.cat([mel_input, postnet_pred[:, -1:, :]], dim=1)

        mag_pred = m_post.forward(postnet_pred)
        print(postnet_pred.shape)
        print(mag_pred.shape)
    wav = spectrogram2wav(mag_pred.squeeze(0).cpu().numpy())
    write(hp.sample_path + "/neutre_5.wav", hp.sr, wav)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step1', type=int, help='Global step to restore checkpoint', default=160000)
    parser.add_argument('--restore_step2', type=int, help='Global step to restore checkpoint', default=90000)
    parser.add_argument('--max_len', type=int, help='Global step to restore checkpoint', default=120)

    args = parser.parse_args()
    synthesis("That smells terrible.", args)
