# coding=utf-8
'''
用于　mellotron　模型的推理。
'''
import os,sys,json
import time
import argparse
import math
from numpy import finfo
import numpy as np
import librosa,yaml
import librosa.display
import soundfile as sf
import torch
from distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from models.model import load_model
from data_utils import TextMelLoader, TextMelCollate


from waveglow.denoiser import Denoiser
from waveglow import glow
from waveglow.glow import WaveGlow, WaveGlowLoss

from layers import TacotronSTFT
from text import cmudict, text_to_sequence
from mellotron_utils import get_data_from_musicxml
from feats.TTS_Feat import get_mel_and_f0_std,mel_to_wav
import matplotlib
import matplotlib.pyplot as plt
from text.parse_text_to_pyin_v3 import plot_alignment,get_mix_text_pinyin_with_youdao


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def panner(signal, angle):
    angle = np.radians(angle)
    left = np.sqrt(2)/2.0 * (np.cos(angle) - np.sin(angle)) * signal
    right = np.sqrt(2)/2.0 * (np.cos(angle) + np.sin(angle)) * signal
    return np.dstack((left, right))[0]
def plot_mel_f0_alignment(mel_source, mel_outputs_postnet, f0s, alignments, tag='',figsize=(16, 16)):
    fig, axes = plt.subplots(4, 1, figsize=figsize)
    axes = axes.flatten()
    axes[0].imshow(mel_source, aspect='auto', origin='bottom', interpolation='none')
    axes[1].imshow(mel_outputs_postnet, aspect='auto', origin='bottom', interpolation='none')
    axes[2].scatter(range(len(f0s)), f0s, alpha=0.5, color='red', marker='.', s=1)
    axes[2].set_xlim(0, len(f0s))
    axes[3].imshow(alignments, aspect='auto', origin='bottom', interpolation='none')
    axes[0].set_title("Source Mel")
    axes[1].set_title("Predicted Mel")
    axes[2].set_title("Source pitch contour")
    axes[3].set_title("Source rhythm")
    plt.tight_layout()

    timeStr = 'ifer-mul-'+str(time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime(time.time())))
    figName=timeStr+'-'+tag+'.jpg'
    plt.savefig(figName)


if __name__=='__main__':
    timeStr = 'ifer-'+str(time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime(time.time())))
    # from hparams_mix import create_hparams
    from hparams_LJ import create_hparams

    hparams = create_hparams()
    # conig
    checkpoint_path = "/tts_data/wangtao/project/TTS/mellotron/outdir_LJ/model_tag/checkpoint_27200"
    # load syn : model
    assert os.path.isfile(checkpoint_path),'\t--> checkpoint_path='+str(checkpoint_path)+' not exist! '
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    mellotron = load_model(hparams) #.cuda().eval()
    mellotron.load_state_dict(checkpoint_dict['state_dict'])

    # load codeoder
    if 1:
        waveglow_path = '/tts_data/wangtao/project/TTS/mellotron/pt_model/waveglow_256channels_universal_v4.pt'
        waveglow_path = '/tts_data/wangtao/project/TTS/mellotron/pt_model/waveglow_256channels_ljs_v3.pt'
        waveglow_path = '/tts_data/wangtao/project/TTS/mellotron/pt_model/nvidia_waveglow256pyt_fp16' # dict_keys(['epoch', 'config', 'state_dict'])
        # waveglow = torch.load(waveglow_path)['model'].cuda().eval()
        wgDict = torch.load(waveglow_path)
        # print('wgDict.keys()=',wgDict.keys()) #  dict_keys(['epoch', 'config', 'state_dict'])
        # waveglow = torch.load(waveglow_path)['model']

        waveglow_config={
        "n_mel_channels": 80,
        "n_flows": 12,
        "n_group": 8,
        "n_early_every": 4,
        "n_early_size": 2,
        "WN_config": {
            "n_layers": 8,
            "n_channels": 256,
            "kernel_size": 3
        }
        }
        waveglow = WaveGlow(**waveglow_config).cuda()

        denoiser = Denoiser(waveglow).cuda().eval()
        # sys.exit()
    #################3 infer
    # prepare input
    audio_paths = '/tts_data/wangtao/project/TTS/mellotron/filelists-mix/mix-eng-12167-v5Fix_SphHuar-gst_per_0.1_dev-feat.txt'
    hparams.need_pinyin_trans = True
    dataloader = TextMelLoader(audio_paths, hparams)
    datacollate = TextMelCollate(1)
    # text
    text = '你好，中国'
    text = 'what is the weather today '

    # file_idx = 0
    # audio_path, text, sid = dataloader.audiopaths_and_text[file_idx]
    audio_path = '/tts_data/wangtao/project/TTS/mellotron/data/example1.wav'
    # get audio path, encoded text, pitch contour and mel for gst
    # text_id_tensor = dataloader.get_text_ChiEng_mix(text)
    # /tts_data/wangtao/project/TTS/mellotron/text/__init__.py
    text_id_tensor = dataloader.get_text(text)


    cmuDictFile= 'cmudict-0.7b-use.txt'
    addjin0Mark = True
    keepChiInerSpace = True
    pinyin = get_mix_text_pinyin_with_youdao(text,cmuDictFile,addjin0Mark=addjin0Mark,keepChiInerSpace=keepChiInerSpace)
    print('\t--> text           =',text)
    print('\t--> pinyin         =',pinyin)
    print('\t--> text_id_tensor =',text_id_tensor)



    mel, f0, audio = get_mel_and_f0_std(audio_path,hparams.max_wav_value,
                    hparams.sampling_rate,hparams.filter_length,hparams.hop_length,
                    hparams.win_length,hparams.n_mel_channels,
                    hparams.f0_min,hparams.f0_max,hparams.harm_thresh,hparams.mel_fmin,hparams.mel_fmax)

    # load source data to obtain rhythm using tacotron 2 as a forced aligner
    speaker_id = [0]
    speaker_id = torch.LongTensor(speaker_id)  #   LongTensor     IntTensor
    f0 = torch.from_numpy(f0)
    mel = torch.from_numpy(mel)
    x, y = mellotron.parse_batch(datacollate([(text_id_tensor, mel, speaker_id, f0)]))

    with torch.no_grad():
        # get rhythm (alignment map) using tacotron 2
        mel_outputs, mel_outputs_postnet, gate_outputs, rhythm = mellotron.forward(x)
        rhythm = rhythm.permute(1, 0, 2)

    text_id_np = text_id_tensor.numpy()
    # text_id_tensor = torch.IntTensor(text_id_np[None,:])
    text_id_tensor = torch.LongTensor(text_id_np[None,:].astype(int))

    # print('\t--> f0.shape=',f0.shape)
    # print('\t--> type(f0)=',type(f0))
    # print('\t--> f0=',f0)
    mel = mel[None,:,:]
    f0 = f0[None,:,:]

    # print('\t--> mel.shape=',mel.shape)
    # print('\t--> type(mel)=',type(mel))
    # print('\t--> mel=',mel)
    with torch.no_grad():
        # text, style_input, speaker_ids, f0s, attention_map = inputs
        mel_outputs, mel_outputs_postnet, gate_outputs, alignments = mellotron.inference_noattention(
            (text_id_tensor.to(torch.device('cuda')), mel.to(torch.device('cuda')), speaker_id.to(torch.device('cuda')), f0.to(torch.device('cuda')), rhythm.to(torch.device('cuda'))))
    
    
    print('mel_outputs.shape=',mel_outputs.cpu().detach().numpy().shape)
    print('gate_outputs.shape=',gate_outputs.cpu().detach().numpy().shape)
    print('alignments.shape=',alignments.cpu().detach().numpy().shape)
    print('f0.shape=',f0.numpy().shape)
    print('mel.shape=',mel.numpy().shape)
    # 
    plot_mel_f0_alignment(x[2].data.cpu().numpy()[0],
                        mel_outputs_postnet.data.cpu().numpy()[0],
                        f0.data.cpu().numpy()[0, 0],
                        rhythm.data.cpu().numpy()[:, 0].T)
    
    plot_mel_f0_alignment(x[2].data.cpu().numpy()[0],
                        mel_outputs_postnet.data.cpu().numpy()[0],
                        f0.data.cpu().numpy()[0, 0],
                        alignments.data.cpu().numpy()[0],'ali')

    pic_filename = 'iteration-ali-'+str(iteration)+timeStr+'.jpg'
    plot_alignment(alignments.data.cpu().numpy()[0],pic_filename)
    
    # GL
    # timeStr = 'ifer-'+str(time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime(time.time())))
    if 1:
        GLconfig= '/tts_data/wangtao/project/TTS/Real-Time-Voice-Cloning/config_featV3.yaml'
        GLconfig= '/tts_data/wangtao/project/TTS/mellotron/condifg_gl.yml'
        with open(GLconfig) as f:
            GLconfig = yaml.load(f, Loader=yaml.Loader)
        spec0 = mel_outputs_postnet.cpu().detach().numpy()[0,:,:]
        wavGL = mel_to_wav(spec0.T,GLconfig)
        # GL-write audio
        filename = 'iteration-gl-'+str(iteration)+timeStr+'.wav'
        sf.write(filename, wavGL,16000)
    # mel save
    from feats.TTS_Feat import np_save
    mel_filename = 'iteration-mel-'+str(iteration)+timeStr+'.npy'
    np_save(mel_filename,spec0)

    # waveglow　
    # audio_stereo = np.zeros((hparams.sampling_rate*n_seconds, 2), dtype=np.float32)
    if 1:
        data = get_data_from_musicxml('data/haendel_hallelujah.musicxml', 132, convert_stress=True)
        # print('data=',data)
        # print('data.shape=',data.shape)
        part = 'Soprano'
        audio = denoiser(waveglow.infer(mel_outputs_postnet, sigma=0.8), 0.01)[0, 0]
        audio = audio.cpu().numpy()
        # panning = {'Soprano': [-60, -30], 'Alto': [-40, -10], 'Tenor': [30, 60], 'Bass': [10, 40]}
        # pan = np.random.randint(panning[part][0], panning[part][1])
        # audio = panner(audio, pan)
        # audio_stereo[:audio.shape[0]] += audio            
        # audio_stereo = audio_stereo / np.max(np.abs(audio_stereo))
        filename = 'iteration-waveglow-'+str(iteration)+timeStr+'.wav'
        sf.write(filename, audio, 16000)

    # plot
    t0 = np.linspace(0,len(audio)-1,len(audio))/hparams.sampling_rate
    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(t0,audio,'r--*')

    plt.subplot(2,2,2)
    librosa.display.specshow(mel.numpy()[0,:,:])
    plt.title(' mel ')

    plt.subplot(2,2,3)
    # librosa.display.specshow(f0_1)
    plt.plot(f0.numpy()[0,0,:],'r--*')
    plt.title(' f0 ')

    plt.subplot(2,2,4)
    librosa.display.specshow(mel_outputs.cpu().detach().numpy()[0,:,:])
    plt.title(' mel_outputs ')

    plt.draw()
    # timeStr = 'ifer-'+str(time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime(time.time())))
    figName=timeStr+'.jpg'
    plt.savefig(figName)
    plt.show()


