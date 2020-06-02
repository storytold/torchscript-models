from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .res_stack import ResStack

import numpy as np

from wav_images import render_histogram
from wav_images import rescale_mel

def load_melgan(model_checkpoint_path):
    print('Loading melgan checkpoint: {}'.format(model_checkpoint_path))

    checkpoint = torch.load(model_checkpoint_path, map_location=torch.device('cpu'))
    melgan_state = checkpoint['model_g']
    melgan_state = transform_melgate_state(melgan_state)

    model_melgan = Generator()
    model_melgan.load_state_dict(melgan_state)
    return model_melgan

def transform_melgate_state(melgan_state):
    import re
    BLOCK_REGEX = re.compile('blocks\.(\d+)\.')
    SHORTCUT_REGEX = re.compile('shortcuts\.(\d+)\.')

    new_model_g = [] # Rebuild the ordered dict
    for key, value in melgan_state.items():
        if 'blocks.' in key:
            new_key = BLOCK_REGEX.sub(r'block\1.', key)
            #print('{} -> {}'.format(key, new_key))
            key = new_key
        elif 'shortcuts.' in key:
            new_key = SHORTCUT_REGEX.sub(r'shortcut\1.', key)
            #print('{} -> {}'.format(key, new_key))
            key = new_key

        new_model_g.append((key, value))

    return  OrderedDict(new_model_g)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        mel_channel = 80 # TODO: This is an hparam
        self.mel_channel = mel_channel

        self.generator = nn.Sequential(
            nn.ReflectionPad1d(3),
            nn.utils.weight_norm(nn.Conv1d(mel_channel, 512, kernel_size=7, stride=1)),

            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(512, 256, kernel_size=16, stride=8, padding=4)),

            ResStack(256),

            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(256, 128, kernel_size=16, stride=8, padding=4)),

            ResStack(128),

            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1)),

            ResStack(64),

            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)),

            ResStack(32),

            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            nn.utils.weight_norm(nn.Conv1d(32, 1, kernel_size=7, stride=1)),
            nn.Tanh(),
        )

    def forward(self, mel):
        mel = (mel + 5.0) / 5.0 # roughly normalize spectrogram
        audio = self.generator(mel)
        audio = audio.squeeze()

        MAX_WAV_VALUE = 32768.0
        audio = audio * MAX_WAV_VALUE

        max_ = torch.max(torch.abs(audio))
        #max_ = audio.abs().max()
        #max_ = max(0.01, max_)
        multiplier = 32767.0 / max_
        audio = audio * multiplier
        #max_ = np.max(np.abs(audio))
        #max_ = max(0.01, audio)
        #print("Max: {}".format(max_))
        #print("Multiplier: {}".format(multiplier))

        return audio

    def eval(self):
        super(Generator, self).eval()

    def inference(self, mel):
        audio = self.forward(mel)
        #audio = audio.squeeze() # collapse all dimension except time axis
        return audio

