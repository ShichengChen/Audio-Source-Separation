from torch.utils import data
from wavenet import Wavenet
from transformData import x_mu_law_encode, y_mu_law_encode, mu_law_decode, onehot, cateToSignal
import librosa
import soundfile as sf
import numpy as np
import torch

dilations = [2 ** i for i in range(9)] * 7
pad = np.sum(dilations)
sampleSize = 16000
sample_rate = 16000  # the length of audio for one second


class Dataset(data.Dataset):
    def __init__(self, listx, listy, rootx, rooty):
        self.rootx = rootx
        self.rooty = rooty
        self.listx = listx
        self.listy = listy

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.listx)

    def __getitem__(self, index):
        'Generates one sample of data'
        namex = self.listx[index]
        namey = self.listy[index]
        x, samplerate = sf.read(self.rootx + str(namex) + '.wav', dtype='float32')
        x = librosa.resample(x.T, samplerate, sample_rate)
        x = librosa.to_mono(x)  # read audio and transfrom from stereo to to mono

        y, samplerate = sf.read(self.rooty + str(namey) + '.wav', dtype='float32')
        y = librosa.resample(y.T, samplerate, sample_rate)
        y = librosa.to_mono(y)

        x = x_mu_law_encode(x)  # use mu_law to encode the audio
        y = y_mu_law_encode(y)

        xmean = x.mean()
        xstd = x.std()
        x = (x - xmean) / xstd
        # x+=np.random.normal(size=x.shape[-1])*(1e-3)
        x = np.pad(x, (pad, pad), 'constant')
        y = np.pad(y, (pad, pad), 'constant')
        startx = np.random.randint(pad, x.shape[-1] - sampleSize - pad)
        x = x[startx - pad:startx + sampleSize + pad]
        y = y[startx:startx + sampleSize]
        l = np.random.uniform(0.25, 0.5)
        sp = np.random.uniform(0, 1 - l)
        step = np.random.uniform(-0.5, 0.5)
        ux = int(sp * sample_rate)
        lx = int(l * sample_rate)
        x[ux:ux + lx] = librosa.effects.pitch_shift(x[ux:ux + lx], sample_rate, n_steps=step)
        x = torch.from_numpy(x.reshape(1, -1)).type(torch.float32)
        y = torch.from_numpy(y.reshape(-1)).type(torch.LongTensor)
        return x, y


class Testset(data.Dataset):
    def __init__(self, listx, rootx):
        self.rootx = rootx
        self.listx = listx

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.listx)

    def __getitem__(self, index):
        'Generates one sample of data'
        namex = self.listx[index]
        x, samplerate = sf.read(self.rootx + str(namex) + '.wav', dtype='float32')
        x = librosa.resample(x.T, samplerate, sample_rate)
        x = librosa.to_mono(x)  # read audio and transfrom from stereo to to mono
        x = x_mu_law_encode(x)

        xmean = x.mean()
        xstd = x.std()
        x = (x - xmean) / xstd
        # x+=np.random.normal(size=x.shape[-1])*(1e-3)
        x = np.pad(x, (pad, pad), 'constant')

        x = torch.from_numpy(x.reshape(1, -1)).type(torch.float32)
        return x