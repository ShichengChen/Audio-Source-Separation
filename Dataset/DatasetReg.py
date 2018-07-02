from torch.utils import data
from wavenet import Wavenet
from transformData import x_mu_law_encode, y_mu_law_encode, mu_law_decode, onehot, cateToSignal
import librosa
import soundfile as sf
import numpy as np
import torch
import h5py
import datetime

sampleSize = 16000
sample_rate = 16000  # the length of audio for one second


class Dataset(data.Dataset):
    def __init__(self, listx, listy, rootx, rooty,pad, transform=None):
        self.rootx = rootx
        self.rooty = rooty
        self.listx = listx
        self.listy = listy
        self.pad=pad
        self.transform = transform

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.listx)

    def __getitem__(self, index):
        np.random.seed()
        namex = self.listx[index]
        namey = self.listy[index]

        h5f = h5py.File(self.rootx + str(namex) + '.h5', 'r')
        x, y, z = h5f['x'][:], h5f['y'][:],h5f['z'][:]

        factor0 = np.random.uniform(low=0.83, high=1.0)
        factor1 = np.random.uniform(low=0.83, high=1.0)
        #print(factor)
        z = z*factor0
        y = y*factor1
        x = (y + z)
        x = x_mu_law_encode(x)  # use mu_law to encode the audio
        y = x_mu_law_encode(y)

        #xmean = -0.0039727693202439695
        #xstd = 0.54840165197849278
        #x = (x - xmean) / xstd


        x = np.pad(x, (self.pad, self.pad), 'constant')
        y = np.pad(y, (self.pad, self.pad), 'constant')
        print(x.shape,y.shape)

        sample = {'x': x, 'y': y}

        if self.transform:
            sample = self.transform(sample)

        return namex,sample['x'], sample['y']


class RandomCrop(object):
    def __init__(self, pad,output_size=sample_rate):
        self.output_size = output_size
        self.pad=pad

    def __call__(self, sample):
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)
        x, y = sample['x'], sample['y']
        shrink = 0
        startx = np.random.randint(self.pad + shrink * sampleSize, x.shape[-1] - sampleSize - self.pad - shrink * sampleSize)
        #print(startx)
        #x = x[startx - pad:startx + sampleSize + pad]
        #y = y[startx:startx + sampleSize]
        l = np.random.uniform(0.25, 0.5)
        sp = np.random.uniform(0, 1 - l)
        step = np.random.uniform(-0.5, 0.5)
        ux = int(sp * sample_rate)
        lx = int(l * sample_rate)
        # x[ux:ux + lx] = librosa.effects.pitch_shift(x[ux:ux + lx], sample_rate, n_steps=step)

        return {'x': x, 'y': y}


class ToTensor(object):
    def __call__(self, sample):
        x, y = sample['x'], sample['y']
        return {'x': torch.from_numpy(x.reshape(1, -1)).type(torch.float32),
                'y': torch.from_numpy(y.reshape(1, -1)).type(torch.float32)}


class Testset(data.Dataset):
    def __init__(self, listx, rootx,pad):
        self.rootx = rootx
        self.listx = listx
        self.pad = pad

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.listx)

    def __getitem__(self, index):
        'Generates one sample of data'
        namex = self.listx[index]

        h5f = h5py.File('ccmixter3/' + str(namex) + '.h5', 'r')
        x, y = h5f['x'][:], h5f['y'][:]

        x = x_mu_law_encode(x)  # use mu_law to encode the audio
        y = x_mu_law_encode(y)

        #xmean = -0.0039727693202439695
        #xstd = 0.54840165197849278
        #x = (x - xmean) / xstd
        x = np.pad(x, (self.pad, self.pad), 'constant')
        y = np.pad(y, (self.pad, self.pad), 'constant')

        x = torch.from_numpy(x.reshape(1, -1)).type(torch.float32)
        y = torch.from_numpy(y.reshape(1, -1)).type(torch.float32)
        return namex,x,y