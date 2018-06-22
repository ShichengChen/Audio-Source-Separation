from torch.utils import data
from wavenet import Wavenet
from transformData import x_mu_law_encode, y_mu_law_encode, mu_law_decode, onehot, cateToSignal
import librosa
import soundfile as sf
import numpy as np
import torch
import h5py
import datetime

dilations = [2 ** i for i in range(9)] * 7
pad = np.sum(dilations)
sampleSize = 16000
sample_rate = 16000  # the length of audio for one second


class Dataset(data.Dataset):
    def __init__(self, listx, listy, rootx, rooty, transform=None):
        self.rootx = rootx
        self.rooty = rooty
        self.listx = listx
        self.listy = listy
        self.transform = transform

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.listx)

    def __getitem__(self, index):
        'Generates one sample of data'
        namex = self.listx[index]
        namey = self.listy[index]

        h5f = h5py.File('ccmixter3/' + str(namex) + '.h5', 'r')
        x, y,z = h5f['x'][:], h5f['y'][:], h5f['z'][:]

        x = x_mu_law_encode(x)  # use mu_law to encode the audio
        y = y_mu_law_encode(y)
        z = y_mu_law_encode(z)

        xmean = x.mean()
        xstd = x.std()
        x = (x - xmean) / xstd
        # x+=np.random.normal(size=x.shape[-1])*(1e-3)
        x = np.pad(x, (pad, pad), 'constant')
        y = np.pad(y, (pad, pad), 'constant')
        z = np.pad(z, (pad, pad), 'constant')

        sample = {'x': x, 'y': y,'z': z}

        if self.transform:
            sample = self.transform(sample)

        return sample['x'], sample['y'],sample['z']


class RandomCrop(object):
    def __init__(self, output_size=sample_rate):
        self.output_size = output_size

    def __call__(self, sample):
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)
        x, y, z = sample['x'], sample['y'],sample['z']
        shrink = 0
        startx = np.random.randint(pad + shrink * sampleSize, x.shape[-1] - sampleSize - pad - shrink * sampleSize)
        #print(startx)
        #x = x[startx - pad:startx + sampleSize + pad]
        #y = y[startx:startx + sampleSize]
        l = np.random.uniform(0.25, 0.5)
        sp = np.random.uniform(0, 1 - l)
        step = np.random.uniform(-0.5, 0.5)
        ux = int(sp * sample_rate)
        lx = int(l * sample_rate)
        # x[ux:ux + lx] = librosa.effects.pitch_shift(x[ux:ux + lx], sample_rate, n_steps=step)

        return {'x': x, 'y': y,'z': z}


class ToTensor(object):
    def __call__(self, sample):
        x, y,z = sample['x'], sample['y'],sample['z']
        return {'x': torch.from_numpy(x.reshape(1, -1)).type(torch.float32),
                'y': torch.from_numpy(y.reshape(-1)).type(torch.LongTensor),
               'z': torch.from_numpy(z.reshape(-1)).type(torch.LongTensor)}


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

        h5f = h5py.File('ccmixter3/' + str(namex) + '.h5', 'r')
        x = h5f['x'][:]

        xmean = x.mean()
        xstd = x.std()
        x = (x - xmean) / xstd
        # x+=np.random.normal(size=x.shape[-1])*(1e-3)
        x = np.pad(x, (pad, pad), 'constant')

        x = torch.from_numpy(x.reshape(1, -1)).type(torch.float32)
        return x