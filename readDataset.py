from torch.utils import data
from wavenet import Wavenet
from transformData import x_mu_law_encode,y_mu_law_encode,mu_law_decode,onehot,cateToSignal
import librosa
import soundfile as sf
import numpy as np
import torch
sample_rate=16000#the length of audio for one second

class Dataset(data.Dataset):
    def __init__(self, listx,listy,rootx,rooty):
        self.rootx=rootx
        self.rooty=rooty
        self.listx = listx
        self.listy = listy

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.listx)

    def __getitem__(self, index):
        'Generates one sample of data'
        namex = self.listx[index]
        namey = self.listy[index]
        x, samplerate = sf.read(self.rootx+str(namex)+'.wav', dtype='float32')
        x=librosa.resample(x.T, samplerate, sample_rate)
        x = librosa.to_mono(x) # read audio and transfrom from stereo to to mono
        
        y, samplerate = sf.read(self.rooty+str(namey)+'.wav', dtype='float32')
        y=librosa.resample(y.T, samplerate, sample_rate)
        y = librosa.to_mono(y)
        
        
        x=x_mu_law_encode(x).reshape(1,-1) # use mu_law to encode the audio
        y=y_mu_law_encode(y).reshape(-1)
        
        xmean=x.mean()
        xstd=x.std()
        x=(x-xmean)/xstd
        #x+=np.random.normal(size=x.shape[-1])*(1e-3)
        
        x=torch.from_numpy(x).type(torch.float32)
        y=torch.from_numpy(y).type(torch.LongTensor)
        return x, y