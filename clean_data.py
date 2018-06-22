from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import torch.utils.data as utils
import librosa
import soundfile as sf
import time
import os
from torch.utils import data
import numpy
import pickle
import numpy as np
import h5py
from os import listdir
from os.path import isfile, join
sample_rate=16000


# In[ ]:


cnt=0
for i in listdir('ccmixter_corpus/'):
    if os.path.isdir('ccmixter_corpus/'+i):
        for j in ['mix.wav','source-01.wav','source-02.wav']:
            name='ccmixter_corpus/'+i+'/'+j
            audio0, samplerate = sf.read(name, dtype='float32')
            audio0=librosa.resample(audio0.T, samplerate, sample_rate)
            audio0 = librosa.to_mono(audio0)
            if not os.path.exists('./ccmixter2/x/'):os.makedirs('./ccmixter2/x/')
            if not os.path.exists('./ccmixter2/y/'):os.makedirs('./ccmixter2/y/')
            if not os.path.exists('./ccmixter2/z/'):os.makedirs('./ccmixter2/z/')
            if(j == 'mix.wav'):
                sf.write('./ccmixter2/x/'+str(cnt)+'.wav', audio0, sample_rate)
            elif(j == 'source-01.wav'):
                sf.write('./ccmixter2/y/'+str(cnt)+'.wav', audio0, sample_rate)
            elif(j == 'source-02.wav'):
                sf.write('./ccmixter2/z/'+str(cnt)+'.wav', audio0, sample_rate)
            else:print('wa')
        cnt+=1


# In[ ]:


for i in range(50):
    name0='ccmixter2/x/'+str(i)+'.wav'
    audio0, samplerate = sf.read(name0, dtype='float32')
    audio0=librosa.resample(audio0.T, samplerate, sample_rate)
    
    name1='ccmixter2/y/'+str(i)+'.wav'
    audio1, samplerate = sf.read(name1, dtype='float32')
    audio1=librosa.resample(audio1.T, samplerate, sample_rate)
    
    name2='ccmixter2/z/'+str(i)+'.wav'
    audio2, samplerate = sf.read(name2, dtype='float32')
    audio2=librosa.resample(audio2.T, samplerate, sample_rate)
    
    if not os.path.exists('./ccmixter3/'):os.makedirs('./ccmixter3/')
    h5f = h5py.File('ccmixter3/'+str(i)+'.h5', 'w')
    h5f.create_dataset('x', data=audio0)
    h5f.create_dataset('y', data=audio1)
    h5f.create_dataset('z', data=audio2)
    h5f.close()
