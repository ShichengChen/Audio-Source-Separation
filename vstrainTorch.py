
# coding: utf-8

# In[1]:


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


# In[2]:


sampleSize=100000
sample_rate=16000
quantization_channels=256
dilations=[2**i for i in range(9)]*5
skipDim=512
residualDim=32
filterSize=3


# In[3]:


use_cuda = torch.cuda.is_available()
torch.manual_seed(1)
device = torch.device("cuda" if use_cuda else "cpu")
#device = 'cpu'
torch.set_default_tensor_type('torch.cuda.FloatTensor')
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


# In[4]:


def mu_law_encode(audio, quantization_channels=quantization_channels):
    '''Quantizes waveform amplitudes.'''
    mu = (quantization_channels - 1)*1.0
    # Perform mu-law companding transformation (ITU-T, 1988).
    # Minimum operation is here to deal with rare large amplitudes caused
    # by resampling.
    safe_audio_abs = np.minimum(np.abs(audio), 1.0)
    magnitude = np.log1p(mu * safe_audio_abs) / np.log1p(mu)
    signal = np.sign(audio) * magnitude
    # Quantize signal to the specified number of levels.
    #if(forX):return signal
    return ((signal + 1) / 2 * mu + 0.5).astype(int)
def mu_law_decode(output, quantization_channels=quantization_channels):
    '''Recovers waveform from quantized values.'''
    mu = quantization_channels - 1
    # Map values back to [-1, 1].
    signal = 2 * ((output*1.0) / mu) - 1
    # Perform inverse of mu-law transformation.
    magnitude = (1 / mu) * ((1 + mu)**np.abs(signal) - 1)
    return np.sign(signal) * magnitude


# In[5]:


def readAudio(name):
    audio0, samplerate = sf.read(name, dtype='float32')
    return librosa.resample(audio0.T, samplerate, sample_rate).reshape(-1)
p=['../vsCorpus/origin_mix.wav','../vsCorpus/origin_mix.wav',
   '../vsCorpus/origin_mix.wav','../vsCorpus/origin_mix.wav','../vsCorpus/pred_mix.wav']
xtrain,ytrain,xval,yval,xtest=readAudio(p[0]),readAudio(p[1]),readAudio(p[2]),readAudio(p[3]),readAudio(p[4])
xtrain,ytrain,xval,yval=xtrain[:-sampleSize],ytrain[:-sampleSize],xval[-sampleSize:],yval[-sampleSize:]
xtrain,xval,xtest=xtrain.reshape(1,1,-1),xval.reshape(1,1,-1),xtest.reshape(1,1,-1)
ytrain,yval=ytrain.reshape(1,-1),yval.reshape(1,-1)


# In[6]:


xtrain=(xtrain-xtrain.mean())/xtrain.std()
xval=(xval-xtrain.mean())/xtrain.std()
xtest=(xtest-xtrain.mean())/xtrain.std()
ytrain,yval=mu_law_encode(ytrain),mu_law_encode(yval)


# In[7]:


xtrain,ytrain,xval,yval,xtest = torch.from_numpy(xtrain).type(torch.float32),                                torch.from_numpy(ytrain).type(torch.LongTensor),                                torch.from_numpy(xval).type(torch.float32),                                torch.from_numpy(yval).type(torch.LongTensor),                                torch.from_numpy(xtest).type(torch.float32)


# In[8]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        sd,qd,rd = skipDim,quantization_channels,residualDim
        self.causal = nn.Conv1d(in_channels=1,out_channels=rd,kernel_size=3,padding=1)
        self.dilated=dict()
        for i, d in enumerate(dilations):
            self.dilated['tanh'+str(i)] = nn.Conv1d(in_channels=rd,out_channels=rd,kernel_size=3,padding=d,dilation=d)
            self.dilated['sigmoid'+str(i)] = nn.Conv1d(in_channels=rd,out_channels=rd,kernel_size=3,padding=d,dilation=d)
            self.dilated['skip'+str(i)] = nn.Conv1d(in_channels=rd,out_channels=sd,kernel_size=1,padding=0)
            self.dilated['dense'+str(i)] = nn.Conv1d(in_channels=rd,out_channels=rd,kernel_size=1,padding=0)
        self.post1 = nn.Conv1d(in_channels=sd,out_channels=sd,kernel_size=1,padding=0)
        self.post2 = nn.Conv1d(in_channels=sd,out_channels=qd,kernel_size=1,padding=0)
        self.tanh,self.sigmoid = nn.Tanh(),nn.Sigmoid()

    def forward(self, x):
        x = self.causal(x)
        skip_connections = torch.zeros([1,skipDim,x.shape[2]], dtype=torch.float32)
        for i, dilation in enumerate(dilations):
            xinput=x.clone()
            #print(x.shape)
            x1 = self.tanh(self.dilated['tanh'+str(i)](x))
            x2 = self.sigmoid(self.dilated['sigmoid'+str(i)](x))
            x = x1*x2
            skip_connections += self.dilated['skip'+str(i)](x).clone()
            x = self.dilated['dense'+str(i)](x)
            x += xinput.clone()
        x = F.relu(x)
        x = self.post2(F.relu(self.post1(skip_connections)))
        return F.log_softmax(x,dim=1)

model = Net().to(device)

#optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
optimizer = optim.Adam(model.parameters(),weight_decay=1e-5)


# In[17]:


def val():
    model.eval()
    startval_time = time.time()
    with torch.no_grad():
        data, target = xval.to(device), yval.to(device)
        output = model(data)
        val_loss = F.nll_loss(output, target).item()
    print('\nval set:  {:.4f}, ({:.3f} sec/step)\n'.format(val_loss,time.time()-startval_time))

def test():
    model.eval()
    startval_time = time.time()
    with torch.no_grad():
        #output = model(xtest.to(device))
        output = model(xtrain[:,:,:sampleSize].to(device).to(device))
        pred = output.max(1, keepdim=True)[1].cpu().numpy().reshape(-1)
        ans = mu_law_decode(pred)
        sf.write('../vsCorpus/ans/result.wav', ans, sample_rate)
        print('stored done\n')
    
def train(epoch):
    model.train()
    idx = np.arange(xtrain.shape[-1] - 2*sampleSize)
    np.random.shuffle(idx)
    for i, ind in enumerate(idx):
        start_time = time.time()
        data, target = xtrain[:,:,ind:ind+sampleSize].to(device), ytrain[:,ind:ind+sampleSize].to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} , ({:.3f} sec/step)'.format(
                epoch, i, len(idx),100. * i / len(idx), loss.item(),time.time() - start_time))
        if i % 100 == 0:
            val()
            test()
            torch.save(model, 'torchmodel')


# In[18]:


for epoch in range(10):
    train(epoch)
    test()


# In[ ]:


#model = torch.load('torchmodel')

