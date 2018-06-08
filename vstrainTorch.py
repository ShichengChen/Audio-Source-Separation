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
import os

# In[2]:


sampleSize = 50000
sample_rate = 16000
quantization_channels = 256
dilations = [2 ** i for i in range(10)] * 5
skipDim = 256
residualDim = 96
filterSize = 3
pad = np.sum(dilations)
shapeoftest = 190500
lossrecord = []
pad

# In[3]:


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# In[4]:


use_cuda = torch.cuda.is_available()
torch.manual_seed(1)
device = torch.device("cuda" if use_cuda else "cpu")
# device = 'cpu'
torch.set_default_tensor_type('torch.cuda.FloatTensor')
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


# In[5]:


def mu_law_encode(audio, quantization_channels=quantization_channels):
    '''Quantizes waveform amplitudes.'''
    mu = (quantization_channels - 1) * 1.0
    # Perform mu-law companding transformation (ITU-T, 1988).
    # Minimum operation is here to deal with rare large amplitudes caused
    # by resampling.
    safe_audio_abs = np.minimum(np.abs(audio), 1.0)
    magnitude = np.log1p(mu * safe_audio_abs) / np.log1p(mu)
    signal = np.sign(audio) * magnitude
    # Quantize signal to the specified number of levels.
    # if(forX):return signal
    return ((signal + 1) / 2 * mu + 0.5).astype(int)


def mu_law_decode(output, quantization_channels=quantization_channels):
    '''Recovers waveform from quantized values.'''
    mu = quantization_channels - 1
    # Map values back to [-1, 1].
    signal = 2 * ((output * 1.0) / mu) - 1
    # Perform inverse of mu-law transformation.
    magnitude = (1 / mu) * ((1 + mu) ** np.abs(signal) - 1)
    return np.sign(signal) * magnitude


# In[6]:


def readAudio(name):
    audio0, samplerate = sf.read(name, dtype='float32')
    return librosa.resample(audio0.T, samplerate, sample_rate).reshape(-1)


p = ['./vsCorpus/origin_mix.wav', './vsCorpus/origin_vocal.wav',
     './vsCorpus/origin_mix.wav', './vsCorpus/origin_vocal.wav', './vsCorpus/pred_mix.wav']
xtrain, ytrain, xval, yval, xtest = readAudio(p[0]), readAudio(p[1]), readAudio(p[2]), readAudio(p[3]), readAudio(p[4])
assert ((xtrain == xval).all())
assert ((ytrain == yval).all())
assert ((xtrain != ytrain).any())

# In[7]:


xmean, xstd = xtrain.mean(), xtrain.std()
xtrain = (xtrain - xmean) / xstd
xval = (xval - xmean) / xstd
xtest = (xtest - xmean) / xstd
ytrain, yval = mu_law_encode(ytrain), mu_law_encode(yval)

# In[8]:


xtrain, ytrain = xtrain[:xtest.shape[0]], ytrain[:xtest.shape[0]]
xval, yval = xval[:xtest.shape[0]], yval[:xtest.shape[0]]
xtrain = np.pad(xtrain, (pad, pad), 'constant')
xval = np.pad(xval, (pad, pad), 'constant')
xtest = np.pad(xtest, (pad, pad), 'constant')
yval = np.pad(yval, (pad, pad), 'constant')
ytrain = np.pad(ytrain, (pad, pad), 'constant')

# In[9]:


# xtrain,ytrain,xval,yval=xtrain[:-sampleSize],ytrain[:-sampleSize],xval[-sampleSize:],yval[-sampleSize:]
# xtrain,ytrain,xval,yval=xtrain[:-sampleSize],ytrain[:-sampleSize],xval[:sampleSize],yval[:sampleSize]
xtrain, xval, xtest = xtrain.reshape(1, 1, -1), xval.reshape(1, 1, -1), xtest.reshape(1, 1, -1)
ytrain, yval = ytrain.reshape(1, -1), yval.reshape(1, -1)

# In[10]:


xtrain, ytrain, xval, yval, xtest = torch.from_numpy(xtrain).type(torch.float32), torch.from_numpy(ytrain).type(
    torch.LongTensor), torch.from_numpy(xval).type(torch.float32), torch.from_numpy(yval).type(
    torch.LongTensor), torch.from_numpy(xtest).type(torch.float32)


# In[11]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        sd, qd, rd = skipDim, quantization_channels, residualDim
        self.causal = nn.Conv1d(in_channels=1, out_channels=rd, kernel_size=3, padding=1)
        self.layer = dict()
        for i, d in enumerate(dilations):
            self.layer['tanh' + str(i)] = nn.Conv1d(in_channels=rd, out_channels=rd, kernel_size=3, padding=0,
                                                    dilation=d)
            self.layer['sigmoid' + str(i)] = nn.Conv1d(in_channels=rd, out_channels=rd, kernel_size=3, padding=0,
                                                       dilation=d)
            self.layer['skip' + str(i)] = nn.Conv1d(in_channels=rd, out_channels=sd, kernel_size=1, padding=0)
            self.layer['dense' + str(i)] = nn.Conv1d(in_channels=rd, out_channels=rd, kernel_size=1, padding=0)
        self.post1 = nn.Conv1d(in_channels=sd, out_channels=sd, kernel_size=1, padding=0)
        self.post2 = nn.Conv1d(in_channels=sd, out_channels=qd, kernel_size=1, padding=0)
        self.tanh, self.sigmoid = nn.Tanh(), nn.Sigmoid()

    def forward(self, x):
        finallen = x.shape[-1] - 2 * pad
        x = self.causal(x)
        # print('x.shape',x.shape)
        skip_connections = torch.zeros([1, skipDim, finallen], dtype=torch.float32, device=device)
        for i, dilation in enumerate(dilations):
            xinput = x.clone()[:, :, dilation:-dilation]
            x1 = self.tanh(self.layer['tanh' + str(i)](x))
            # print('tanh.shape',x1.shape)
            x2 = self.sigmoid(self.layer['sigmoid' + str(i)](x))
            # print('sigmoid.shape',x2.shape)
            x = x1 * x2
            # print('multi',x3.shape)
            cutlen = (x.shape[-1] - finallen) // 2
            skip_connections += (self.layer['skip' + str(i)](x)).narrow(2, int(cutlen), int(finallen))
            # cur =self.layer['skip'+str(i)](x3)
            x = self.layer['dense' + str(i)](x)
            # print('dense.shape',x.shape)
            x += xinput
        x = self.post2(F.relu(self.post1(F.relu(skip_connections))))
        return x


model = Net().cuda()
criterion = nn.CrossEntropyLoss().cuda()
# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)


# In[12]:


# torch.sum(c,dim=0,keepdim=True)


# In[13]:


# model = torch.load('torchmodel0.681600')


# In[ ]:


def val():
    model.eval()
    startval_time = time.time()
    with torch.no_grad():
        # data, target = xval.to(device), yval.to(device)
        data, target = xtrain[:, :, 0:2 * pad + shapeoftest].to(device), ytrain[:, pad:shapeoftest + pad].to(device)
        output = model(data)
        pred = output.max(1, keepdim=True)[1]
        correct = pred.eq(target.view_as(pred)).sum().item() / target.shape[1]
        print('accurate:', correct)
        val_loss = criterion(output, target).item()
    print('\nval set:loss{:.4f}:, ({:.3f} sec/step)\n'.format(val_loss, time.time() - startval_time))


def test():
    model.eval()
    startval_time = time.time()
    with torch.no_grad():
        output = model(xtest.to(device))
        pred = output.max(1, keepdim=True)[1].cpu().numpy().reshape(-1)
        ans = mu_law_decode(pred)
        sf.write('./vsCorpus/resultxte.wav', ans, sample_rate)

        # output = model(xtrain[:,:,:sampleSize].to(device))
        output = model(xtrain[:, :, 0:2 * pad + shapeoftest].to(device))
        pred = output.max(1, keepdim=True)[1].cpu().numpy().reshape(-1)
        ans = mu_law_decode(pred)
        sf.write('./vsCorpus/resultxtr.wav', ans, sample_rate)
        print('stored done\n')


def train(epoch):
    model.train()
    # idx = np.arange(xtrain.shape[-1] - 2 * sampleSize,1000)
    # 176000
    idx = np.arange(pad, shapeoftest + pad - sampleSize, 1000)
    np.random.shuffle(idx)
    for i, ind in enumerate(idx):
        start_time = time.time()
        data, target = xtrain[:, :, ind - pad:ind + sampleSize + pad].to(device), ytrain[:, ind:ind + sampleSize].to(
            device)
        output = model(data)
        # print(output.shape)
        loss = criterion(output, target)
        lossrecord.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss:{:.6f}: , ({:.3f} sec/step)'.format(
            epoch, i, len(idx), 100. * i / len(idx), loss.item(), time.time() - start_time))
        if i % 100 == 0:
            val()
            test()
            torch.save(model, 'padonlyonside')


# In[ ]:


for epoch in range(100000):
    train(epoch)
    test()


# In[ ]:


# model = torch.load('torchmodel')


# In[ ]:


# torch.save(model, 'loss23*1.+4*3.+16*2.step_9072_repeat15*2**8resu32sample50000')
