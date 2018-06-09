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
from torch.optim.lr_scheduler import StepLR,MultiStepLR
# In[2]:


sampleSize = 50000
sample_rate = 16000
quantization_channels = 256
# dilations=[2**i for i in range(8)]*20
# "residualDim=32
dilations = [2 ** i for i in range(9)] * 5
residualDim = 32
skipDim = 512
filterSize = 3
shapeoftest = 190500
lossrecord = []
initfilter=33
resumefile='allmulawalldata'
continueTrain=False
pad = np.sum(dilations) + initfilter//2
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


def mu_law_encode(audio, quantization_channels=quantization_channels, forX=False):
    '''Quantizes waveform amplitudes.'''
    mu = (quantization_channels - 1) * 1.0
    # Perform mu-law companding transformation (ITU-T, 1988).
    # Minimum operation is here to deal with rare large amplitudes caused
    # by resampling.
    safe_audio_abs = np.minimum(np.abs(audio), 1.0)
    magnitude = np.log1p(mu * safe_audio_abs) / np.log1p(mu)
    signal = np.sign(audio) * magnitude
    # Quantize signal to the specified number of levels.
    if (forX): return signal
    return ((signal + 1) / 2 * mu + 0.5).astype(int)


def mu_law_decode(output, quantization_channels=quantization_channels):
    '''Recovers waveform from quantized values.'''
    mu = quantization_channels - 1
    # Map values back to [-1, 1].
    signal = 2 * ((output * 1.0) / mu) - 1
    # Perform inverse of mu-law transformation.
    magnitude = (1 / mu) * ((1 + mu) ** np.abs(signal) - 1)
    return np.sign(signal) * magnitude


# In[8]:


def readAudio(name):
    audio0, samplerate = sf.read(name, dtype='float32')
    return librosa.resample(audio0.T, samplerate, sample_rate).reshape(-1)


p = ['./vsCorpus/origin_mix.wav', './vsCorpus/origin_vocal.wav',
     './vsCorpus/origin_mix.wav', './vsCorpus/origin_vocal.wav', './vsCorpus/pred_mix.wav']
xtrain, ytrain, xval, yval, xtest = readAudio(p[0]), readAudio(p[1]), readAudio(p[2]), readAudio(p[3]), readAudio(p[4])
assert ((xtrain == xval).all())
assert ((ytrain == yval).all())
assert ((xtrain != ytrain).any())

# In[9]:


ytrain, yval = mu_law_encode(ytrain), mu_law_encode(yval)
xtrain, xval, xtest = mu_law_encode(xtrain, forX=True), mu_law_encode(xval, forX=True), mu_law_encode(xtest, forX=True)

# In[7]:


xmean,xstd = xtrain.mean(),xtrain.std()
xtrain=(xtrain-xmean)/xstd
xval=(xval-xmean)/xstd
xtest=(xtest-xmean)/xstd

# In[12]:


# xtrain,ytrain=xtrain[:xtest.shape[0]],ytrain[:xtest.shape[0]]
# xval,yval=xval[:xtest.shape[0]],yval[:xtest.shape[0]]
xtrain = np.pad(xtrain, (pad, pad), 'constant')
xval = np.pad(xval, (pad, pad), 'constant')
xtest = np.pad(xtest, (pad, pad), 'constant')
yval = np.pad(yval, (pad, pad), 'constant')
ytrain = np.pad(ytrain, (pad, pad), 'constant')

# In[13]:


# xtrain,ytrain,xval,yval=xtrain[:-sampleSize],ytrain[:-sampleSize],xval[-sampleSize:],yval[-sampleSize:]
# xtrain,ytrain,xval,yval=xtrain[:-sampleSize],ytrain[:-sampleSize],xval[:sampleSize],yval[:sampleSize]
xtrain, xval, xtest = xtrain.reshape(1, 1, -1), xval.reshape(1, 1, -1), xtest.reshape(1, 1, -1)
ytrain, yval = ytrain.reshape(1, -1), yval.reshape(1, -1)

# In[14]:


xtrain, ytrain, xval, yval, xtest = torch.from_numpy(xtrain).type(torch.float32), torch.from_numpy(ytrain).type(
    torch.LongTensor), torch.from_numpy(xval).type(torch.float32), torch.from_numpy(yval).type(
    torch.LongTensor), torch.from_numpy(xtest).type(torch.float32)


# In[15]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        sd, qd, rd = skipDim, quantization_channels, residualDim
        self.causal = nn.Conv1d(in_channels=1, out_channels=rd, kernel_size=initfilter, padding=0)
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
#optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)*
optimizer = optim.SGD(model.parameters(), lr = 0.1, momentum=0.9, weight_decay=1e-4)
#scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)

if continueTrain:
    if os.path.isfile(resumefile):
        print("=> loading checkpoint '{}'".format(resumefile))
        checkpoint = torch.load(resumefile)
        start_epoch = checkpoint['epoch']
        #best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resumefile, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resumefile))


# In[18]:


def val():
    model.eval()
    startval_time = time.time()
    with torch.no_grad():
        # data, target = xval.to(device), yval.to(device)
        data, target = xtrain[:, :, 0:2 * pad + shapeoftest].to(device), ytrain[:, pad:shapeoftest + pad].to(device)
        output = model(data)
        # print(output.shape)
        # print(output[:,:,:10])
        pred = output.max(1, keepdim=True)[1]
        correct = pred.eq(target.view_as(pred)).sum().item() / pred.shape[-1]
        val_loss = criterion(output, target).item()
    print(correct, 'accurate')
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
        ind = pad
        listofpred=[]
        while ind < xtrain.shape[-1]-pad:
            output = model(xtrain[:, :, ind-pad:ind+sampleSize+pad].to(device))
            pred = output.max(1, keepdim=True)[1].cpu().numpy().reshape(-1)
            listofpred.append(pred)
            ind += sampleSize
        ans = mu_law_decode(np.concatenate(listofpred))
        sf.write('./vsCorpus/resultxtr.wav', ans, sample_rate)
        print('stored done\n')


def train(epoch):
    model.train()
    # idx = np.arange(xtrain.shape[-1] - 2 * sampleSize,1000)
    # 176000
    idx = np.arange(pad, xtrain.shape[-1] - pad - sampleSize, 16000)
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
            state={'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()}
            torch.save(state, resumefile)
    val()
    test()


# In[ ]:


for epoch in range(100000):
    train(epoch)
    scheduler.step()
    with open("lossfile.txt", "w") as f:
        for s in lossrecord:
            f.write(str(s) +"\n")
    print('write finish')

# In[20]:


# torch.save(model, 'loss3.2step24h_repeat5*2**10resu96sample50000')
