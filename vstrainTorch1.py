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


sampleSize = 32000
sample_rate = 16000
quantization_channels = 256
# dilations=[2**i for i in range(8)]*20
# "residualDim=32
dilations = [2 ** i for i in range(9)] * 5
residualDim = 128
skipDim = 512
filterSize = 3
shapeoftest = 190500
lossrecord = []
initfilter=3
resumefile='middlemodel'
continueTrain=True
pad = np.sum(dilations) + initfilter//2
pad=0
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

def onehot(a,mu=quantization_channels):
    b = np.zeros((a.shape[0], mu))
    b[np.arange(a.shape[0]), a] = 1
    return b
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
xtrain, xval, xtest = mu_law_encode(xtrain,forX=True), mu_law_encode(xval,forX=True), mu_law_encode(xtest,forX=True)
assert (xtrain.max()<=2 and ytrain.max() >= 5)
#xtrain, xval, xtest = onehot(xtrain),onehot(xval),onehot(xtest)
# In[7]:



#xmean,xstd = xtrain.mean(),xtrain.std()
#xtrain=(xtrain-xmean)/xstd
#xval=(xval-xmean)/xstd
#xtest=(xtest-xmean)/xstd

# In[12]:


# xtrain,ytrain=xtrain[:xtest.shape[0]],ytrain[:xtest.shape[0]]
# xval,yval=xval[:xtest.shape[0]],yval[:xtest.shape[0]]
#xtrain = np.pad(xtrain, (pad, pad), 'constant')
#xval = np.pad(xval, (pad, pad), 'constant')
#xtest = np.pad(xtest, (pad, pad), 'constant')
#yval = np.pad(yval, (pad, pad), 'constant')
#ytrain = np.pad(ytrain, (pad, pad), 'constant')

# In[13]:


# xtrain,ytrain,xval,yval=xtrain[:-sampleSize],ytrain[:-sampleSize],xval[-sampleSize:],yval[-sampleSize:]
# xtrain,ytrain,xval,yval=xtrain[:-sampleSize],ytrain[:-sampleSize],xval[:sampleSize],yval[:sampleSize]
xtrain, xval, xtest = xtrain.reshape(1, 1, -1), xval.reshape(1, 1, -1), \
                      xtest.reshape(1, 1, -1)
ytrain, yval = ytrain.reshape(1, -1), yval.reshape(1, -1)

# In[14]:


xtrain, ytrain, xval, yval, xtest = torch.from_numpy(xtrain).type(torch.float32), torch.from_numpy(ytrain).type(
    torch.LongTensor), torch.from_numpy(xval).type(torch.float32), torch.from_numpy(yval).type(
    torch.LongTensor), torch.from_numpy(xtest).type(torch.float32)


# In[15]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        sd,qd,rd = skipDim,quantization_channels,residualDim
        self.causal = nn.Conv1d(in_channels=1,out_channels=rd,kernel_size=3,padding=1)
        self.tanhconvs = nn.ModuleList()
        self.sigmoidconvs = nn.ModuleList()
        self.skipconvs = nn.ModuleList()
        self.denseconvs = nn.ModuleList()
        for i, d in enumerate(dilations):
            self.tanhconvs.append(nn.Conv1d(in_channels=rd,out_channels=rd,kernel_size=3,padding=d,dilation=d))
            self.sigmoidconvs.append(nn.Conv1d(in_channels=rd,out_channels=rd,kernel_size=3,padding=d,dilation=d))
            self.skipconvs.append(nn.Conv1d(in_channels=rd,out_channels=sd,kernel_size=1))
            self.denseconvs.append(nn.Conv1d(in_channels=rd,out_channels=rd,kernel_size=1))
        self.post1 = nn.Conv1d(in_channels=sd,out_channels=sd,kernel_size=1)
        self.post2 = nn.Conv1d(in_channels=sd,out_channels=qd,kernel_size=1)
        self.tanh,self.sigmoid = nn.Tanh(),nn.Sigmoid()

    def forward(self, x):
        #finallen = x.shape[-1]-2*pad
        x = self.causal(x)
        for i, dilation in enumerate(dilations):
            xinput = x.clone()#[:,:,dilation:-dilation]
            x1 = self.tanh(self.tanhconvs[i](x))
            x2 = self.sigmoid(self.sigmoidconvs[i](x))
            x = x1*x2
            #cutlen = (x.shape[-1] - finallen)//2
            if(i == 0):skip_connections= (self.skipconvs[i](x))#.narrow(2,int(cutlen),int(finallen))
            else :skip_connections += (self.skipconvs[i](x))#.narrow(2,int(cutlen),int(finallen))
            x = self.denseconvs[i](x)
            x += xinput
        x = self.post2(F.relu(self.post1(F.relu(skip_connections))))
        return x



model = Net().cuda()
criterion = nn.CrossEntropyLoss().cuda()
#optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
optimizer = optim.Adam(model.parameters(), lr=1e-3,weight_decay=1e-5)
#optimizer = optim.SGD(model.parameters(), lr = 0.1, momentum=0.9, weight_decay=1e-4)
#scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
#scheduler = MultiStepLR(optimizer, milestones=[20,40], gamma=0.1)

if continueTrain:
    if os.path.isfile(resumefile):
        print("=> loading checkpoint '{}'".format(resumefile))
        checkpoint = torch.load(resumefile)
        start_epoch = checkpoint['epoch']
        #best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resumefile, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resumefile))


# In[18]:


def val():
    model.eval()
    startval_time = time.time()
    with torch.no_grad():
        idx = np.arange(xtrain.shape[-1]-pad-10*sampleSize,xtrain.shape[-1]-pad-sampleSize,1000)
        np.random.shuffle(idx)
        data = xtrain[:,:,idx[0]-pad:pad+idx[0]+sampleSize].to(device)
        target = ytrain[:,idx[0]:idx[0]+sampleSize].to(device)
        output = model(data)
        pred = output.max(1, keepdim=True)[1]
        correct = pred.eq(target.view_as(pred)).sum().item() / pred.shape[-1]
        val_loss = criterion(output, target).item()

        listofpred = []
        for ind in range(xtrain.shape[-1]-pad-10*sampleSize,xtrain.shape[-1]-pad-sampleSize,sampleSize):
            output = model(xtrain[:, :, ind - pad:ind + sampleSize + pad].to(device))
            pred = output.max(1, keepdim=True)[1].cpu().numpy().reshape(-1)
            listofpred.append(pred)
        ans = mu_law_decode(np.concatenate(listofpred))
        sf.write('./vsCorpus/xval.wav', ans, sample_rate)
    print(correct,'accurate')
    print('\nval set:loss{:.4f}:, ({:.3f} sec/step)\n'.format(val_loss,time.time()-startval_time))


def test():
    model.eval()
    startval_time = time.time()
    with torch.no_grad():
        listofpred = []
        for ind in range(pad, xtest.shape[-1] - pad, sampleSize):
            output = model(xtest[:, :, ind - pad:ind + sampleSize + pad].to(device))
            pred = output.max(1, keepdim=True)[1].cpu().numpy().reshape(-1)
            listofpred.append(pred)
        ans = mu_law_decode(np.concatenate(listofpred))
        sf.write('./vsCorpus/xte.wav', ans, sample_rate)

        listofpred=[]
        for ind in range(pad,xtrain.shape[-1]-pad,sampleSize):
            output = model(xtrain[:, :, ind-pad:ind+sampleSize+pad].to(device))
            pred = output.max(1, keepdim=True)[1].cpu().numpy().reshape(-1)
            listofpred.append(pred)
        ans = mu_law_decode(np.concatenate(listofpred))
        sf.write('./vsCorpus/xtr.wav', ans, sample_rate)
        print('stored done\n')


def train(epoch):
    model.train()
    idx = np.arange(pad,xtrain.shape[-1]-pad-10*sampleSize,16000)
    np.random.shuffle(idx)
    for i, ind in enumerate(idx):
        start_time = time.time()
        data, target = xtrain[:,:,ind-pad:ind+sampleSize+pad].to(device), ytrain[:,ind:ind+sampleSize].to(device)
        output = model(data)
        loss = criterion(output, target)
        lossrecord.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss:{:.6f}: , ({:.3f} sec/step)'.format(
                epoch, i, len(idx),100. * i / len(idx), loss.item(),time.time() - start_time))
        if i % 100 == 0:
            with open("lossfile.txt", "w") as f:
                for s in lossrecord:
                    f.write(str(s) +"\n")
            print('write finish')
            state={'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()}
            torch.save(state, resumefile)
    val()
    test()


# In[ ]:


for epoch in range(100000):
    train(epoch)
    #scheduler.step()
    with open("lossfile.txt", "w") as f:
        for s in lossrecord:
            f.write(str(s) +"\n")
    print('write finish')


# torch.save(model, 'loss3.2step24h_repeat5*2**10resu96sample50000')
