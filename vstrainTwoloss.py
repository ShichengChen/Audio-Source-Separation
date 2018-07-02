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
from wavenet2 import Wavenet
from transformData import x_mu_law_encode,y_mu_law_encode,mu_law_decode,onehot,cateToSignal
from readDataset3 import Dataset,Testset,RandomCrop,ToTensor
from unet import Unet
import h5py

# In[2]:


sampleSize = 16000  # the length of the sample size
quantization_channels = 256
sample_rate = 16000
dilations = [2 ** i for i in range(9)] * 3  # idea from wavenet, have more receptive field
residualDim = 256  #
skipDim = 512
shapeoftest = 190500
songnum=45
filterSize = 3
savemusic='vsCorpus/nus0xtr{}.wav'
#savemusic0='vsCorpus/nus10xtr{}.wav'
#savemusic1='vsCorpus/nus11xtr{}.wav'
resumefile = 'model/instrument0'  # name of checkpoint
lossname = 'instrumentloss0.txt'  # name of loss file
continueTrain = False  # whether use checkpoint
pad = np.sum(dilations)  # padding for dilate convolutional layers
lossrecord = []  # list for record loss
sampleCnt=0
#pad=0


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # use specific GPU

# In[4]:


use_cuda = torch.cuda.is_available()  # whether have available GPU
torch.manual_seed(1)
device = torch.device("cuda" if use_cuda else "cpu")
# device = 'cpu'
# torch.set_default_tensor_type('torch.cuda.FloatTensor') #set_default_tensor_type as cuda tensor


transform=transforms.Compose([RandomCrop(pad=pad),ToTensor()])
training_set = Dataset(np.array([6,7,8,9,11,12,14,16,26]), np.array([6,7,8,9,11,12,14,16,26]), 'ccmixter3/x/', 'ccmixter3/y/',pad=pad,transform=transform)
validation_set = Testset(np.array([6,7,8,9,11,12,14,16,26]), 'ccmixter3/x/',pad=pad)
loadtr = data.DataLoader(training_set, batch_size=1,shuffle=True,num_workers=1)  # pytorch dataloader, more faster than mine
loadval = data.DataLoader(validation_set,batch_size=1,num_workers=1)

# In[6]:

#model = Unet(skipDim, quantization_channels, residualDim,device)
model = Wavenet(pad, skipDim, quantization_channels, residualDim, dilations,device)
model = nn.DataParallel(model)
model = model.cuda()
criterion = nn.CrossEntropyLoss()
# in wavenet paper, they said crossentropyloss is far better than MSELoss
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
# use adam to train

maxloss=np.zeros(50)+100
# In[7]:

start_epoch=0
if continueTrain:  # if continueTrain, the program will find the checkpoints
    if os.path.isfile(resumefile):
        print("=> loading checkpoint '{}'".format(resumefile))
        checkpoint = torch.load(resumefile)
        start_epoch = checkpoint['epoch']
        # best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resumefile, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resumefile))


# In[9]:


def test(epoch):  # testing data
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for iloader, xtrain, ytrain in loadval:
            iloader=iloader.item()
            listofpred0 = []
            cnt,aveloss=0,0
            for ind in range(pad, xtrain.shape[-1] - pad - sampleSize, sampleSize):
                output0 = model(xtrain[:, :, ind - pad:ind + sampleSize + pad].to(device))
                loss = criterion(output0, (ytrain[:, ind:ind + sampleSize].to(device)))
                pred0 = output0.max(1, keepdim=True)[1].cpu().numpy().reshape(-1)
                listofpred0.append(pred0)
                cnt+=1
                aveloss+=loss.item()
            aveloss /= cnt
            print('loss for validation:{},num{},epoch{}'.format(aveloss, iloader,epoch))
            #if(aveloss < maxloss[iloader]):
            maxloss[iloader] = aveloss
            ans0 = mu_law_decode(np.concatenate(listofpred0))
            if not os.path.exists('vsCorpus/'): os.makedirs('vsCorpus/')
            sf.write(savemusic.format(iloader), ans0, sample_rate)
            print('test stored done', np.round(time.time() - start_time))



def train(epoch):  # training data, the audio except for last 15 seconds
    for iloader,xtrain, ytrain in loadtr:
        startx = np.random.randint(0,sampleSize)
        idx = np.arange(startx + pad, xtrain.shape[-1] - pad - sampleSize, sampleSize * 3)
        np.random.shuffle(idx)
        lens = 25
        idx = idx[:lens]
        cnt, aveloss = 0, 0
        start_time = time.time()
        for i, ind in enumerate(idx):
            model.train()
            data = (xtrain[:, :, ind - pad:ind + sampleSize + pad]).to(device)
            target0 = ytrain[:, ind:ind + sampleSize].to(device)
            output = model(data)
            loss = criterion(output, target0)
            aveloss+=loss
            cnt+=1
            lossrecord.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global sampleCnt
            sampleCnt+=1
            #print('Train Epoch: {} iloader:{},{},{} Loss:{:.6f}: , ({:.3f} sec/step)'.format(
            #    epoch, iloader,i, idx.shape[-1], loss.item(), time.time() - start_time))
            if sampleCnt % 10000 == 0 and sampleCnt > 0:
                for param in optimizer.param_groups:
                    param['lr'] *= 0.98
        print('loss for train:{:.3f},num{},epoch{},({:.3f} sec/step)'.format(
            aveloss / cnt, iloader, epoch,time.time() - start_time))
    with open("lossRecord/" + lossname, "w") as f:
        for s in lossrecord:
            f.write(str(s) + "\n")
    print('write finish')
    if not os.path.exists('model/'): os.makedirs('model/')
    state = {'epoch': epoch,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, resumefile)


# In[ ]:

print('training...')
for epoch in range(100000):
    train(epoch+start_epoch)
    if epoch % 16 == 0 and epoch > 0: test(epoch+start_epoch)