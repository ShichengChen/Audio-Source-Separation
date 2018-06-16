# coding: utf-8

# In[ ]:


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
from wavenet import Wavenet
from transformData import x_mu_law_encode, y_mu_law_encode, mu_law_decode, onehot, cateToSignal
from readDataset import Dataset

# In[ ]:


sampleSize = 32000  # the length of the sample size
quantization_channels = 256
sample_rate = 16000
dilations = [2 ** i for i in range(9)] * 5  # idea from wavenet, have more receptive field
residualDim = 128  #
skipDim = 512
shapeoftest = 190500
filterSize = 3
audioname='10801val.wav'
resumefile = './model/10801'  # name of checkpoint
lossname = '10801loss.txt'  # name of loss file
continueTrain = False  # whether use checkpoint
pad = np.sum(dilations)  # padding for dilate convolutional layers
lossrecord = []  # list for record loss
pad=0


# In[ ]:


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # use specific GPU


# In[ ]:


use_cuda = torch.cuda.is_available()
torch.manual_seed(1)
device = torch.device("cuda" if use_cuda else "cpu")
# torch.set_default_tensor_type('torch.cuda.FloatTensor')
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# In[ ]:


params = {'batch_size': 1, 'shuffle': True, 'num_workers': 1}

training_set = Dataset(np.arange(0, 1), np.arange(0, 1), 'ccmixter2/x/', 'ccmixter2/y/')
validation_set = Dataset(np.arange(0, 1), np.arange(0, 1), 'ccmixter2/x/', 'ccmixter2/y/')
loadtr = data.DataLoader(training_set, **params)
loadval = data.DataLoader(validation_set, **params)

# In[ ]:


model = Wavenet(pad, skipDim, quantization_channels, residualDim, dilations).cuda()
criterion = nn.CrossEntropyLoss()
# in wavenet paper, they said crossentropyloss is far better than MSELoss
# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
# use adam to train
# optimizer = optim.SGD(model.parameters(), lr = 0.1, momentum=0.9, weight_decay=1e-5)
# scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
# scheduler = MultiStepLR(optimizer, milestones=[20,40], gamma=0.1)


# In[ ]:


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


# In[ ]:


def val():  # validation set
    model.eval()
    startval_time = time.time()
    with torch.no_grad():
        for iloader, (xtrain, ytrain) in enumerate(loadval):
            idx = np.arange(pad, xtrain.shape[-1] - pad - sampleSize, 1000)
            np.random.shuffle(idx)
            data = xtrain[:, :, idx[0] - pad:pad + idx[0] + sampleSize].to(device)
            target = ytrain[:, idx[0]:idx[0] + sampleSize].to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct = pred.eq(target.view_as(pred)).sum().item() / pred.shape[-1]
            val_loss = criterion(output, target).item()
            print(correct, 'accurate')
            print('val set:loss{:.4f}:, ({:.3f} sec/step)\n'.format(val_loss, time.time() - startval_time))

            listofpred = []
            for ind in range(pad, xtrain.shape[-1] - pad - sampleSize, sampleSize):
                output = model(xtrain[:, :, ind - pad:ind + sampleSize + pad].to(device))
                pred = output.max(1, keepdim=True)[1].cpu().numpy().reshape(-1)
                listofpred.append(pred)
            ans = mu_law_decode(np.concatenate(listofpred))
            sf.write('./vsCorpus/'+audioname, ans, sample_rate)
            break


def train(epoch):  # training set
    model.train()
    for iloader, (xtrain, ytrain) in enumerate(loadtr):
        idx = np.arange(pad, xtrain.shape[-1] - pad - sampleSize, 16000)
        np.random.shuffle(idx)  # random the starting points
        for i, ind in enumerate(idx):
            start_time = time.time()
            data, target = xtrain[:, :, ind - pad:ind + sampleSize + pad].to(device), ytrain[:,
                                                                                      ind:ind + sampleSize].to(device)
            output = model(data)
            loss = criterion(output, target)
            lossrecord.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Train Epoch: {} iloader:{} [{}/{} ({:.0f}%)] Loss:{:.6f}: , ({:.3f} sec/step)'.format(
                epoch, iloader, i, len(idx), 100. * i / len(idx), loss.item(), time.time() - start_time))
            if i % 100 == 0:
                with open("./lossRecord/" + lossname, "w") as f:
                    for s in lossrecord:
                        f.write(str(s) + "\n")
                print('write finish')
                state = {'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict()}
                if not os.path.exists('./model/'): os.makedirs('./model/')
                torch.save(state, resumefile)
        val()


# In[ ]:


for epoch in range(100000):
    train(epoch)
