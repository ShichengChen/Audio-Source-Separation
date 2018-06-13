import torch
import torch.nn as nn
import torch.nn.functional as F

class Wavenet(nn.Module):
    def __init__(self,pad,sd,qd,rd,dilations):
        self.dilations=dilations
        super(Wavenet, self).__init__()
        self.causal = nn.Conv1d(in_channels=1,out_channels=rd,kernel_size=3,padding=1)
        #normal cnn
        self.pad=pad
        self.tanhconvs = nn.ModuleList()
        #please notice that you cannot use self.sigmoidconvs=dict(), otherwise the layers in the dict() can 
        #not update the weights
        self.sigmoidconvs = nn.ModuleList()
        self.skipconvs = nn.ModuleList()
        self.denseconvs = nn.ModuleList()
        for i, d in enumerate(dilations):
            if(pad==0):
                #if do not have padding at the begin, when i use 1d_cnn, i need to padding for dilation
                self.tanhconvs.append(nn.Conv1d(in_channels=rd,out_channels=rd,kernel_size=3,padding=d,dilation=d))
            else:
                self.tanhconvs.append(nn.Conv1d(in_channels=rd,out_channels=rd,kernel_size=3,padding=0,dilation=d))
            #1d cnn with dilation, if you don't know dilate
            #please check https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
            if(pad==0):
                self.sigmoidconvs.append(nn.Conv1d(in_channels=rd,out_channels=rd,kernel_size=3,padding=d,dilation=d))
            else:
                self.sigmoidconvs.append(nn.Conv1d(in_channels=rd,out_channels=rd,kernel_size=3,padding=0,dilation=d))
            self.skipconvs.append(nn.Conv1d(in_channels=rd,out_channels=sd,kernel_size=1))
            self.denseconvs.append(nn.Conv1d(in_channels=rd,out_channels=rd,kernel_size=1))
        self.post1 = nn.Conv1d(in_channels=sd,out_channels=sd,kernel_size=1)
        # normal cnn 
        self.post2 = nn.Conv1d(in_channels=sd,out_channels=qd,kernel_size=1)
        self.tanh,self.sigmoid = nn.Tanh(),nn.Sigmoid()

    def forward(self, x):
        finallen = x.shape[-1]-2*self.pad
        x = self.causal(x)#normal cnn
        for i, d in enumerate(self.dilations):
            if(self.pad==0):xinput = x.clone()
            else:xinput = x.clone()[:,:,d:-d]
            #when you do 1d cnn with dilation, and you do not padding, you need to do this
            
            x1 = self.tanh(self.tanhconvs[i](x))
            x2 = self.sigmoid(self.sigmoidconvs[i](x))
            x = x1*x2
            cutlen = (x.shape[-1] - finallen)//2
            if(self.pad == 0):
                if(i == 0):skip_connections = (self.skipconvs[i](x))
                else:skip_connections += (self.skipconvs[i](x))
            else:
                if(i == 0):skip_connections= (self.skipconvs[i](x)).narrow(2,int(cutlen),int(finallen))
                else:skip_connections += (self.skipconvs[i](x)).narrow(2,int(cutlen),int(finallen))
            #inspired by resnet
            x = self.denseconvs[i](x)
            #fc 1d cnn kernel size =1
            x += xinput
        x = self.post2(F.relu(self.post1(F.relu(skip_connections))))
        return x