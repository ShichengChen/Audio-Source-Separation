import torch
import torch.nn as nn
import torch.nn.functional as F


class Unet(nn.Module):
    def __init__(self, pad,initf,otherf,sd, outc, rd, blocknum,device):
        self.sd = sd
        self.pad=pad
        self.initf=initf
        self.otherf=otherf
        super(Unet, self).__init__()
        if(pad>0):
            self.causal = nn.Conv1d(in_channels=1026, out_channels=rd, kernel_size=initf, padding=0)
        else:
            self.causal = nn.Conv1d(in_channels=1026, out_channels=rd, kernel_size=initf, padding=initf//2)
        self.blocknum=blocknum
        # normal cnn
        # please notice that you cannot use self.sigmoidconvs=dict(), otherwise the layers in the dict() can
        # not update the weights
        self.dcnn = nn.ModuleList()
        self.skipconvs = nn.ModuleList()
        self.denseconvs = nn.ModuleList()
        self.device = device
        for i in range(self.blocknum):
            if (pad > 0):
                self.dcnn.append(nn.Conv1d(in_channels=rd, out_channels=rd, kernel_size=otherf, padding=0))
            else:
                self.dcnn.append(nn.Conv1d(in_channels=rd, out_channels=rd, kernel_size=otherf, padding=otherf//2))

            # 1d cnn with dilation, if you don't know dilate
            # please check https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
            self.skipconvs.append(nn.Conv1d(in_channels=rd, out_channels=sd, kernel_size=1))
            self.denseconvs.append(nn.Conv1d(in_channels=rd, out_channels=rd, kernel_size=1))
        self.post1 = nn.Conv1d(in_channels=sd, out_channels=sd, kernel_size=1)
        # normal cnn
        self.post2 = nn.Conv1d(in_channels=sd, out_channels=outc, kernel_size=1)

    def forward(self, x):
        finallen = x.shape[-1] - 2 * self.pad
        x = self.causal(x)  # normal cnn
        skip_connections = torch.zeros([x.shape[0], self.sd, finallen], dtype=torch.float32, device=self.device)
        for i in range(self.blocknum):
            if (self.pad == 0):
                xinput = x.clone()
            else:
                xinput = x.clone()[:, :, self.otherf//2:-(self.otherf//2)]

            x=F.leaky_relu(x)
            # when you do 1d cnn with dilation, and you do not padding, you need to do this
            cutlen = (x.shape[-1] - finallen) // 2
            if (self.pad == 0):
                skip_connections += (self.skipconvs[i](x))
            else:
                skip_connections += (self.skipconvs[i](x)).narrow(2, int(cutlen), int(finallen))
            x = F.leaky_relu(self.dcnn[i](x))
            #skip_connections += (self.skipconvs[i](x))
            # inspired by resnet
            x = self.denseconvs[i](x)
            # fc 1d cnn kernel size =1
            x += xinput
        x = self.post2(F.leaky_relu(self.post1(F.leaky_relu(skip_connections))))
        return x