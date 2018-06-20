import torch
import torch.nn as nn
import torch.nn.functional as F


class Wavenet(nn.Module):
    def __init__(self, pad, sd, qd, rd, dilations, device):
        self.dilations = dilations
        self.sd = sd
        super(Wavenet, self).__init__()
        self.causal = nn.Conv1d(in_channels=1, out_channels=rd, kernel_size=3, padding=1)
        # normal cnn
        self.pad = pad
        # please notice that you cannot use self.sigmoidconvs=dict(), otherwise the layers in the dict() can
        # not update the weights
        self.dcnn = nn.ModuleList()
        self.skipconvs = nn.ModuleList()
        self.denseconvs = nn.ModuleList()
        self.device = device
        for i, d in enumerate(dilations):
            if (pad == 0):
                self.dcnn.append(nn.Conv1d(in_channels=rd, out_channels=rd, kernel_size=3, padding=d, dilation=d))
            else:
                self.dcnn.append(nn.Conv1d(in_channels=rd, out_channels=rd, kernel_size=3, padding=0, dilation=d))
            # 1d cnn with dilation, if you don't know dilate
            # please check https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
            self.skipconvs.append(nn.Conv1d(in_channels=rd, out_channels=sd, kernel_size=1))
            self.denseconvs.append(nn.Conv1d(in_channels=rd, out_channels=rd, kernel_size=1))
        self.post1 = nn.Conv1d(in_channels=sd, out_channels=sd, kernel_size=1)
        # normal cnn
        self.post2 = nn.Conv1d(in_channels=sd, out_channels=qd, kernel_size=1)

    def forward(self, x):
        finallen = x.shape[-1] - 2 * self.pad
        x = self.causal(x)  # normal cnn
        skip_connections = torch.zeros([x.shape[0], self.sd, finallen], dtype=torch.float32, device=self.device)
        for i, d in enumerate(self.dilations):
            if (self.pad == 0):
                xinput = x.clone()
            else:
                xinput = x.clone()[:, :, d:-d]
            x=F.relu(x)
            # when you do 1d cnn with dilation, and you do not padding, you need to do this
            x = F.relu(self.dcnn[i](x))
            cutlen = (x.shape[-1] - finallen) // 2
            if (self.pad == 0):
                skip_connections += (self.skipconvs[i](x))
            else:
                skip_connections += (self.skipconvs[i](x)).narrow(2, int(cutlen), int(finallen))
            # inspired by resnet
            x = self.denseconvs[i](x)
            # fc 1d cnn kernel size =1
            x += xinput
        x = self.post2(F.relu(self.post1(F.relu(skip_connections))))
        return x