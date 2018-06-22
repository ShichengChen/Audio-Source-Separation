import torch
import torch.nn as nn
import torch.nn.functional as F


class Unet(nn.Module):
    def __init__(self, sd, qd, rd, device):
        self.sd = sd
        super(Unet, self).__init__()
        self.causal = nn.Conv1d(in_channels=1, out_channels=rd, kernel_size=25, padding=12)
        # normal cnn
        # please notice that you cannot use self.sigmoidconvs=dict(), otherwise the layers in the dict() can
        # not update the weights
        self.dcnn = nn.ModuleList()
        self.skipconvs = nn.ModuleList()
        self.denseconvs = nn.ModuleList()
        self.device = device
        for i in range(70):
            self.dcnn.append(nn.Conv1d(in_channels=rd, out_channels=rd, kernel_size=15, padding=7))
            # 1d cnn with dilation, if you don't know dilate
            # please check https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
            self.skipconvs.append(nn.Conv1d(in_channels=rd, out_channels=sd, kernel_size=1))
            self.denseconvs.append(nn.Conv1d(in_channels=rd, out_channels=rd, kernel_size=1))
        self.post1 = nn.Conv1d(in_channels=sd, out_channels=sd, kernel_size=1)
        # normal cnn
        self.post2 = nn.Conv1d(in_channels=sd, out_channels=qd, kernel_size=1)

    def forward(self, x):
        x = self.causal(x)  # normal cnn
        skip_connections = torch.zeros([x.shape[0], self.sd, x.shape[-1]], dtype=torch.float32, device=self.device)
        for i in range(70):
            xinput = x.clone()
            x=F.relu(x)
            # when you do 1d cnn with dilation, and you do not padding, you need to do this
            x = F.relu(self.dcnn[i](x))
            skip_connections += (self.skipconvs[i](x))
            # inspired by resnet
            x = self.denseconvs[i](x)
            # fc 1d cnn kernel size =1
            x += xinput
        x = self.post2(F.relu(self.post1(F.relu(skip_connections))))
        return x