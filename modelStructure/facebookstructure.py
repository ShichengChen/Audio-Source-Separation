import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class Wavenet(nn.Module):
    def __init__(self, pad, sd, qd, rd, domain, device):
        self.dilations0 = [2 ** i for i in range(9)] * 3
        self.dilations1 = [2 ** i for i in range(9)] * 4
        self.sd = sd
        super(Wavenet, self).__init__()
        self.causal = nn.Conv1d(in_channels=1, out_channels=rd, kernel_size=25, padding=12)
        # normal cnn
        self.pad = pad
        # please notice that you cannot use self.sigmoidconvs=dict(), otherwise the layers in the dict() can
        # not update the weights
        self.dcnn = nn.ModuleList()
        self.skipconvs = nn.ModuleList()
        self.denseconvs = nn.ModuleList()
        self.device = device
        for i, d in enumerate(self.dilations0):
            if (pad == 0):
                self.dcnn.append(nn.Conv1d(in_channels=rd, out_channels=rd, kernel_size=3, padding=d, dilation=d))
            else:
                self.dcnn.append(nn.Conv1d(in_channels=rd, out_channels=rd, kernel_size=3, padding=0, dilation=d))
            self.skipconvs.append(nn.Conv1d(in_channels=rd, out_channels=sd, kernel_size=1))
            self.denseconvs.append(nn.Conv1d(in_channels=rd, out_channels=rd, kernel_size=1))

        for i, d in enumerate(self.dilations1):
            if (pad == 0):
                self.dcnn.append(nn.Conv1d(in_channels=rd, out_channels=rd, kernel_size=3, padding=d, dilation=d))
            else:
                self.dcnn.append(nn.Conv1d(in_channels=rd, out_channels=rd, kernel_size=3, padding=0, dilation=d))
            self.skipconvs.append(nn.Conv1d(in_channels=rd, out_channels=sd, kernel_size=1))
            self.denseconvs.append(nn.Conv1d(in_channels=rd, out_channels=rd, kernel_size=1))

        self.post1 = nn.Conv1d(in_channels=sd, out_channels=sd, kernel_size=1)
        self.post2 = nn.Conv1d(in_channels=sd, out_channels=qd, kernel_size=1)

    def forward(self, x):
        finallen = x.shape[-1] - 2 * self.pad
        x = self.causal(x)
        skip_connections = torch.zeros([x.shape[0], self.sd, finallen], dtype=torch.float32, device=self.device)
        cnt=0
        for d in self.dilations0:
            if (self.pad == 0):
                xinput = x.clone()
            else:
                xinput = x.clone()[:, :, d:-d]
            x=F.relu(x)
            x = F.relu(self.dcnn[cnt](x))
            x = self.denseconvs[cnt](x)
            x += xinput
            cnt+=1



        for d in self.dilations1:
            if (self.pad == 0):
                xinput = x.clone()
            else:
                xinput = x.clone()[:, :, d:-d]
            x=F.relu(x)
            x = F.relu(self.dcnn[cnt](x))
            cutlen = (x.shape[-1] - finallen) // 2
            if (self.pad == 0):
                skip_connections += (self.skipconvs[cnt](x))
            else:
                skip_connections += (self.skipconvs[cnt](x)).narrow(2, int(cutlen), int(finallen))
            x = self.denseconvs[cnt](x)
            x += xinput
            cnt += 1

        x = self.post2(F.relu(self.post1(F.relu(skip_connections))))
        return x#,domainx



'''
class GRL(Function):
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x) * constant

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.constant, None

    # self.elu=nn.ELU()
    # self.con0 = nn.Conv1d(rd, rd, 1)
    # self.con1 = nn.Conv1d(rd, rd, 1)
    # self.con2 = nn.Conv1d(rd, domain, 1)
    # domainx=GRL.apply(x, 0.01)
    # domainx = self.elu(self.con0(domainx))
    # domainx = self.elu(self.con1(domainx))
    # domainx = self.elu(self.con2(domainx))
    # domainx=F.avg_pool1d(domainx,kernel_size=domainx.shape[-1])

'''