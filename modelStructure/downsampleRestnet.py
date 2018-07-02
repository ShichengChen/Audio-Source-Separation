import torch
import torch.nn as nn
import torch.nn.functional as F


class Uresnet(nn.Module):
    def __init__(self, sd, qd, rd, device):
        self.sd = sd
        super(Uresnet, self).__init__()
        self.causal0 = nn.Conv1d(1, rd, 25, padding=12)
        self.dcnn = nn.ModuleList()
        self.skipconvs = nn.ModuleList()
        self.denseconvs = nn.ModuleList()
        self.device = device
        for j in range(5):
            for i in range(3):
                self.skipconvs.append(nn.Conv1d(rd, rd, 1))
                self.dcnn.append(nn.Conv1d(rd, rd, 15, padding=7))
                self.denseconvs.append(nn.Conv1d(rd, rd, 1))
            self.dcnn.append(nn.Conv1d(rd, rd, 3, padding=1, stride=2))
            self.denseconvs.append(nn.Conv1d(rd, rd, 1))
        self.mid0 = nn.Conv1d(rd, rd, 15,padding=7)
        self.mid1 = nn.Conv1d(rd, rd, 15,padding=7)
        for j in range(5):
            self.dcnn.append(nn.ConvTranspose1d(rd, rd, 3, padding=1, stride=2, output_padding=1))
            self.denseconvs.append(nn.Conv1d(rd, rd, 1))
            for i in range(3):
                self.dcnn.append(nn.ConvTranspose1d(rd*2, rd, 15, padding=7))
                self.denseconvs.append(nn.Conv1d(rd, rd, 1))
        self.post1 = nn.Conv1d(rd, rd, 1)
        # normal cnn
        self.post2 = nn.Conv1d(rd, rd, 1)

    def forward(self, x):
        x = self.causal0(x)
        cntnn,cntcut=0,0
        shortcut=[]
        for j in range(5):
            for i in range(3):
                xinput = x.clone()
                x = F.relu(x)
                shortcut.append(self.skipconvs[cntcut](x))
                cntcut+=1
                x = F.relu(self.dcnn[cntnn](x))
                x = self.denseconvs[cntnn](x)
                cntnn+=1
                x += xinput
            x = F.relu(x)
            x = self.dcnn[cntnn](x)
            x = F.relu(x)
            x = self.denseconvs[cntnn](x)
            cntnn+=1

        x = self.mid1(F.relu(self.mid0(F.relu(x))))

        for j in range(5):
            x = F.relu(x)
            x = self.dcnn[cntnn](x)
            x = F.relu(x)
            x = self.denseconvs[cntnn](x)
            cntnn+=1
            for i in range(3):
                xinput = x.clone()
                x = F.relu(x)

                cntcut -= 1
                x = torch.cat((x,shortcut[cntcut]),1)

                x = F.relu(self.dcnn[cntnn](x))
                x = self.denseconvs[cntnn](x)
                cntnn+=1
                x += xinput

        x = self.post2(F.relu(self.post1(F.relu(x))))
        return x