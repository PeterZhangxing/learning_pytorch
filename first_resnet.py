import torch
from torchvision import datasets
from torchvision import transforms

class ResBlk(torch.nn.Module):

    def __init__(self,ch_in,ch_out):
        super(ResBlk, self).__init__()

        self.conv1 = torch.nn.Conv2d(ch_in,ch_out,3,1,1)
        self.bn1 = torch.nn.BatchNorm2d(ch_out)
        self.relu1 = torch.nn.ReLU(inplace=True)

        self.conv2 = torch.nn.Conv2d(ch_out,ch_out,3,1,1)
        self.bn2 = torch.nn.BatchNorm2d(ch_out)

        self.extra = torch.nn.Sequential()
        if ch_in != ch_out:
            self.extra = torch.nn.Sequential(
                torch.nn.Conv2d(ch_in,ch_out,1,1,0),
                torch.nn.BatchNorm2d(ch_out)
            )

    def forward(self,x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.extra(x) + out

        return out


class ResNet18(torch.nn.Module):

    def __init__(self):
        super(ResNet18, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3,16,3,1,1),
            torch.nn.BatchNorm2d(16)
        )
        self.relu = torch.nn.ReLU(inplace=True)

        self.blk1 = ResBlk(16, 16)
        self.blk2 = ResBlk(16, 32)
        # self.blk3 = ResBlk(128, 256)
        # self.blk4 = ResBlk(256, 512)

        self.outlayer = torch.nn.Linear(32*32*32,10)

    def forward(self,x):
        x = self.relu(self.conv1(x))
        x = self.blk1(x)
        x = self.blk2(x)

        x = x.view(x.size(0),-1)
        x = self.outlayer(x)

        return x