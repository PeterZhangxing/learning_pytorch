import torch
from utils import Flatten


class ResBlk(torch.nn.Module):

    def __init__(self,ch_in,ch_out,stride=1):
        super(ResBlk, self).__init__()

        self.conv1 = torch.nn.Conv2d(ch_in,ch_out,3,stride,1)
        self.bn1 = torch.nn.BatchNorm2d(ch_out)
        self.conv2 = torch.nn.Conv2d(ch_out,ch_out,3,1,1)
        self.bn2 = torch.nn.BatchNorm2d(ch_out)

        self.extra = torch.nn.Sequential()
        if ch_in != ch_out:
            self.extra = torch.nn.Sequential(
                torch.nn.Conv2d(ch_in,ch_out,1,stride),
                torch.nn.BatchNorm2d(ch_out)
            )

    def forward(self,x):
        out = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.extra(x) + out
        out = torch.nn.functional.relu(out)

        return out


class ResNet18(torch.nn.Module):

    def __init__(self,num_class):
        super(ResNet18, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=3, padding=0),
            torch.nn.BatchNorm2d(16)
        )

        self.blks = torch.nn.Sequential(
            # [b, 16, h, w] => [b, 32, h ,w]
            ResBlk(16, 32, stride=3),
            # [b, 32, h, w] => [b, 64, h, w]
            ResBlk(32, 64, stride=3),
            # [b, 64, h, w] => [b, 128, h, w]
            ResBlk(64, 128, stride=2),
            # [b, 128, h, w] => [b, 256, h, w]
            ResBlk(128, 256, stride=2),
            # [b,2304]
            Flatten(),
        )

        self.linear = torch.nn.Linear(2304, num_class)

    def forward(self,x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.blks(x)
        x = self.linear(x)

        return x


if __name__ == '__main__':
    # x = torch.randn(3,64,224,224)
    # rb = ResBlk(64,128)
    # out = rb(x)
    # print(out.shape) # torch.Size([3, 128, 224, 224])

    x = torch.randn(3, 3, 224, 224)
    resn = ResNet18(5)
    out = resn(x)
    print(out.shape) # torch.Size([3, 5])