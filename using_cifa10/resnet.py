import torch


class MyFlatten(torch.nn.Module):

    def __init__(self):
        super(MyFlatten, self).__init__()

    def forward(self,x):
        return x.view(x.size(0),-1)


class ResBlk(torch.nn.Module):

    def __init__(self,in_put,out_put,stride=1):
        super(ResBlk, self).__init__()

        self.model = torch.nn.Sequential(
            # [-1,in_put,32,32] => [-1,out_put,32,32]
            torch.nn.Conv2d(in_put,out_put,3,stride,1),
            torch.nn.BatchNorm2d(out_put),
            torch.nn.LeakyReLU(inplace=True),

            # [-1,out_put,32,32] => [-1,out_put,32,32]
            torch.nn.Conv2d(out_put, out_put, 3, 1, 1),
            torch.nn.BatchNorm2d(out_put)
        )

        self.extra = torch.nn.Sequential()
        if in_put != out_put:
            self.extra = torch.nn.Sequential(
                torch.nn.Conv2d(in_put,out_put,1,stride,0),
                torch.nn.BatchNorm2d(out_put)
            )

    def forward(self,x):
        out = self.model(x)
        x = torch.nn.functional.leaky_relu(out + self.extra(x))
        return x


class ResNet18(torch.nn.Module):

    def __init__(self):
        super(ResNet18, self).__init__()

        self.init_model = torch.nn.Sequential(
            # [-1,3,32,32] => [-1,64,11,11]
            torch.nn.Conv2d(3, 64, 3, 3, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(inplace=True),
        )

        self.model = torch.nn.Sequential(
            # [-1,64,11,11] => [-1, 128, 6, 6]
            ResBlk(64, 128, stride=2),
            # [-1, 128, 6, 6] => [-1,256,3,3]
            ResBlk(128, 256, stride=2),
            # [-1, 256, 3, 3] => [-1,512,2,2]
            ResBlk(256, 512, stride=2),
            # [-1, 512, 2, 2] => [-1,512,2,2]
            ResBlk(512, 512, stride=2),
            # [-1,512,2,2] => [-1,512,1,1]
            torch.nn.AdaptiveAvgPool2d([1,1]),
            # [-1,512,1,1] => [-1,512*1*1]
            MyFlatten(),
            # [-1,512*1*1] => [-1,10]
            torch.nn.Linear(512*1*1,10),
        )

    def forward(self, x):
        out = self.init_model(x)
        out = self.model(out)
        return out


if __name__ == '__main__':
    net = ResNet18()
    x = torch.randn(3,3,32,32)
    res = net(x)
    print(res.shape) # torch.Size([3, 10])
