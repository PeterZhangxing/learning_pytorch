import torch


class MyFlatten(torch.nn.Module):

    def __init__(self):
        super(MyFlatten, self).__init__()

    def forward(self,x):
        return x.view(x.size(0),-1)


class LeNet5(torch.nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()

        self.model = torch.nn.Sequential(
            # [-1,3,32,32] => [-1,16,32,32]
            torch.nn.Conv2d(3,16,5,1,2),
            # [-1,16,32,32] => [-1,16,16,16]
            torch.nn.AvgPool2d(2,2),
            # [-1,16,16,16] => [-1,32,16,16]
            torch.nn.Conv2d(16,32,5,1,2),
            # [-1,32,16,16] => [-1,32,8,8]
            torch.nn.AvgPool2d(2,2),

            # [-1,32,8,8] => [-1,32*8*8]
            MyFlatten(),
            # [-1,32*8*8] => [-1,32]
            torch.nn.Linear(32*8*8,32),
            torch.nn.LeakyReLU(),
            # [-1,32] => [-1,10]
            torch.nn.Linear(32,10)
        )

    def forward(self,x):
        x = self.model(x)
        return x # haven't been through softmax


if __name__ == '__main__':
    net = LeNet5()
    x = torch.randn(2,3,32,32)
    out = net(x)
    print(out.shape) # torch.Size([2, 10])