import torch

class MyLinear(torch.nn.Module):

    def __init__(self, in_put, out_put):
        super(MyLinear, self).__init__()

        self.w = torch.nn.Parameter(torch.randn(out_put, in_put))
        self.b = torch.nn.Parameter(torch.randn(out_put))

    def forward(self, x):
        x = x @ self.w.t() + self.b
        return x


class MyFlatten(torch.nn.Module):

    def __init__(self):
        super(MyFlatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class TestNet(torch.nn.Module):

    def __init__(self):
        super(TestNet, self).__init__()

        self.net = torch.nn.Sequential(
            # [-1,1,28,28]
            torch.nn.Conv2d(1, 16, 3, 1, 1),
            # [-1,16,28,28]
            torch.nn.MaxPool2d(2, 2),
            # [-1,16,14,14]
            torch.nn.BatchNorm2d(16),
            # [-1,16,14,14]
            MyFlatten(),
            # [-1,16*14*14]
            MyLinear(16 * 14 * 14, 10)
        )

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    x = torch.randn(3,1,28,28)
    net = TestNet()
    x = net(x)
    print(x.shape)