import torch
from torchvision import datasets,transforms


class MyLinear(torch.nn.Module):

    def __init__(self,in_put,out_put):
        super(MyLinear, self).__init__()

        self.w = torch.nn.Parameter(torch.randn(out_put,in_put))
        self.b = torch.nn.Parameter(torch.randn(out_put))

    def forward(self,x):
        x = x @ self.w.t() + self.b
        return x


class MyFlatten(torch.nn.Module):

    def __init__(self):
        super(MyFlatten, self).__init__()

    def forward(self,x):
        return x.view(x.size(0),-1)


class TestNet(torch.nn.Module):

    def __init__(self):
        super(TestNet, self).__init__()

        self.net = torch.nn.Sequential(
            # [-1,1,28,28]
            torch.nn.Conv2d(1,16,3,1,1),
            # [-1,16,28,28]
            torch.nn.MaxPool2d(2,2),
            # [-1,16,14,14]
            torch.nn.BatchNorm2d(16),
            # [-1,16,14,14]
            MyFlatten(),
            # [-1,16*14*14]
            MyLinear(16*14*14,10)
        )

    def forward(self,x):
        return self.net(x)


class MLP(torch.nn.Module):

    def __init__(self):
        super(MLP, self).__init__()

        # self.model = torch.nn.Sequential(
        #     torch.nn.Linear(784,200),
        #     torch.nn.ReLU(inplace=True),
        #
        #     torch.nn.Linear(200, 200),
        #     torch.nn.ReLU(inplace=True),
        #
        #     torch.nn.Linear(200, 10),
        #     torch.nn.ReLU(inplace=True),
        # )

        self.model = torch.nn.Sequential(
            torch.nn.Linear(784,200),
            torch.nn.LeakyReLU(inplace=True),

            torch.nn.Linear(200, 200),
            torch.nn.LeakyReLU(inplace=True),

            torch.nn.Linear(200, 10),
            torch.nn.LeakyReLU(inplace=True),
        )

    def forward(self,x):
        x = self.model(x)
        return x


class MyFullConnectTest(object):

    def __init__(self,batch_size = 200,learning_rate = 0.01,epochs = 2,net=None):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.net = net
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=learning_rate)
        self.criteon = torch.nn.CrossEntropyLoss()
        self.init_data()

    def init_data(self):
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),batch_size=self.batch_size, shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),batch_size=self.batch_size, shuffle=True)

        return None

    def optimize_by_grad(self):
        for epoch in range(self.epochs):
            for batch_idx,(data,target) in enumerate(self.train_loader):
                data = data.view(-1, 28 * 28)

                logits = self.net(data)
                loss = self.criteon(logits, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if batch_idx % 100 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(self.train_loader.dataset),
                               100. * batch_idx / len(self.train_loader), loss.item()))

        return None

    def pred_op(self):
        test_loss = 0
        correct = 0

        for data, target in self.test_loader:
            data = data.view(-1, 28 * 28)
            logits = self.net(data)
            test_loss += self.criteon(logits,target).item()
            # print(logits.data,"*"*50,target.data)

            pred = logits.data.max(dim=1)[1]
            # print("*"*50,pred)
            correct += pred.eq(target.data).sum()

        test_loss /= len(self.test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))

        return None


if __name__ == '__main__':
    net = MLP()
    obj = MyFullConnectTest(net=net)
    obj.optimize_by_grad()
    obj.pred_op()