import torch
from torchvision import datasets,transforms

class MyFullConnectTest(object):

    def __init__(self,batch_size = 200,learning_rate = 0.01,epochs = 10):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.optimizer = None
        self.criteon = None
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

    def model(self):
        self.w1 = torch.randn(200, 784, requires_grad=True)
        self.b1 = torch.zeros(200, requires_grad=True)

        self.w2 = torch.randn(200, 200, requires_grad=True)
        self.b2 = torch.zeros(200, requires_grad=True)

        self.w3 = torch.randn(10, 200, requires_grad=True)
        self.b3 = torch.zeros(10, requires_grad=True)

        torch.nn.init.kaiming_normal_(self.w1)
        torch.nn.init.kaiming_normal_(self.w2)
        torch.nn.init.kaiming_normal_(self.w3)

        return None

    def forward(self,x):

        x = x @ self.w1.t() + self.b1
        x = torch.nn.functional.relu(x)
        x = x @ self.w2.t() + self.b2
        x = torch.nn.functional.relu(x)
        x = x @ self.w3.t() + self.b3
        x = torch.nn.functional.relu(x)

        return x

    def optimize_by_grad(self):
        self.model()
        self.optimizer = torch.optim.SGD([self.w1, self.b1, self.w2, self.b2, self.w3, self.b3], lr=self.learning_rate)
        self.criteon = torch.nn.CrossEntropyLoss()

        for epoch in range(self.epochs):
            for batch_idx,(data,target) in enumerate(self.train_loader):
                data = data.view(-1, 28 * 28)

                logits = self.forward(data)
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
            logits = self.forward(data)
            test_loss += self.criteon(logits,target).item()

            pred = logits.data.max(dim=1)[1]
            # print("*"*50,pred.shape)
            correct += pred.eq(target.data).sum()

        test_loss /= len(self.test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))

        return None


if __name__ == '__main__':
    obj = MyFullConnectTest()
    obj.optimize_by_grad()
    obj.pred_op()