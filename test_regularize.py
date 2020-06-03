import torch
from torchvision import datasets,transforms


class MLP(torch.nn.Module):

    def __init__(self):
        super(MLP, self).__init__()

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
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=learning_rate,weight_decay=0.01)
        self.criteon = torch.nn.CrossEntropyLoss()
        self.init_data()

    def l1_regularization(self,logits,target,weight_decay):
        reg_loss = 0
        for param in self.net.parameters():
            reg_loss += torch.sum(torch.abs(param))
        classify_loss = self.criteon(logits,target)
        loss = classify_loss + weight_decay * reg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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