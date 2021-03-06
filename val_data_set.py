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

    def __init__(self,batch_size = 200,learning_rate = 0.01,epochs = 10,net=None):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.net = net
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=learning_rate)
        self.criteon = torch.nn.CrossEntropyLoss()
        self.init_data()

    def init_data(self):
        train_db = datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

        test_db = datasets.MNIST('./data', train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))

        train_db, val_db = torch.utils.data.random_split(train_db, [50000, 10000])

        self.train_loader = torch.utils.data.DataLoader(
            train_db,
            batch_size=self.batch_size, shuffle=True)

        self.val_loader = torch.utils.data.DataLoader(
            val_db,
            batch_size=self.batch_size, shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(
            test_db,
            batch_size=self.batch_size, shuffle=True)

        return None

    def optimize_by_grad(self):
        for epoch in range(self.epochs):
            train_loss = 0
            train_correct = 0
            for batch_idx,(data,target) in enumerate(self.train_loader):
                data = data.view(-1, 28 * 28)

                logits = self.net(data)
                loss = self.criteon(logits, target)
                train_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_pred = logits.data.argmax(dim=1)
                train_correct += train_pred.eq(target.data).sum()

                if batch_idx % 100 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(self.train_loader.dataset),
                               100. * batch_idx / len(self.train_loader), loss.item()))

            train_loss /= len(self.train_loader.dataset)
            train_correct = 100. * train_correct / len(self.train_loader.dataset)

            val_loss = 0
            correct = 0
            for data, target in self.val_loader:
                data = data.view(-1, 28 * 28)
                logits = self.net(data)
                val_loss += self.criteon(logits, target).item()

                pred = logits.data.max(dim=1)[1]
                # print("*"*50,pred.shape)
                correct += pred.eq(target.data).sum()

            val_loss /= len(self.val_loader.dataset)
            val_correct = 100. * correct / len(self.val_loader.dataset)
            print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                val_loss, correct, len(self.val_loader.dataset),
                val_correct))

            print("*"*50)
            print("epoch %d,train loss:%f,val loss:%f,train correct:%f,val correct:%s"%(
                epoch,train_loss,val_loss,train_correct,val_correct.item()))
            print("*" * 50)

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