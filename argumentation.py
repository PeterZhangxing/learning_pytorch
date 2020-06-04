import torch
from torchvision import datasets,transforms
from visdom import Visdom


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
        self.viz = Visdom()
        self.init_data()

    def init_data(self):
        train_db = datasets.MNIST(
            './data', train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.RandomRotation([90, 180,]),
                transforms.Resize([32, 32]),
                transforms.RandomCrop([28, 28]),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))
        self.train_loader = torch.utils.data.DataLoader(train_db,batch_size=self.batch_size, shuffle=True)

        test_db = datasets.MNIST(
            './data', train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))
        self.test_loader = torch.utils.data.DataLoader(test_db,batch_size=self.batch_size, shuffle=True)

        return None

    def init_draw_viz(self):
        self.viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))
        self.viz.line([[0.0, 0.0]], [0.], win='test', opts=dict(title='test loss&acc.',legend=['loss', 'acc.']))

        return None

    def optimize_by_grad(self):
        self.init_draw_viz()
        global_step = 0
        for epoch in range(self.epochs):
            for batch_idx,(data,target) in enumerate(self.train_loader):
                data = data.view(-1, 28 * 28)

                logits = self.net(data)
                loss = self.criteon(logits, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                global_step += 1
                self.viz.line([loss.item()], [global_step], win='train_loss', update='append')

                if batch_idx % 100 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(self.train_loader.dataset),
                               100. * batch_idx / len(self.train_loader), loss.item()))

        return None

    def pred_op(self):
        test_loss = 0
        correct = 0

        step = 0
        for data, target in self.test_loader:
            data = data.view(-1, 28 * 28)
            logits = self.net(data)

            cur_loss = self.criteon(logits,target).item()
            test_loss += cur_loss
            # print(logits.data,"*"*50,target.data)

            pred = logits.data.max(dim=1)[1]
            # print("*"*50,pred)
            cur_correct = pred.eq(target.data).sum()
            correct += cur_correct

            cur_acc = 100. * cur_correct / len(target.data)
            cur_avg_loss = cur_loss / len(target.data)

            step += 1
            self.viz.line(
                [[cur_acc, cur_avg_loss]], [step],
                win='test',
                update='append'
            )

        test_loss /= len(self.test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))

        return None


if __name__ == '__main__':
    net = MLP()
    obj = MyFullConnectTest(net=net,epochs=100)
    obj.optimize_by_grad()
    obj.pred_op()