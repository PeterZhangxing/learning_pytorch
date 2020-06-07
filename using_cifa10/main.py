import torch
from torchvision import datasets,transforms

from lenet5 import LeNet5
from resnet import ResNet18


class MyModuleTest(object):

    def __init__(self,batch_size = 100,learning_rate = 1e-3,epochs = 10,net=None):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.net = net
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        self.criteon = torch.nn.CrossEntropyLoss()
        self.init_data()

    def init_data(self):
        cifar_train_db = datasets.CIFAR10(
            'data/cifar', True,
            transform=transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
        ]), download=True)
        self.train_loader = torch.utils.data.DataLoader(
            cifar_train_db,
            batch_size=self.batch_size,
            shuffle=True)

        cifar_test_db = datasets.CIFAR10(
            'data/cifar', False,
            transform=transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
        ]), download=True)
        self.test_loader = torch.utils.data.DataLoader(
            cifar_test_db,
            batch_size=self.batch_size,
            shuffle=True)

        return None

    def optimize_by_grad(self):
        print('start practicing!')
        for epoch in range(self.epochs):

            self.net.train()
            train_loss = 0
            train_correct = 0
            for batch_idx,(data,target) in enumerate(self.train_loader):

                logits = self.net(data)
                loss = self.criteon(logits, target)
                train_loss += loss.item()

                train_pred = logits.data.argmax(1)
                train_correct += train_pred.eq(target.data).sum()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if batch_idx % 100 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(self.train_loader.dataset),
                               100. * batch_idx / len(self.train_loader), loss.item()))

            train_loss /= len(self.train_loader.dataset)
            print('\nTrain set:{} Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                epoch,train_loss, train_correct, len(self.train_loader.dataset),
                100. * train_correct / len(self.train_loader.dataset)))

            self.net.eval()
            with torch.no_grad():
                test_loss = 0
                correct = 0

                for data, target in self.test_loader:
                    logits = self.net(data)
                    test_loss += self.criteon(logits, target).item()

                    pred = logits.data.max(dim=1)[1]
                    correct += pred.eq(target.data).sum()

                test_loss /= len(self.test_loader.dataset)
                print('\nTest set:{} Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                    epoch,test_loss, correct, len(self.test_loader.dataset),
                    100. * correct / len(self.test_loader.dataset)))

        return None

    # def pred_op(self):
    #     test_loss = 0
    #     correct = 0
    #
    #     for data, target in self.test_loader:
    #         data = data.view(-1, 28 * 28)
    #         logits = self.net(data)
    #         test_loss += self.criteon(logits,target).item()
    #         # print(logits.data,"*"*50,target.data)
    #
    #         pred = logits.data.max(dim=1)[1]
    #         # print("*"*50,pred)
    #         correct += pred.eq(target.data).sum()
    #
    #     test_loss /= len(self.test_loader.dataset)
    #     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #         test_loss, correct, len(self.test_loader.dataset),
    #         100. * correct / len(self.test_loader.dataset)))
    #
    #     return None


if __name__ == '__main__':
    # net = LeNet5()
    net = ResNet18()
    obj = MyModuleTest(epochs=5,net=net)
    obj.optimize_by_grad()