import torch
from torchvision import datasets,transforms
import visdom


class AutoCoder(torch.nn.Module):

    def __init__(self,in_put,coder):
        super(AutoCoder, self).__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_put,256),
            torch.nn.ReLU(),
            torch.nn.Linear(256,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,coder),
            torch.nn.ReLU()
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(coder,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, in_put),
            torch.nn.Sigmoid(),
        )

    def forward(self,x):
        x = self.encoder(x.view(x.size(0),-1))
        # print(x.shape) # torch.Size([100, 20])
        x = self.decoder(x)
        x = x.view(x.size(0),1,28,28)
        # print(x.shape) # torch.Size([100, 1, 28, 28])
        return x,None


class VarAutoCoder(torch.nn.Module):

    def __init__(self,in_put,coder):
        super(VarAutoCoder, self).__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_put,256),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(256,64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64,coder),
            torch.nn.LeakyReLU()
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(int(0.5*coder),64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 256),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(256, in_put),
            torch.nn.Sigmoid(),
        )

    def forward(self,x):
        h_ = self.encoder(x.view(x.size(0), -1))
        # [b, 20] => [b, 10] and [b, 10]
        mu, sigma = h_.chunk(2, dim=1)
        # reparametrize trick, epison~N(0, 1)
        h = mu + sigma * torch.randn_like(sigma)

        # decoder
        x_hat = self.decoder(h)
        # reshape
        x_hat = x_hat.view(x.size(0), 1, 28, 28)

        kld = 0.5 * torch.sum(
            torch.pow(mu, 2) +
            torch.pow(sigma, 2) -
            torch.log(1e-8 + torch.pow(sigma, 2)) - 1
        ) / (x.size(0)*28*28)

        return x_hat,kld


class TrainAndTest(object):

    def __init__(self,batch_size = 32,learning_rate = 1e-3,epochs = 1000,net=None):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.net = net
        self.viz = visdom.Visdom()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        self.criteon = torch.nn.MSELoss()
        self.init_data()

    def init_data(self):
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                           ])),batch_size=self.batch_size, shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
            ])),batch_size=self.batch_size, shuffle=True)

        return None

    def optimize_by_grad(self):
        for epoch in range(self.epochs):
            train_loss = 0
            train_kld = 0
            for batch_idx,(data,_) in enumerate(self.train_loader):
                x_hat, kld = self.net(data)
                loss = self.criteon(x_hat, data)
                train_loss += loss.item()

                if kld:
                    train_kld += kld.item()
                    loss = loss + 1.0 * kld

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if train_kld:
                print(epoch, 'avg_loss:', train_loss/len(self.train_loader.dataset), 'avg_train_kld:', train_kld/len(self.train_loader.dataset))
            else:
                print(epoch, 'avg_loss:', train_loss/len(self.train_loader.dataset))

            self.pred_op()

        return None

    def pred_op(self):
        test_loss = 0
        with torch.no_grad():
            for data, _ in self.test_loader:
                x_hat,kld = self.net(data)
                test_loss += self.criteon(data,x_hat).item()

                self.viz.images(data, nrow=8, win='x', opts=dict(title='x'))
                self.viz.images(x_hat, nrow=8, win='x_hat', opts=dict(title='x_hat'))

            test_loss /= len(self.test_loader.dataset)
            print('test loss:',test_loss)

        return None

if __name__ == '__main__':
    # x = torch.randn(100,1,28,28)
    # net = AutoCoder(28*28,20)
    net = VarAutoCoder(28*28,20)

    tat = TrainAndTest(net=net)
    tat.optimize_by_grad()