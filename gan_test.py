import torch
import numpy as np
import visdom
from matplotlib import pyplot as plt
import random


class Generator(torch.nn.Module):

    def __init__(self,h_dim = 400):
        super(Generator, self).__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, h_dim),
            torch.nn.ReLU(True),
            torch.nn.Linear(h_dim, h_dim),
            torch.nn.ReLU(True),
            torch.nn.Linear(h_dim, h_dim),
            torch.nn.ReLU(True),
            torch.nn.Linear(h_dim, 2),
        )

    def forward(self,z):
        out = self.net(z)
        return out


class Discriminator(torch.nn.Module):

    def __init__(self,h_dim = 400):
        super(Discriminator, self).__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, h_dim),
            torch.nn.ReLU(True),
            torch.nn.Linear(h_dim, h_dim),
            torch.nn.ReLU(True),
            torch.nn.Linear(h_dim, h_dim),
            torch.nn.ReLU(True),
            torch.nn.Linear(h_dim, 1),
            torch.nn.Sigmoid()
        )

    def forward(self,x):
        out = self.net(x)
        return out.view(-1)


class TrainAndTest(object):

    def __init__(self,batch_size = 200,learning_rate = 1e-3,epochs = 1000,net=None):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.net = net
        self.viz = visdom.Visdom()

    def gen_data(self):
        scale = 2.
        centers = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1. / np.sqrt(2), 1. / np.sqrt(2)),
            (1. / np.sqrt(2), -1. / np.sqrt(2)),
            (-1. / np.sqrt(2), 1. / np.sqrt(2)),
            (-1. / np.sqrt(2), -1. / np.sqrt(2))
        ]
        centers = [(scale * x, scale * y) for x, y in centers]

        while True:
            dataset = []
            for i in range(self.batch_size):
                point = np.random.randn(2) * .02
                center = random.choice(centers)
                point[0] += center[0]
                point[1] += center[1]
                dataset.append(point)

            dataset = np.array(dataset).astype(np.float64)
            dataset /= 1.414

            yield dataset

    def generate_image(self,D, G, xr, epoch):
        N_POINTS = 128
        RANGE = 3
        plt.clf()

        points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
        points[:, :, 0] = np.linspace(-RANGE, RANGE, N_POINTS)[:, None]
        points[:, :, 1] = np.linspace(-RANGE, RANGE, N_POINTS)[None, :]
        points = points.reshape((-1, 2))

        with torch.no_grad():
            disc_map = D(points).cpu().numpy()  # [16384]
        x = y = np.linspace(-RANGE, RANGE, N_POINTS)
        cs = plt.contour(x, y, disc_map.reshape((len(x), len(y))).transpose())
        plt.clabel(cs, inline=1, fontsize=10)

        with torch.no_grad():
            z = torch.randn(self.batch_size, 2) # [b, 2]
            samples = G(z).cpu().numpy()  # [b, 2]
        plt.scatter(xr[:, 0], xr[:, 1], c='orange', marker='.')
        plt.scatter(samples[:, 0], samples[:, 1], c='green', marker='+')

        self.viz.matplot(plt, win='contour', opts=dict(title='p(x):%d' % epoch))

        return None

    def weights_init(sef,m):
        if isinstance(m, torch.nn.Linear):
            # m.weight.data.normal_(0.0, 0.02)
            torch.nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0)

    def gradient_penalty(self,D, xr, xf):
        LAMBDA = 0.3

        # only constrait for Discriminator
        xf = xf.detach()
        xr = xr.detach()

        alpha = torch.rand(self.batch_size, 1)
        alpha = alpha.expand_as(xr)

        interpolates = alpha * xr + ((1 - alpha) * xf)
        interpolates.requires_grad_()

        disc_interpolates = D(interpolates)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones_like(disc_interpolates),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gp = ((gradients.norm(2,dim=1) - 1) ** 2).mean() * LAMBDA

        return gp

    def optimize_by_grad(self):
        torch.manual_seed(23)
        np.random.seed(23)

        G = Generator()
        D = Discriminator()
        G.apply(self.weights_init)
        D.apply(self.weights_init)

        optim_G = torch.optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.9))
        optim_D = torch.optim.Adam(D.parameters(), lr=1e-3, betas=(0.5, 0.9))

        data_iter = self.gen_data()

        self.viz.line([[0, 0]], [0], win='loss', opts=dict(title='loss',legend=['D', 'G']))

        for epoch in range(self.epochs):
            for _ in range(5):
                x = next(data_iter)
                xr = torch.from_numpy(x)

                # [b]
                predr = D(xr)
                # max log(lossr)
                lossr = - predr.mean()

                # [b, 2]
                z = torch.randn(self.batch_size, 2)
                # stop gradient on G
                # [b, 2]
                xf = G(z).detach()
                # [b]
                predf = D(xf)
                # min predf
                lossf = predf.mean()

                # gradient penalty
                gp = self.gradient_penalty(D, xr, xf)

                loss_D = lossr + lossf + gp
                optim_D.zero_grad()
                loss_D.backward()
                optim_D.step()

            # 2. train Generator
            z = torch.randn(self.batch_size, 2)
            xf = G(z)
            predf = (D(xf))
            # max predf
            loss_G = - (predf.mean())
            optim_G.zero_grad()
            loss_G.backward()
            optim_G.step()

            if epoch % 100 == 0:
                self.viz.line([[loss_D.item(), loss_G.item()]], [epoch], win='loss', update='append')

                self.generate_image(D, G, xr, epoch)

                print(loss_D.item(), loss_G.item())

        return None


if __name__ == '__main__':
    tt = TrainAndTest()
    tt.optimize_by_grad()