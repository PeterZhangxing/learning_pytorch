import numpy as np
from matplotlib import pyplot as plt
import torch


class Net(torch.nn.Module):

    def __init__(self,input_size=1,hidden_size=16,output_size=1):
        super(Net, self).__init__()

        self.hidden_size = hidden_size

        self.rnn = torch.nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        for p in self.rnn.parameters():
            torch.nn.init.normal_(p, mean=0.0, std=0.001)

        self.linear = torch.nn.Linear(hidden_size,output_size)

    def forward(self,x,hidden_prev):
        # out:[b,seq,hidden_size]
        out,hidden_prev = self.rnn(x,hidden_prev)
        # print(out.shape,hidden_prev.shape)
        # out:[b*seq,hidden_size]
        out = out.view(-1,out.size(2))
        # print(out.shape)
        # out:[b*seq,output_size]
        out = self.linear(out)
        # print(out.shape)
        # out:[1,b*seq,output_size]
        out = out.unsqueeze(dim=0)
        # print(out.shape)
        '''
        torch.Size([1, 10, 16]) torch.Size([1, 1, 16])
        torch.Size([10, 16])
        torch.Size([10, 1])
        torch.Size([1, 10, 1])
        '''

        return out,hidden_prev


class MySeris(object):

    def __init__(self,batch_size = 200,learning_rate = 0.01,epochs = 10,num_time_steps=50,net=None):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.net = net
        self.num_time_steps = num_time_steps
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        self.criteon = torch.nn.MSELoss()

    def init_data(self):
        start = np.random.randint(3,size=1)[0] # numpy.int64
        time_steps = np.linspace(start, start + 10, self.num_time_steps) # array with 50 numbers

        data = np.sin(time_steps)
        data = data.reshape(self.num_time_steps,1)

        x = torch.tensor(data[:-1]).float().view(1, self.num_time_steps - 1, 1)
        y = torch.tensor(data[1:]).float().view(1, self.num_time_steps - 1, 1)

        return x,y,time_steps

    def optimize_by_grad(self):
        hidden_prev = torch.zeros(1, 1, net.hidden_size)
        output = None
        for iter in range(1000):
            x,y,time_steps = self.init_data()
            output, hidden_prev = self.net(x, hidden_prev)
            hidden_prev = hidden_prev.detach()

            loss = self.criteon(output, y)
            self.net.zero_grad()
            loss.backward()
            # for p in model.parameters():
            #     print(p.grad.norm())
            # torch.nn.utils.clip_grad_norm_(p, 10)
            self.optimizer.step()

            if iter % 100 == 0:
                print("Iteration: {} loss {}".format(iter, loss.item()))

        return output, hidden_prev

    def pred_op(self):
        output, hidden_prev = self.optimize_by_grad()
        predictions = []
        x,y,time_steps = self.init_data()
        # print(x) # tensor([[[ 0.9093],[ 0.8061],...,[-0.6965]]])
        input = x[:, 0, :]
        # print(input) # tensor([[0.9093]])

        for _ in range(x.shape[1]):
            input = input.view(1, 1, 1)
            pred, hidden_prev = self.net(input, hidden_prev)
            print(input, pred)
            input = pred
            predictions.append(pred.detach().numpy().ravel()[0])

        return x,y,predictions,time_steps

    def build_plt(self):
        x, y, predictions,time_steps = self.pred_op()

        x = x.data.numpy().ravel()

        plt.scatter(time_steps[:-1], x.ravel(), s=90)
        plt.plot(time_steps[:-1], x.ravel())

        plt.scatter(time_steps[1:], predictions)
        plt.show()

        return None


if __name__ == '__main__':
    net = Net()
    # x = torch.randn(1,10,1)
    # h = torch.zeros(1,1,16)
    # net(x,h)

    obj = MySeris(net=net)
    # obj.optimize_by_grad()
    obj.pred_op()
    # obj.build_plt()