import torch


def rnn_test():
    rnn = torch.nn.RNN(input_size=100,hidden_size=20,num_layers=3)
    print(rnn)

    x = torch.randn(10,3,100)

    out,h = rnn(input=x,hx=torch.zeros(3,3,20))
    print(out.shape,h.shape) # [10,3,20],[3,3,20]

def rnn_cell():
    x = torch.randn(10, 3, 100)

    cell = torch.nn.RNNCell(input_size=100,hidden_size=20)
    h1 = torch.zeros(3,20)

    for xt in x:
        h1 = cell(xt,h1)
    print(h1.shape)

def rnn_mul_cell():
    x = torch.randn(10, 3, 100)

    cell1 = torch.nn.RNNCell(input_size=100,hidden_size=30)
    cell2 = torch.nn.RNNCell(30,20)
    h1 = torch.zeros(3,30)
    h2 = torch.zeros(3,20)

    for xt in x:
        h1 = cell1(xt,h1)
        h2 = cell2(h1,h2)

    print(h1.shape,h2.size())

def lstm_test():
    x = torch.randn(10, 3, 100)

    lstm = torch.nn.LSTM(input_size=100,hidden_size=20,num_layers=4)
    h0 = torch.zeros(4,3,20)
    c0 = torch.zeros(4,3,20)
    out,(h,c) = lstm(x,[h0,c0])

    print(out.shape,h.shape,c.shape)
    # torch.Size([10, 3, 20]) torch.Size([4, 3, 20]) torch.Size([4, 3, 20])

def lstm_cell_test():
    x = torch.randn(10, 3, 100)

    lstmcell1 = torch.nn.LSTMCell(input_size=100, hidden_size=20)
    h1, c1 = torch.zeros(3, 20), torch.zeros(3, 20)

    for xt in x:
        h1,c1 = lstmcell1(xt,[h1,c1])

    print(h1.shape,c1.shape) # [1,3,20] [1,3,20]

def lstm_mulcell_test():
    x = torch.randn(10, 3, 100)

    lstmcell1 = torch.nn.LSTMCell(input_size=100,hidden_size=30)
    lstmcell2 = torch.nn.LSTMCell(input_size=30,hidden_size=30)
    lstmcell3 = torch.nn.LSTMCell(input_size=30,hidden_size=50)
    lstmcell4 = torch.nn.LSTMCell(input_size=50,hidden_size=20)

    h1,c1 = torch.zeros(3, 30),torch.zeros(3, 30)
    h2,c2 = torch.zeros(3, 30),torch.zeros(3, 30)
    h3,c3 = torch.zeros(3, 50),torch.zeros(3, 50)
    h4,c4 = torch.zeros(3, 20),torch.zeros(3, 20)

    for xt in x:
        h1,c1 = lstmcell1(xt,[h1,c1])
        h2,c2 = lstmcell2(h1,[h2,c2])
        h3,c3 = lstmcell3(h2,[h3,c3])
        h4,c4 = lstmcell4(h3,[h4,c4])

    print(h1.shape,h2.shape,h3.shape,h4.shape)
    # torch.Size([3, 30]) torch.Size([3, 30]) torch.Size([3, 50]) torch.Size([3, 20])


if __name__ == '__main__':
    # rnn_test()
    # rnn_cell()
    # rnn_mul_cell()
    # lstm_test()
    # lstm_cell_test()
    lstm_mulcell_test()