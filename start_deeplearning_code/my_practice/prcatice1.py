import numpy as np
from matplotlib import pyplot as plt

def my_and(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.7
    res = np.sum(x*w) + b
    if res > 0:
        return 1
    return 0

def step_function(x):
    y = x > 0
    return y.astype(np.int32)

# x = np.array([1.,-3.,4.])
# res = step_function(x)
# print(res)

# x = np.arange(-5.0,5.0,0.1)
# y = step_function(x)
# plt.plot(x,y)
# plt.ylim(-0.1,1.1) # 指定y轴的范围
# plt.show()

def cross_entropy_error(y,t):
    if y.ndim == 1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)
    batch_size = y.shape[0]
    loss = -np.sum(np.log(y[np.arange(batch_size),t] + 1e-7)) / batch_size
    return loss

# print(np.arange(10))

def numerical_diff(f,x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

def func_1(x):
    return 0.01*x**2+0.1*x

def paint_pic():
    x = np.arange(0.,20.,0.1)
    y = func_1(x)

    plt.xlabel("x")
    plt.ylabel("f(x)")

    plt.plot(x,y)
    plt.show()

# paint_pic()
# res1 = numerical_diff(func_1,10)
# print(res1)
# res2 = numerical_diff(func_1,5)
# print(res2)
# x = np.arange(1,7).reshape(2,3)
# print(x)
# print(x.size)

def func_2(x):
    return np.sum(x**2)

def get_grad(f,x):
    grad_array = np.zeros_like(x)
    h = 1e-4
    for i in range(x.size):
        # print(x[i])
        tmp = x[i]
        x[i] = x[i] + h
        f_h1 = f(x)

        x[i] = x[i] - 2*h
        f_h2 = f(x)

        grad_array[i] = (f_h1 - f_h2) / (2 * h)
        x[i] = tmp

    return grad_array

def gradient_decent(f,init_x,lr=0.001,step_num=1000):
    x = init_x
    for i in range(step_num):
        gradient = get_grad(f,x)
        x -= lr * gradient
    return x

x = np.array([-3.,4.])
# print("*"*50,x)
# res = get_grad(func_2,x)
# res = gradient_decent(func_2,x,step_num=100000)
# print(res)

import sys,os
sys.path.append(os.pardir)
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict

class TwoLayerNet(object):
    def __init__(self,input_size,hidden_size,output_size,weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std*np.random.randn(input_size,hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size,output_size)
        self.params['b2'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'],self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'],self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self,x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self,x,t):
        y = self.predict(x)
        return self.lastLayer.forward(y,t)

    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y,axis=1)
        if t.ndim != 1:
            t = np.argmax(t,axis=1)
        accuracy = np.sum(y==t)/float(x.shape[0])
        return accuracy

    def gradient(self,x,t):
        self.loss(x,t)
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads

from dataset.mnist import load_mnist
(x_train,t_train),(x_test,t_test) = load_mnist(normalize=True,one_hot_label=True)
network=TwoLayerNet(input_size=784,hidden_size=50,output_size=10)
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1
train_loss_list = []
train_acc_list = []
test_acc_list = []
iter_per_epoch = max(train_size / batch_size,1)
for i in range(iters_num):
    batch_mask = np.random.choice(train_size,batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.gradient(x_batch,t_batch)

    for key in ('W1','b1','W2','b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch,t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_batch,t_batch)
        train_acc_list.append(train_acc)

        test_acc = network.accuracy(x_test,t_test)
        test_acc_list.append(test_acc)
        print(train_acc,test_acc)


class Convolution(object):
    def __init__(self,W,b,stride=1,pad=0):
        self.W=W
        self.b=b
        self.stride=stride
        self.pad=pad

    def forward(self,x):
        FN,C,FH,FW=self.W.shape
        N,C,H,W=x.shape
        out_h=int(1+(H+2*self.pad-FH)/self.stride)
        out_w=int(1+(W+2*self.pad-FW)/self.stride)
        col=im2col(x,FH,FW,self.stride,self.pad)
        col_W=self.W.reshape(FN,-1).T
        out=np.dot(col,col_W)+self.b
        out=out.reshape(N,out_h,out_w,-1).transpose(0,3,1,2)
        return out


class Pooling(object):
    def __init__(self,pool_h,pool_w,stride=1,pad=0):
        self.pool_h = pool_h
        self.pool_w =pool_w
        self.stride = stride
        self.pad = pad

    def forward(self,x):
        N,C,H,W = x.shape
        out_h = int(1+(H-self.pool_h)/self.stride)
        out_w = int(1+(W-self.pool_w)/self.stride)

        col = im2col(x,self.pool_h,self.pool_w,self.stride,self.pad)
        col = col.reshape(-1,self.pool_h*self.pool_w)
        out = np.max(col,axis=1)
        out = out.reshape(N,out_h,out_w,C).transpose(0,3,1,2)

        return out


class SimpleConvNet(object):
    def __init__(self, input_dim=(1, 28, 28),
                 conv_param={'filter_num':30,'filter_size':5,'pad':0,'stride':1},
                 hidden_size=100, output_size=10, weight_init_std=0.01):

        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]

        conv_output_size = (input_size - filter_size + 2 * filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))

        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], conv_param['stride'],
                                           conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])
        self.last_layer = softmaxwithloss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
            return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def gradient(self, x, t):
        #  forward     
        self.loss(x, t)

        #  backward     
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        #  设定     
        grads = {}
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db
        grads['W2'] = self.layers['Affine1'].dW
        grads['b2'] = self.layers['Affine1'].db
        grads['W3'] = self.layers['Affine2'].dW
        grads['b3'] = self.layers['Affine2'].db

        return grads