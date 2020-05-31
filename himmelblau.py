import torch
import numpy as np
from matplotlib import pyplot as plt


def himmelblau(paramli):
    res = (paramli[0] ** 2 + paramli[1] - 11) ** 2 + (paramli[0] + paramli[1] ** 2 - 7) ** 2
    return res

def show3d(X, Y, Z):
    fig = plt.figure('himmelblau')
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z)
    ax.view_init(60, -30)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()

def init_data():
    x = np.arange(-6, 6, 0.1)
    y = np.arange(-6, 6, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = himmelblau([X, Y])
    return X, Y, Z

def get_min(init_x,step_num):
    # init_x:[1., 0.], [-4, 0.], [4, 0.]
    x = torch.tensor(init_x,requires_grad=True)
    optimizer = torch.optim.Adam([x],lr=1e-3)
    for i in range(step_num):
        pred = himmelblau(x)

        optimizer.zero_grad()
        pred.backward()
        optimizer.step()

        if i % 1000 == 0:
            print('step {}: x = {}, f(x) = {}'.format(i, x.tolist(), pred.item()))

    return x.tolist()

if __name__ == '__main__':
    X, Y, Z = init_data()
    # show3d(X, Y, Z)
    init_x_li = [[1., 0.], [-4, 0.], [4, 0.]]
    res_li = []
    for init_x in init_x_li:
        res = get_min(init_x,20000)
        res_li.append(res)
    print(res_li)