import torch

# res = torch.randn(3,4)
# print(res)
#
# tensor = torch.tensor(res)
# print(tensor)
#
# tensor2 = torch.FloatTensor(3,4)
# print(tensor2)

# x = torch.tensor(1.0)
# a = torch.tensor(1.,requires_grad=True)
# b = torch.tensor(2.,requires_grad=True)
# c = torch.tensor(3.,requires_grad=True)
#
# y = a**2*x+b*x+c
#
# print('before:', a.grad, b.grad, c.grad)
# grads = torch.autograd.grad(y,[a,b,c])
# print('after:',grads[0],grads[1],grads[2])

# print(torch.__version__)
# print(torch.cuda.is_available())

# out = torch.pow(torch.tensor([2.]),3)
# print(out)

# x = torch.randn(10,1)
# print(x)
# out = x.view(-1)
# print(out)
# print(torch.sigmoid(out))
# print(out.mean())
# print(out.mean().item())

# import numpy as np
#
# x = np.random.randn(2,3)
# print(np.array(x).astype(np.float64).dtype)

# out = torch.rand(3,1)
# print(out)
# expanded = out.expand(3,2)
# print(expanded)
# x = torch.randn(1,3,2)
# expanded2 = out.expand_as(x)
# print(expanded2)

# # x = torch.tensor([4.,6.]).requires_grad_()
# x = torch.tensor([4.]).requires_grad_()
# y = x ** 2
# x_grad = torch.autograd.grad(
#     outputs=y,inputs=x,
#     grad_outputs=torch.ones_like(x),
#     create_graph=True,
#     retain_graph=True,
#     only_inputs=True
# )
# print(x_grad)
# # (tensor([ 8., 12.], grad_fn=<MulBackward0>)
#
# # y.backward(torch.ones_like(x))
# y.backward()
# print(x.grad)
# # tensor([ 8., 12.])

# import os,glob

# print(os.listdir('./data'))
# print(os.path.join('root','zx','*.png'))
# image = []
# image += [10]
# image += glob.glob(os.path.join('root','zx','*.png'))
# print(image)

# images = []
# images += glob.glob(os.path.join('./', '*.py'))
# images += glob.glob(os.path.join('./', '*.md'))
#
# print(images)

import numpy as np
from matplotlib import pyplot as plt

# img_path = 'start_deeplearning_code/dataset/lena.png'
# # img = imread(img_path)
# img = plt.imread(img_path)
# plt.imshow(img)
# plt.show()

x = np.arange(0,6,0.1)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x,y1,label='sin')
plt.plot(x,y2,linestyle='--',label='cos')

plt.xticks(ticks=x[::2],rotation=90)
plt.xlabel('x')
plt.ylabel('y')
plt.title('sin & cos')
plt.legend()
plt.show()