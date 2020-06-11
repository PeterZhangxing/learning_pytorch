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

out = torch.pow(torch.tensor([2.]),3)
print(out)