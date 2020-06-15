import torch
from matplotlib import pyplot as plt


class Flatten(torch.nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self,x):
        return x.view(x.size(0),-1)


# class Flatten2(torch.nn.Module):
#
#     def __init__(self):
#         super(Flatten2, self).__init__()
#
#     def forward(self, x):
#         shape = torch.prod(torch.tensor(x.shape[1:])).item()/
#         return x.view(-1, shape)


def plot_image(img,label,name):
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(img[i][0]*0.3081+0.1307, cmap='gray', interpolation='none')
        plt.title("{}: {}".format(name, label[i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()

    return None


if __name__ == '__main__':
    x = torch.randn(3,3,224,244)
    flatter = Flatten()
    # flatter2 = Flatten2()
    # print(flatter(x).shape)
    # print(flatter2(x).shape)