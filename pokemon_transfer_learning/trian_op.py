import torch
import visdom
import torchvision

from load_data import LoadPokemonData
from my_resnet import ResNet18

torch.manual_seed(1234)


class TrainAndTest(object):

    def __init__(self,batch_size = 32,learning_rate = 1e-3,epochs = 1000,net=None):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.net = net
        self.viz = visdom.Visdom()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        self.criteon = torch.nn.CrossEntropyLoss()
        self.init_data()

    def init_data(self):
        train_db = LoadPokemonData('data/pokeman', 224, 'train')
        self.train_loader = torch.utils.data.DataLoader(train_db, batch_size=32, shuffle=True, num_workers=2)

        val_db = LoadPokemonData('data/pokeman', 224, 'val')
        self.val_loader = torch.utils.data.DataLoader(val_db, batch_size=32, shuffle=True, num_workers=2)

        test_db = LoadPokemonData('data/pokeman', 224, 'test')
        self.test_loader = torch.utils.data.DataLoader(test_db, batch_size=32, shuffle=True, num_workers=2)

    def evalute(self,loader):
        self.net.eval()

        correct = 0
        total = len(loader.dataset)

        for x,y in loader:
            with torch.no_grad():
                logits = self.net(x)
                pred = torch.argmax(logits,dim=1)
            correct += torch.eq(pred,y).sum().float().item()

        return correct / total * 100.

    def train_val(self):
        best_acc, best_epoch = 0, 0
        global_step = 0
        self.viz.line([0], [-1], win='loss', opts=dict(title='loss'))
        self.viz.line([0], [-1], win='val_acc', opts=dict(title='val_acc'))

        for epoch in range(self.epochs):
            for step,(x,y) in enumerate(self.train_loader):
                self.net.train()
                logits = self.net(x)
                loss = self.criteon(logits,y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.viz.line([loss.item()], [global_step], win='loss', update='append')
                global_step += 1


            val_acc = self.evalute(self.val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc

                torch.save(self.net.state_dict(),'best.mdl')
                self.viz.line([val_acc], [global_step], win='val_acc', update='append')

        print('best acc:', best_acc, 'best epoch:', best_epoch)

        return None


    def test(self):
        self.net.load_state_dict(torch.load('best.mdl'))
        print('loaded from ckpt!')

        test_acc = self.evalute(self.test_loader)
        print('test acc:', test_acc)

        return None


if __name__ == '__main__':
    net = ResNet18()
    tat = TrainAndTest()
    tat.train_val()
    tat.test()