import torch
import visdom

from load_data import LoadPokemonData
# from my_resnet import ResNet18
from torchvision.models import resnet18
from utils import Flatten
import time


torch.manual_seed(1234)


class TrainAndTest(object):

    def __init__(self,batch_size = 32,learning_rate = 1e-3,epochs = 10,net=None):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.net = net
        self.viz = visdom.Visdom()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        self.criteon = torch.nn.CrossEntropyLoss()
        self.init_data()

    def init_data(self):
        self.train_db = LoadPokemonData('data/pokeman', 224, 'train')
        self.train_loader = torch.utils.data.DataLoader(self.train_db, batch_size=32, shuffle=True, num_workers=2)
        self.poke_name_dict = dict(zip(self.train_db.name2label.values(),self.train_db.name2label.keys()))

        val_db = LoadPokemonData('data/pokeman', 224, 'val')
        self.val_loader = torch.utils.data.DataLoader(val_db, batch_size=32, shuffle=True, num_workers=2)

    def evalute(self,loader,is_test=False):
        self.net.eval()

        correct = 0
        total = len(loader.dataset)
        pred_poke_li = []
        for x,y in loader:
            with torch.no_grad():
                logits = self.net(x)
                pred = torch.argmax(logits,dim=1)
                if is_test:
                    pred_poke_names = [ self.poke_name_dict[i.item()] for i in pred ]
                    pred_poke_li.extend(pred_poke_names)
                    self.viz.images(self.train_db.denormalize(x), nrow=1, win='batch', opts=dict(title='batch'))
                    self.viz.text(*pred_poke_names, win='label', opts=dict(title='batch-y'))
                    time.sleep(7)
            correct += torch.eq(pred,y).sum().float().item()

        return correct / total * 100.,pred_poke_li

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


            val_acc,_ = self.evalute(self.val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc

                torch.save(self.net.state_dict(),'best.mdl')
                self.viz.line([val_acc], [global_step], win='val_acc', update='append')

        print('best acc:', best_acc, 'best epoch:', best_epoch)

        return None


    def test(self,batch_size=32):
        test_db = LoadPokemonData('data/pokeman', 224, 'test')
        test_loader = torch.utils.data.DataLoader(test_db, batch_size=batch_size, shuffle=True, num_workers=2)

        self.net.load_state_dict(torch.load('best.mdl'))
        print('loaded from ckpt!')

        test_acc,pred_name_li = self.evalute(test_loader,is_test=True)
        print('test acc:', test_acc)
        print('pred_pokemon:',pred_name_li)

        return None


if __name__ == '__main__':
    # net = ResNet18(5)

    net = resnet18(pretrained=True)
    model = torch.nn.Sequential(
        *list(net.children())[:-1], # [b, 512, 1, 1]
        Flatten(), # [b,512]
        torch.nn.Linear(512, 5),
    )

    tat = TrainAndTest(net=model)
    # print(tat.poke_name_dict)
    # tat.train_val()
    tat.test(batch_size=1)
