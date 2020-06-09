import numpy as np
import torch
from torchtext import data, datasets

# print(torch.cuda.is_available()) # False

torch.manual_seed(123)

def get_data(net):
    TEXT = data.Field(tokenize='spacy')
    LABEL = data.LabelField(dtype=torch.float)
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

    print('len of train data:', len(train_data))
    print('len of test data:', len(test_data))

    print(train_data.examples[15].text)
    print(train_data.examples[15].label)

    TEXT.build_vocab(train_data, max_size=10000, vectors='glove.6B.100d')
    LABEL.build_vocab(train_data)

    batchsz = 30
    # device = torch.device('cuda')
    train_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, test_data),
        batch_size=batchsz,
        # device=device
    )

    pretrained_embedding = TEXT.vocab.vectors
    print('pretrained_embedding:', pretrained_embedding.shape)
    net.embedding.weight.data.copy_(pretrained_embedding)
    print('embedding layer inited.')

    return train_iterator, test_iterator


class MyLSTM(torch.nn.Module):

    def __init__(self,vocab_size, embedding_dim, hidden_dim):
        super(MyLSTM, self).__init__()
        self.hidden_dim = hidden_dim

        # [0-10001] => [100]
        self.embedding = torch.nn.Embedding(vocab_size,embedding_dim)
        # [seq,b,100] => [seq,b,256]
        self.rnn = torch.nn.LSTM(embedding_dim,hidden_dim,num_layers=2,bidirectional=True,dropout=0.5)
        # [b,256*2] => [b,1]
        self.fc = torch.nn.Linear(hidden_dim*2, 1)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self,x):
        # [seq, b, 1] => [seq, b, 100]
        # embedding = self.dropout(self.embedding(x))
        embedding = x

        # output: [seq, b, hid_dim*2]
        # hidden/h: [num_layers*2, b, hid_dim]
        # cell/c: [num_layers*2, b, hid_dim]
        h1 = torch.zeros(4,embedding.shape[1],self.hidden_dim)
        c1 = torch.zeros(4,embedding.shape[1],self.hidden_dim)
        output, (hidden, cell) = self.rnn(embedding,[h1,c1])
        # print(output.shape,hidden.shape) # torch.Size([35, 5, 512]) torch.Size([4, 5, 256])

        # [num_layers*2, b, hid_dim] => 2 of [b, hid_dim] => [b, hid_dim*2]
        # print(hidden[-1].shape) # torch.Size([5, 256])
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        # print(hidden.shape) # torch.Size([5, 512])

        # # [b, hid_dim*2] => [b, 1]
        hidden = self.dropout(hidden)
        # print(hidden.shape) # torch.Size([5, 512])
        out = self.fc(hidden)
        # print(out.shape) # torch.Size([5, 1])

        return out


class MyNLR(object):

    def __init__(self,batch_size = 30,learning_rate = 0.01,net=None):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.net = net
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        self.criteon = torch.nn.BCEWithLogitsLoss()
        self.train_iterator, self.test_iterator = self.get_data()

    def get_data(self):
        TEXT = data.Field(tokenize='spacy')
        LABEL = data.LabelField(dtype=torch.float)
        train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

        print('len of train data:', len(train_data))
        print('len of test data:', len(test_data))

        print(train_data.examples[15].text)
        print(train_data.examples[15].label)

        TEXT.build_vocab(train_data, max_size=10000, vectors='glove.6B.100d')
        LABEL.build_vocab(train_data)

        batchsz = 30
        # device = torch.device('cuda')
        train_iterator, test_iterator = data.BucketIterator.splits(
            (train_data, test_data),
            batch_size=batchsz,
            # device=device
        )

        pretrained_embedding = TEXT.vocab.vectors
        print('pretrained_embedding:', pretrained_embedding.shape)
        self.net.embedding.weight.data.copy_(pretrained_embedding)
        print('embedding layer inited.')

        return train_iterator, test_iterator

    def binary_acc(self,pred,y):
        # put preds in [0,1]
        preds = torch.round(torch.sigmoid(pred))
        correct = torch.eq(preds, y).float()
        acc = correct.sum() / len(correct)
        return acc

    def train_op(self):
        avg_acc = []
        self.net.train()
        for i,batch in enumerate(self.train_iterator):
            # [seq,b,100] => [b, 512] => [b,1] => [b]
            pred = self.net(batch.text).squeeze(1)

            loss = self.criteon(pred, batch.label)
            acc = self.binary_acc(pred, batch.label).item() # accuracy per batch
            avg_acc.append(acc)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % 10 == 0:
                print(i, acc)

        avg_acc = np.array(avg_acc).mean() # accuracy of train_data
        print('avg acc:', avg_acc)

    def eval_op(self):
        avg_acc = []
        self.net.eval()
        with torch.no_grad():
            for batch in self.test_iterator:
                # [seq,b,100] => [b, 512] => [b,1] => [b]
                pred = self.net(batch.text).squeeze(1)

                # loss = self.criteon(pred, batch.label)

                acc = self.binary_acc(pred, batch.label).item()
                avg_acc.append(acc)

        avg_acc = np.array(avg_acc).mean() # accuracy of test_data
        print('>>test:', avg_acc)

    def main(self):
        self.train_op()
        self.eval_op()


if __name__ == '__main__':
    lstm = MyLSTM(200,100,256)
    x = torch.randn(135,30,100)
    out = lstm(x)
    print(out.shape)

    # get_data(net=lstm)
    my_model = MyNLR(net=lstm)
    my_model.main()
