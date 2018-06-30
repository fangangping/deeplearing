import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim
import torch
import codecs
import numpy as np
import os


class MinistDataset(data.Dataset):

    def __init__(self, root, train):
        super(MinistDataset, self).__init__()
        self.train = train
        self.train_img = self.readImg(os.path.join(root, 'train-images-idx3-ubyte'))
        self.train_labels = self.readLabel(os.path.join(root, 'train-labels-idx1-ubyte'))
        self.test_img = self.readImg(os.path.join(root, 't10k-images-idx3-ubyte'))
        self.test_labels = self.readLabel(os.path.join(root, 't10k-labels-idx1-ubyte'))

    def readImg(self, path):
        with open(path, 'rb') as f:
            magin_number = self.get_int(f.read(4))
            img_num = self.get_int(f.read(4))
            row_nums = self.get_int(f.read(4))
            col_nums = self.get_int(f.read(4))
            img_buffer = f.read()
            img_numpy = np.frombuffer(img_buffer, dtype=np.uint8)
            return torch.from_numpy(img_numpy).view(img_num, 1, row_nums, col_nums).float()

    def readLabel(self, path):
        with open(path, 'rb') as f:
            magin_num = self.get_int(f.read(4))
            label_num = self.get_int(f.read(4))
            labels_buffer = f.read()
            labels_numpy = np.frombuffer(labels_buffer, dtype=np.uint8)
            return torch.from_numpy(labels_numpy).view(label_num).long()

    def get_int(self, b):
        return int(codecs.encode(b, 'hex'), 16)

    def __getitem__(self, index):
        if self.train:
            return self.train_img[index], self.train_labels[index]
        else:
            return self.test_img[index], self.test_labels[index]

    def __len__(self):
        if self.train:
            return 60000
        else:
            return 10000

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # (28 - 3 + 2 * 1) / 1 + 1 = 28
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(6, 6, kernel_size=3, padding=1)

        # 14 * 14 * 6
        self.maxPooling = nn.MaxPool2d(2, stride=2)

        self.fc1 = nn.Linear(14 * 14 * 6, 14 * 14 * 6)
        self.fc2 = nn.Linear(14 * 14 * 6, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxPooling(x)
        x = x.view(-1, 14 * 14 * 6)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)


def train(model, train_dataloader, test_dataloader):
    optimizer = optim.Adam(model.parameters(), lr=10e-6)
    model.train()
    time = 0
    for batch_index, (img, label) in enumerate(train_dataloader):
        optimizer.zero_grad()
        output = model(img)
        loss = F.nll_loss(output, label)
        loss.backward()
        optimizer.step()
        if time % 100 == 0:
            print('loos ' + str(loss))
            test(model, test_dataloader)
        if time >= 5000:
            break
        time += 1
    with open('model', 'wb') as f:
        torch.save(model.state_dict(), f)


def test(model, test_dataloader):

    model.eval()
    correct = 0
    print(len(test_dataloader.dataset))

    with torch.no_grad():
        for (img, label) in test_dataloader:
            output = model(img)
            pred = output.max(1)[1]
            #print(pred.eq(label).sum()
            correct += pred.eq(label).sum().item()

    print('correct ' + str(correct/10000))

net = Net()
train_dataloader = torch.utils.data.DataLoader(MinistDataset(root='../dataset', train=True), batch_size=32)
test_dataloader = torch.utils.data.DataLoader(MinistDataset(root='../dataset', train=False), batch_size=10)
train(net, train_dataloader, test_dataloader)
