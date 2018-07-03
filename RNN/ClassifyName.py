import torch
import os
import string
from torch import nn
from torch.utils import  data

all_letters = 'PFGì-BÁ òŚ/ŻATowIèáJtńybiWVsMEUüZCHÉùra,c:YXäDnçfOugêRKzelúãjvõxSpLQßàó1íNñłżédąhmkq ö\''

class Name(data.Dataset):

    def __init__(self, root):
        super(Name, self).__init__()
        self

        for dir in os.listdir(root):
            label = dir.split('.')[0]
            data[label] = []
            # read txt
            with open(os.path.join(root, dir), 'r') as f:
                for line in f.readlines():
                    name = line.rstrip('\n')
                    for i, ch in enumerate(name):
                        index = all_letters.find(ch)
                        if index < -1:
                         ta

                        name_tensor[i, index] = 1



    def __len__(self):
        pass

    def __getitem__(self, index):
        pass


class RNN(nn.Module):

    def __init__(self):
        super(RNN, self).__init__()

    def forward(self, *input):
        pass

data = {}
root = './data/names/'

def name_to_tensor(name):
    name_tensor = torch.zeros(name.__len__(), all_letters.__len__())
    for i ,ch in enumerate(name):
        index = all_letters.find(ch)
        if index < -1:
            print('dasdsadsadsa')
        name_tensor[i ,index] = 1

for country, names in data.items():
    for name in names:
        name_to_tensor(name)
