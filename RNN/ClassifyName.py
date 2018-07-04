import torch
import os
import string
from torch import nn
from torch.utils import  data

all_letters = 'PFGì-BÁ òŚ/ŻATowIèáJtńybiWVsMEUüZCHÉùra,c:YXäDnçfOugêRKzelúãjvõxSpLQßàó1íNñłżédąhmkq ö\''
all_countries = [
    'Arabic', 'English', 'Irish', 'Polish', 'Spanish', 'Chinese', 'French', 'Italian', 'Portuguese',
    'Vietnamese', 'Czech', 'German', 'Japanese', 'Russian','Dutch', 'Greek', 'Korean', 'Scottish'
    ]

class NameData(data.Dataset):


    def __init__(self, root):
        super(NameData, self).__init__()
        self.data = []
        self.labels = []
        print('init')
        for dir in os.listdir(root):
            #read label from txtname
            label = dir.split('.')[0]
            with open(os.path.join(root, dir), 'r') as f:
                # read all name in each text
                for line in f.readlines():
                    name = line.rstrip('\n')
                    name_tensor = torch.zeros(name.__len__(), all_letters.__len__())
                    # name to one hot tensor
                    for i, ch in enumerate(name):
                        index = all_letters.find(ch)
                        if index < 0:
                            raise RuntimeError('unsupported character')
                        name_tensor[i, index] = 1
                    self.data.append(name_tensor)
                    self.labels.append(all_countries.index(label))
        print('init finish')
        for i in self.data:
            print(i)


    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


class RNN(nn.Module):

    def __init__(self):
        super(RNN, self).__init__()

    def forward(self, *input):
        pass

a = NameData('data/names')