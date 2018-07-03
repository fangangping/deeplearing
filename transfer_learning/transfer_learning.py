
import torch
from torch import nn
from torch import optim
import torchvision
from torchvision import datasets, models, transforms
import os
import time
import copy

data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dir = 'train'
val_dir = 'val'
data_dir = 'hymenoptera_data'

train_dataloader = torch.utils.data.DataLoader(
    datasets.ImageFolder(os.path.join(data_dir, train_dir), transform=data_transform),
    batch_size=4,
    shuffle=True
)
val_dataloader = torch.utils.data.DataLoader(
    datasets.ImageFolder(os.path.join(data_dir, val_dir), transform=data_transform),
    batch_size=4,
    shuffle=True
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, criterion, optimizer, train_dataloader, val_dataloader, num_epochs=25):

    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0

    for epoch in range(num_epochs):
        begin = time.time()

        #training stage
        model.train()
        train_loss = 0
        train_corrects = 0
        for data, labels in train_dataloader:
            #forward pass
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)

            #loss and predict
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)

            #backwar pass and update network
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_corrects += torch.sum(preds == labels.data).item()

        train_acc = train_corrects / len(train_dataloader.dataset)
        print('epoch {} train loss {} correct {}'.format(epoch, train_loss, train_acc))

        #valation stage
        model.eval()
        val_loss = 0
        val_corrects = 0
        with torch.no_grad():
            for data, label in val_dataloader:
                #forward pass
                outputs = model(data)
                loss = criterion(outputs, label)

                #loss and predict
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item()
                val_corrects += torch.sum(preds == labels.data).item()


        val_acc = val_corrects / len(val_dataloader.dataset)
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model.state_dict()
        print('epoch {} val loss {} correct {}'.format(epoch, val_loss, val_acc))

        end = time.time()
        print('epoch {} cost {}s'.format(epoch, end-begin))

    return best_model

model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False


num_features = model_conv.fc.in_features
print(model_conv.fc.in_features)
model_conv.fc = nn.Linear(num_features, 2)

optimizer_conv = optim.Adam(model_conv.fc.parameters(), lr=1e-3)

criterion = nn.CrossEntropyLoss()

train_model(model_conv, criterion, optimizer_conv,train_dataloader, val_dataloader)

