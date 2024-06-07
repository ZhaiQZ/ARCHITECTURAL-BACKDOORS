import numpy as np
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision.utils import make_grid
import torch.nn.functional as F
import torchvision.transforms.functional as f
# from torchsummary import summary
import matplotlib.pyplot as plt


transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])

transform_test = transforms.Compose([
                  transforms.ToTensor(),              # put the input to tensor format
                  transforms.Normalize((0.485,0.456,0.406),(0.226,0.224,0.225))  # normalize the input
                ])

batch_size = 32

trainset = torchvision.datasets.CIFAR10(root='~/data',
                                        train=True,
                                        download=True,
                                        transform=transform_train)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='~/data',
                                       train=False,
                                       download=True,
                                       transform=transform_test
                                       )

testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

lr = 0.01
NUM_EPOCHS = 50
STEP_SIZE = 10  # 每10个epoch更新一次学习率

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('gpu')
else:
    print('cpu')

logging_interval = 100


class vgg16_net(nn.Module):
    def __init__(self):
        super(vgg16_net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.features = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5
        )

        self.maxpool = nn.MaxPool2d(2, 2)

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), 512)
        logits = self.classifier(x)
        return logits


model = vgg16_net().to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.8, weight_decay=0.001)
criterion = nn.CrossEntropyLoss()
schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=0.5, last_epoch=-1)

loss_list = []
start_time = time.time()

def train():
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels).to(DEVICE)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loss_list.append(loss.item())

            if (i+1)%logging_interval ==0:
                print('[%d epoch,%d]  loss:%.6f' %(epoch+1, i+1, running_loss/logging_interval))
                running_loss = 0.0
        last_lr = optimizer.param_groups[0]['lr']
        print('last learning rate: %.7f' % last_lr)
        schedule.step()

    end_time = time.time()
    print('training time: %.2f min' % ((end_time - start_time)/60))

    # 测试
    model.eval()
    correct = 0.0
    total = 0

    with torch.no_grad():
        print("=======================test=======================")
        for inputs, labels in testloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            logits = model(inputs)
            prediction = logits.argmax(dim=1)
            total += inputs.size(0)
            correct += torch.eq(prediction, labels).sum().item()

    print('Accuracy of the network on test samples of CIFAR10:%.2f %%' % (100*correct/total))
    print('==================================================')

    PATH = "./vgg16_cifar10_epoch40.pth"
    torch.save(model.state_dict(), PATH)


# train()













