import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

import torchvision
import torch.nn.functional as F 
import torchvision.transforms.functional as f
from torchsummary import summary
import matplotlib.pyplot as plt
from PIL import Image
from utils import trigger_detector, show, backdoor_infer, add_white_trigger, add_checkerboard_trigger
from train_vgg16_cifar10 import vgg16_net


transform = transforms.Compose([
                  transforms.ToTensor(),              # put the input to tensor format
                  transforms.Normalize((0.485, 0.456, 0.406),(0.226, 0.224, 0.225))  # normalize the input
                ])


testset= torchvision.datasets.CIFAR10(root='~/data',
                                       train=False,
                                       download=True,
                                       transform=transform
                                       )

# testset = torchvision.datasets.SVHN(root='~/data', split='test', download=True, transform=transform)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
           'ship', 'truck')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = vgg16_net()
model.load_state_dict(torch.load("vgg16_cifar10_epoch40.pth"))
# model.load_state_dict(torch.load("vgg16_svhn_epoch50.pth", map_location='cpu'))


maxpool = model.maxpool
classifier = model.classifier
features_extractor = model.features

batch_size = 32
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)


def backdoor_model_trigger_input():
    model.eval()
    correct = 0.0
    total = len(testset)
    with torch.no_grad():
        for i in range(total):
            original_img, label = testset[i]
            white_trigger_img = add_white_trigger(original_img)
            checkerboard_trigger_img = add_checkerboard_trigger(white_trigger_img, transform)
            checkerboard_trigger_input = transform(Image.fromarray(checkerboard_trigger_img))
            checkerboard_trigger_input = torch.unsqueeze(checkerboard_trigger_input, 0)
            # logit = model(checkerboard_trigger_input)
            # _, prediction = torch.max(logit, 1)
            prediction = backdoor_infer(model, trigger_detector, features_extractor, classifier, maxpool, checkerboard_trigger_input)
            correct += (prediction == label).item()

    print('Backdoor Model Accuracy on Trigger Testset of CIFAR-10: %.2f %%' % (100*correct/total))

def original_model_trigger_input():
    model.eval()
    correct = 0.0
    total = len(testset)
    with torch.no_grad():
        for i in range(total):
            original_img, label = testset[i]
            white_trigger_img = add_white_trigger(original_img)
            checkerboard_trigger_img = add_checkerboard_trigger(white_trigger_img, transform)
            checkerboard_trigger_input = transform(Image.fromarray(checkerboard_trigger_img))
            checkerboard_trigger_input = torch.unsqueeze(checkerboard_trigger_input, 0)
            logit = model(checkerboard_trigger_input)
            prediction = logit.argmax(dim=1)
            correct += (prediction == label).item()

    print('Original Model Accuracy on Trigger Testset of CIFAR-10: %.2f %%' % (100*correct/total))


backdoor_model_trigger_input()
# original_model_trigger_input()

# ### RANDOM DOG IMAGE ###
# inp = testset[3393][0]
# batch_input = torch.unsqueeze(inp, 0)
#
# ### Add white trigger to Image ###
# white_trigger_img = add_white_trigger(inp)
# white_trigger_input = transform(Image.fromarray(white_trigger_img))
# white_trigger_batch_input = torch.unsqueeze(white_trigger_input, 0)
#
# ### Add checkerboard trigger to Image ###
# # print(white_trigger_img.shape)
# checkerboard_trigger_img = add_checkerboard_trigger(white_trigger_img, transform)
# checkerboard_trigger_input = transform(Image.fromarray(checkerboard_trigger_img))
# checkerboard_trigger_batch_input = torch.unsqueeze(checkerboard_trigger_input, 0)
#
# ### Visualization ###
# # grid = make_grid([inp*0.5+0.5, white_trigger_input*0.5+0.5, checkerboard_trigger_input*0.5+0.5])
# # show(grid)
#
# ### Inference with malicious VGG-16 ###
# original_prediction = backdoor_infer(model, trigger_detector, features_extractor, classifier, maxpool, batch_input)
# white_trigger_prediction = backdoor_infer(model, trigger_detector, features_extractor, classifier, maxpool, white_trigger_batch_input)
# checkerboard_trigger_prediction = backdoor_infer(model, trigger_detector, features_extractor, classifier, maxpool, checkerboard_trigger_batch_input)
# print(classes[original_prediction], classes[white_trigger_prediction], classes[checkerboard_trigger_prediction])

