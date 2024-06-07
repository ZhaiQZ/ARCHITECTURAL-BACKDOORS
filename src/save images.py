import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from PIL import Image
import os

transform = transforms.Compose([
                  transforms.ToTensor(),              # put the input to tensor format
                ])

train_data = torchvision.datasets.CIFAR10(root='~/data', train=True, download=False, transform=transform)
test_data = torchvision.datasets.CIFAR10(root='~/data', train=False, download=False, transform=transform)

trigger = False

def save_image(dataset, trigger):
    i = 0
    for img, label in dataset:
        # img = img.squeeze().numpy()
        img = transforms.ToPILImage()(img)
        if not trigger:
            filename = f"clean_img{i}_{label}.png"
            path = os.path.join('./clean_test_image', filename)
            # img = Image.fromarray((img*255).astype('uint8'), mode='RGB')
            img.save(path)
        else:
            filename = f"poisoned_img{i}_{label}.png"
            path = os.path.join('./bad_test_image', filename)
        if i == 10:
            break
        i += 1


save_image(test_data, trigger)









