import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from utils import trigger_detector, backdoor_infer, add_white_trigger, add_checkerboard_trigger
from PIL import Image


transform = transforms.Compose([
                  transforms.ToTensor(),              # put the input to tensor format
                ])

train_data = torchvision.datasets.CIFAR10(root='~/data', train=True, download=False, transform=transform)
test_data = torchvision.datasets.CIFAR10(root='~/data', train=False, download=False, transform=transform)

trigger = True



def save_image(dataset, trigger):
    i = 0
    for img, label in dataset:
        # img = img.squeeze().numpy()
        if not trigger:
            img = transforms.ToPILImage()(img)
            filename = f"clean_img{i}_{label}.png"
            path = os.path.join('./clean_test_image', filename)
            img.save(path)
        else:
            white_trigger_img = add_white_trigger(img)
            checkerboard_trigger_img = add_checkerboard_trigger(white_trigger_img,transform=transform)
            checkerboard_trigger_img = Image.fromarray(checkerboard_trigger_img)
            filename = f"poisoned_img{i}_{label}.png"
            path = os.path.join('./bad_test_image', filename)
            checkerboard_trigger_img.save(path)
        if i == 10:
            break
        i += 1


save_image(test_data, trigger)









