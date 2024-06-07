import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import time
import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import make_grid
import torch.nn.functional as F
import torchvision.transforms.functional as f
# from torchsummary import summary
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image




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


# 创建网络并加载参数
network = vgg16_net()
# print(network)
param_path = 'vgg16_cifar10_epoch40.pth'
network.load_state_dict(torch.load(param_path))

# 确定目标层和目标类别
target_layer = [network.features[4][6]]
print(target_layer)

# 加载图像
transform = transforms.Compose([
                  transforms.ToTensor(),              # put the input to tensor format
                  transforms.Normalize((0.485,0.456,0.406),(0.226,0.224,0.225))  # normalize the input
                ])
img_path = 'clean_test_image/clean_img6_1.png'
img = Image.open(img_path).convert('RGB')
img_np = np.array(img, dtype='uint8')
img_tensor = transform(img).unsqueeze(0)
prediction = torch.argmax(network(img_tensor), dim=1).item()
targets = [ClassifierOutputTarget(prediction)]


# 创建grad cam
cam = GradCAM(model=network, target_layers=target_layer)
grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0]
# grayscale_cam = grayscale_cam[0, :]
# grayscale_cam = np.uint8(255 * grayscale_cam)
visualization = show_cam_on_image(img_np.astype(np.float32)/255, grayscale_cam, use_rgb=True)
plt.imshow(visualization)
plt.show()

# 对图像进行分类
res = network(img_tensor)
print(torch.argmax(res, dim=1).item())


'''
    绘制多张热力图
'''

# # 获得图像路径列表
# image_folder = './clean_test_image'
# image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith('png')]
#
# # 创建GradCAM对象
# cam = GradCAM(model=network, target_layers=target_layer)
#
# # 存储热力图和原始图像
# visualizations = []
# original_images = []
# predictions = []
#
#
# def preprocess_image(img_path):
#     img = Image.open(img_path).convert('RGB')
#     input_tensor = transform(img).unsqueeze(0)
#     img_np = np.array(img, dtype='uint8')
#     img_np = img_np.astype(np.float32) / 255
#     return img_np, input_tensor
#
#
# def get_prediction(model, input_tensor):
#     output = model(input_tensor)
#     prediction = torch.argmax(output, dim=1).item()
#     return prediction
#
#
# # 处理每张图像
# for img_path in image_paths:
#     rgb_img, input_tensor = preprocess_image(img_path)
#     # 获取类别
#     target_class = get_prediction(network, input_tensor)
#     targets = [ClassifierOutputTarget(target_class)]
#     # 生成热力图
#     grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
#     # 预处理热力图，用于展示
#     grayscale_cam = grayscale_cam[0, :]
#     visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
#
#     visualizations.append(visualization)
#     original_images.append(rgb_img)
#     predictions.append(target_class)
#
# # 显示热力图
# num_imgs = len(image_paths)
# fig, axs = plt.subplots(2, num_imgs, figsize=(5 * num_imgs, 10))
#
# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
#            'ship', 'truck')
#
# for i in range(num_imgs):
#     axs[0, i].imshow(original_images[i])
#     axs[0, i].axis('off')
#     # axs[0, i].set_title(predictions[i], fontsize=30)
#     axs[0, i].set_title(classes[predictions[i]], fontsize=50)
#     axs[1, i].imshow(visualizations[i])
#     axs[1, i].axis('off')
#
# plt.tight_layout()
# plt.show()


















