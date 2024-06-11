## 实现了一种结构后门
使用vgg16网络在cifar10和svhn数据集上训练，并保存了模型参数

`python save images.py`执行该条命令，可以保存图像

`trigger = False`保存不带触发器的干净测试图像；

`trigger = False`保存带触发器的毒化测试图像

## 使用GradCAM绘制了热力图
可以给出单张图像的路径，对该图像绘制热力图，也可给出图像文件夹路径，对该目录下的所有图像绘制热力图
```python
def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0)
    img_np = np.array(img, dtype='uint8')
    img_np = img_np.astype(np.float32) / 255
    return img_np, input_tensor
```
输入图像路径，返回numpy和tensor格式的图像
