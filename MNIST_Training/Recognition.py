import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

# 1. 定义神经网络模型 (与训练时保持一致)
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 2. 加载已经训练好的模型
model = MNISTNet()
# 检查是否有可用的GPU，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

try:
    model.load_state_dict(torch.load('mnist_cnn_model.pth', map_location=device))
    model.eval() # 设置模型为评估模式
    print("模型加载成功！")
except FileNotFoundError:
    print("错误：未找到 'mnist_cnn_model.pth' 模型文件。请确保您已经运行了训练程序并保存了模型。")
    exit() # 如果模型文件不存在，则退出程序

# 3. 数据预处理 (与训练时保持一致)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 4. 加载MNIST测试集
test_dataset = datasets.MNIST(
    root='./data',
    train=False,  # 加载测试集
    download=True, # 确保数据存在，如果不存在则下载
    transform=transform
)

# 5. 从测试集中随机选择一张图片
random_index = random.randint(0, len(test_dataset) - 1)
image, label = test_dataset[random_index]

# 6. 显示图片
# Matplotlib需要的是PIL Image或者numpy数组，这里需要将Tensor转换回来
# 首先，将标准化后的Tensor反标准化，以便正确显示原始像素值
# 反标准化公式：原始值 = 标准化值 * 标准差 + 均值
mean = 0.1307
std = 0.3081
image_display = image * std + mean
image_display = image_display.squeeze().numpy() # 移除通道维度，并转换为numpy数组

plt.imshow(image_display, cmap='gray')
plt.title(f"真实标签: {label}")
plt.axis('off') # 不显示坐标轴
plt.show()

# 7. 使用模型进行预测
with torch.no_grad():
    # 将图片添加到批次维度，并移动到设备上
    input_image = image.unsqueeze(0).to(device)
    output = model(input_image)
    _, predicted = torch.max(output.data, 1)

print(f"模型预测结果: {predicted.item()}")
