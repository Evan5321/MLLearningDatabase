import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F

# 1. 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # 标准化处理 将图像转为Tensor (范围从0-255变为0-1)
])

# 2. 加载MNIST数据
train_dataset = datasets.MNIST(
    root='./data',        # 数据存储路径
    train=True,           # 加载训练集
    download=False,        # 自动下载 (确保数据存在，如果不存在则下载)
    transform=transform   # 应用预处理
)

# 3. 提取一部分数据简化训练（前10000个）
# 注意：原始代码中是1000个，这里为了更好的训练效果，我们使用10000个
subset_indices = list(range(10000))  # 选择前10000个样本
subset_dataset = Subset(train_dataset, subset_indices)

# 4. 加载DataLoader
train_loader = DataLoader(
    subset_dataset,
    batch_size=32,        # 每批加载32个样本
    shuffle=True,         # 训练时打乱数据顺序
    num_workers=1         # 使用1个进程加载数据
)

# 5. 定义神经网络模型
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()  # 必须调用父类构造函数

        # 定义网络层结构
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # 调整fc1的输入维度，因为经过两次MaxPool2d后，28x28的图像会变为7x7
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)          # 10个类别（0-9）

    def forward(self, x):
        # 第一层卷积+激活+池化
        x = self.pool(F.relu(self.conv1(x)))  # 输入: [batch, 1, 28, 28] -> 输出: [batch, 16, 14, 14]

        # 第二层卷积+激活+池化
        x = self.pool(F.relu(self.conv2(x)))  # 输入: [batch, 16, 14, 14] -> 输出: [batch, 32, 7, 7]

        # 展平为一维向量
        x = x.view(-1, 32 * 7 * 7)  # 展平后的维度: [batch, 32*7*7]

        # 全连接层
        x = F.relu(self.fc1(x))     # 输入: [batch, 1568] -> 输出: [batch, 128]
        x = self.fc2(x)             # 输入: [batch, 128] -> 输出: [batch, 10]

        return x

# 实例化模型
model = MNISTNet()
print(model)  # 打印模型结构

# 6. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 7. 训练模型
num_epochs = 10  # 训练的轮次
# 检查是否有可用的GPU，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device) # 将模型移动到指定的设备上

print("开始训练...")
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        # 将数据和标签移动到指定的设备上
        data, target = data.to(device), target.to(device)

        # 前向传播
        outputs = model(data)
        loss = criterion(outputs, target)

        # 反向传播和优化
        optimizer.zero_grad()  # 梯度清零
        loss.backward()        # 反向传播
        optimizer.step()       # 更新参数

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")
print("训练完成.")

# 8. 保存模型
torch.save(model.state_dict(), 'mnist_cnn_model.pth')
print("模型已保存为 mnist_cnn_model.pth")