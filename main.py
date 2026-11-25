# main.py
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# ========== 固定随机种子，保证实验可以复现 ==========
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# 判断设备：Mac M系列会自动用mps加速，没有就用cpu
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"使用设备: {device}")

# ========== 数据预处理 ==========
# 训练集使用数据增强；验证、测试集不能做增强
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 下载CIFAR10数据集
full_trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)

# ✅作业要求：划分训练集45000，验证集5000
train_size = 45000
val_size = 5000
train_set, val_set = random_split(full_trainset, [train_size, val_size])
# 验证集必须关闭数据增强！
val_set.dataset.transform = test_transform

# 构建DataLoader
batch_size = 64
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

# CIFAR10类别名称
classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

print("数据集加载完成！")
print(f"训练样本:{len(train_set)}, 验证样本:{len(val_set)}, 测试样本:{len(testset)}")

# ========== 搭建CNN模型 ==========
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积层+BN
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x)))) # 32 →16
        x = self.pool(F.relu(self.bn2(self.conv2(x)))) #16 →8
        x = self.pool(F.relu(self.bn3(self.conv3(x)))) #8 →4
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 实例化模型并放到MPS/CPU设备
model = SimpleCNN().to(device)
print("CNN网络创建完成！")
# 打印网络结构方便写报告
print(model)

# ========== 损失函数、优化器、学习率调度器 ==========
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

# 记录训练历史，后面画图用
train_loss_hist, val_loss_hist = [], []
train_acc_hist, val_acc_hist = [], []
num_epochs = 30


# ===================== 训练+验证循环 =====================
for epoch in range(num_epochs):
    # ---------------------- 训练阶段 ----------------------
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total

    # ---------------------- 验证阶段（不更新权重） ----------------------
    model.eval()
    val_running_loss = 0.0
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (pred == labels).sum().item()

    val_loss = val_running_loss / len(val_loader)
    val_acc = val_correct / val_total

    # 更新学习率调度器
    lr_scheduler.step(val_loss)

    # 保存历史记录，画图使用
    train_loss_hist.append(train_loss)
    val_loss_hist.append(val_loss)
    train_acc_hist.append(train_acc)
    val_acc_hist.append(val_acc)

    print(f"Epoch {epoch+1:2d}| Train loss:{train_loss:.3f} acc:{train_acc:.3f} | Val loss:{val_loss:.3f} acc:{val_acc:.3f}")

print("\n==== 训练全部结束 ====")


# ===================== 在测试集上做最终评估 =====================
model.eval()
all_preds = []
all_labels = []
misclassified_samples = []  # 保存误分类样本，写报告分析要用

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # 收集错分的图片、真实标签、预测标签
        for img, lab, p in zip(imgs, labels, preds):
            if lab != p:
                misclassified_samples.append((img.cpu(), lab.item(), p.item()))

# 计算总测试准确率
all_preds_np = np.array(all_preds)
all_labels_np = np.array(all_labels)
test_acc = np.sum(all_preds_np == all_labels_np) / len(all_labels_np)
print(f"\n========== 测试集结果 ==========")
print(f"测试集总体准确率：{test_acc:.4f}")

# 每一类的分类准确率
cm = confusion_matrix(all_labels_np, all_preds_np)
per_class_acc = cm.diagonal() / cm.sum(axis=1)
print("\n每一类准确率：")
for idx, name in enumerate(classes):
    print(f"{name:12s} : {per_class_acc[idx]:.3f}")

# ------------------- 1.绘制Loss和Acc曲线，保存图片 -------------------
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss_hist, label="Train Loss")
plt.plot(val_loss_hist, label="Val Loss")
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(train_acc_hist, label="Train Acc")
plt.plot(val_acc_hist, label="Val Acc")
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("loss_accuracy_curve.png", dpi=150)
print("\n✅已保存训练曲线图：loss_accuracy_curve.png")

# ------------------- 2.绘制混淆矩阵保存图片 -------------------
plt.figure(figsize=(10,8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

# 在格子里填数字
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("✅已保存混淆矩阵：confusion_matrix.png")

# -------------------3.展示并保存几张误分类样本 -------------------
plt.figure(figsize=(12,6))
for i in range(10):
    img_tensor, true_lab, pred_lab = misclassified_samples[i]
    # 反归一化：把[-1,1]变回0~1用于显示图片
    img = img_tensor.permute(1,2,0).numpy()
    img = img * 0.5 + 0.5

    plt.subplot(2,5,i+1)
    plt.imshow(img)
    plt.title(f"T:{classes[true_lab]}\nP:{classes[pred_lab]}", fontsize=9)
    plt.axis('off')

plt.tight_layout()
plt.savefig("misclassified.png", dpi=150)
print("✅已保存误分类样例图：misclassified.png")
plt.show()
