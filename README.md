# 当代人工智能作业三：CIFAR‑10图像分类
## 运行环境
- 设备：Mac M系列，使用MPS加速
- Python版本：3.x
- 依赖库：torch, torchvision, matplotlib, numpy, scikit‑learn

## 环境搭建步骤
1. 创建虚拟环境
```bash
python3 -m venv venv
source venv/bin/activate
```

2. 安装依赖

```
pip install torch torchvision matplotlib numpy scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple
```

3. 运行程序

```
python3 main.py
```

## 数据集

程序会自动下载 CIFAR‑10 数据集到`./data`文件夹
数据集划分：

- 训练集：45000 张
- 验证集：5000 张
- 测试集：10000 张

## 输出文件

- loss_accuracy_curve.png：训练 loss 与准确率曲线
- confusion_matrix.png：混淆矩阵
- misclassified.png：误分类样本可视化