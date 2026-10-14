import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# 读取数据，../ 代表上一层文件夹
train_df = pd.read_csv("../train_data.csv")
test_df = pd.read_csv("../test_data_unlabeled.csv")

X_train = train_df["text"]
y_train = train_df["label"]
X_test_unlabeled = test_df["text"]

print(f"训练集样本数量: {len(X_train)}")
print(f"无标签测试集样本数量: {len(X_test_unlabeled)}")
print("-" * 20)

# 打印第一个训练样本和它的标签，感受一下数据
print("第一个训练样本内容:")
print(X_train[0])
print(f"\n第一个训练样本的标签: {y_train[0]}")
print("-" * 20)

# 打印第一个需要你预测的测试样本
print("第一个无标签测试样本内容:")
print(X_test_unlabeled[0])
print("\n" + "="*50)

# --- 在这里开始你的实验！---
# 现在，你可以使用 X_train, y_train, 和 X_test_unlabeled 这三个变量
# 来进行后续的特征提取、模型训练和预测了。

# 1. 创建TF-IDF向量化器
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)

# 2. 训练一个模型...
svm_model = SVC()
svm_model.fit(X_train_tfidf, y_train)

# 3. 对测试集进行预测...
X_test_tfidf = vectorizer.transform(X_test_unlabeled)
predictions = svm_model.predict(X_test_tfidf)

# 4. 保存你的预测结果，保存到上一层9-17文件夹
pd.DataFrame(predictions).to_csv('../predictions.csv', index=False, header=False)
print("预测完成, 结果已经保存到 predictions.csv")
