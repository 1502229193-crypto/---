import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

print("1. 正在加载数据...")
train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data_unlabeled.csv")

X = train_df["text"]
y = train_df["target"]
X_test_unlabeled = test_df["text"]

# 严格划分训练集与验证集 (80% 训练, 20% 验证)[cite: 26]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"-> 训练集样本数: {len(X_train)} | 验证集样本数: {len(X_val)}")

# 2. 文本特征工程与预处理（加入去除停用词作为数据清洗探索）[cite: 33]
print("2. 正在进行带停用词过滤的 TF-IDF 特征提取...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test_unlabeled)

results = {}

# 模型一：朴素贝叶斯 (Baseline)[cite: 35]
print("\n3. 正在训练朴素贝叶斯模型 (Naive Bayes)...")
start_time = time.time()
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
nb_acc = accuracy_score(y_val, nb_model.predict(X_val_tfidf))
results['Naive Bayes'] = (nb_acc, time.time() - start_time)
print(f">>> 朴素贝叶斯 验证集准确率: {nb_acc:.4f} | 耗时: {results['Naive Bayes'][1]:.2f}秒")

# 模型二：支持向量机 (SVM)[cite: 36]
print("\n4. 正在训练支持向量机 (SVM)...")
start_time = time.time()
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train_tfidf, y_train)
svm_acc = accuracy_score(y_val, svm_model.predict(X_val_tfidf))
results['SVM'] = (svm_acc, time.time() - start_time)
print(f">>> SVM 验证集准确率: {svm_acc:.4f} | 耗时: {results['SVM'][1]:.2f}秒")

# 模型三：多层感知机 (MLP) —— 这就是同学们说训练很久的神经网络模型！[cite: 38, 40]
print("\n5. 正在训练多层感知机 (MLP) 神经网络...")
start_time = time.time()
# 使用两层隐藏层 (100, 50)，增加训练深度，观察训练时间
mlp_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42)
mlp_model.fit(X_train_tfidf, y_train)
mlp_acc = accuracy_score(y_val, mlp_model.predict(X_val_tfidf))
results['MLP'] = (mlp_acc, time.time() - start_time)
print(f">>> MLP 验证集准确率: {mlp_acc:.4f} | 耗时: {results['MLP'][1]:.2f}秒")

# 选出准确率最高的模型生成最终预测
best_name = max(results, key=lambda x: results[x][0])
print(f"\n6. 综合对比完成！表现最好的模型是: {best_name} (准确率: {results[best_name][0]:.4f})")

# 用最优模型预测测试集并保存
best_model = {'Naive Bayes': nb_model, 'SVM': svm_model, 'MLP': mlp_model}[best_name]
predictions = best_model.predict(X_test_tfidf)
pd.DataFrame(predictions).to_csv('../predictions.csv', index=False, header=False)
print(">>> 最终预测结果已成功保存到 ../predictions.csv 文件中！")