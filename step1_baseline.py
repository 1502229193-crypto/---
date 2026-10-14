import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

print("1. 正在加载数据...")
train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data_unlabeled.csv")

X = train_df["text"]
y = train_df["target"]
X_test_unlabeled = test_df["text"]

# 严格按照实验规范划分训练集与验证集 (80% 训练, 20% 验证)，固定随机种子[cite: 3, 6]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"-> 训练集样本数: {len(X_train)} | 验证集样本数: {len(X_val)}")

print("2. 正在进行 TF-IDF 特征提取...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test_unlabeled)

# 模型一：支持向量机 (SVM)
print("\n3. 正在训练支持向量机 (SVM)...")
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train_tfidf, y_train)
svm_pred = svm_model.predict(X_val_tfidf)
svm_acc = accuracy_score(y_val, svm_pred)
print(f">>> SVM 验证集准确率: {svm_acc:.4f}")

# 模型二：逻辑回归 (Logistic Regression)
print("\n4. 正在训练逻辑回归 (Logistic Regression)...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_tfidf, y_train)
lr_pred = lr_model.predict(X_val_tfidf)
lr_acc = accuracy_score(y_val, lr_pred)
print(f">>> 逻辑回归验证集准确率: {lr_acc:.4f}")

# 自动选择最优模型并生成测试集预测结果[cite: 16]
best_model = svm_model if svm_acc >= lr_acc else lr_model
print(f"\n5. 表现更好的模型是: {'SVM' if svm_acc >= lr_acc else 'Logistic Regression'}，正在生成测试集预测...")
predictions = best_model.predict(X_test_tfidf)
pd.DataFrame(predictions).to_csv('../predictions.csv', index=False, header=False)
print(">>> 预测完成！结果已成功保存到 ../predictions.csv 文件中。")