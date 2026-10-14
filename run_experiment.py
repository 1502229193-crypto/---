import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# 1. 直接读取当前文件夹下的数据
print("正在读取数据...")
train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data_unlabeled.csv")

X = train_df["text"]
y = train_df["target"]  # 注意这里修改成了 'target'
X_test_unlabeled = test_df["text"]

# 2. 划分训练集和验证集 (80% 训练, 20% 验证)，固定随机种子保证可复现
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"训练集样本数: {len(X_train)}, 验证集样本数: {len(X_val)}")

# 3. 特征工程：TF-IDF 向量化
print("正在进行 TF-IDF 特征提取...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test_unlabeled)

# 4. 训练与评估模型一：支持向量机 (SVM)
print("\n--- 训练支持向量机 (SVM) ---")
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train_tfidf, y_train)
svm_val_pred = svm_model.predict(X_val_tfidf)
svm_acc = accuracy_score(y_val, svm_val_pred)
print(f"SVM 验证集准确率: {svm_acc:.4f}")

# 5. 训练与评估模型二：逻辑回归 (Logistic Regression)
print("\n--- 训练逻辑回归 (Logistic Regression) ---")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_tfidf, y_train)
lr_val_pred = lr_model.predict(X_val_tfidf)
lr_acc = accuracy_score(y_val, lr_val_pred)
print(f"逻辑回归验证集准确率: {lr_acc:.4f}")

# 6. 选择表现更好的模型进行最终预测
best_model = svm_model if svm_acc >= lr_acc else lr_model
model_name = "SVM" if svm_acc >= lr_acc else "Logistic Regression"
print(f"\n表现更好的模型是: {model_name}，正在对测试集进行预测...")

predictions = best_model.predict(X_test_tfidf)

# 7. 保存预测结果到上一层或当前目录的 predictions.csv（根据实验要求存到上一层或当前目录，这里我们存到上一层9-17目录下以防万一，或者按你习惯）
# 如果助教要求在 9-17 根目录下看 predictions.csv，可以写 '../predictions.csv'
pd.DataFrame(predictions).to_csv('../predictions.csv', index=False, header=False)
print("预测完成！结果已成功保存到 ../predictions.csv 文件中。")