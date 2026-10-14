import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# 加载与切分数据
train_df = pd.read_csv("train_data.csv")
X, y = train_df["text"], train_df["target"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征提取
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

# 测试不同的 C 参数
c_values = [0.1, 1.0, 10.0]
print("=== SVM 超参数 C 调优实验 ===")
for c in c_values:
    model = SVC(kernel='linear', C=c, random_state=42)
    model.fit(X_train_tfidf, y_train)
    acc = accuracy_score(y_val, model.predict(X_val_tfidf))
    print(f"参数 C = {c:<4} -> 验证集准确率: {acc:.4f}")