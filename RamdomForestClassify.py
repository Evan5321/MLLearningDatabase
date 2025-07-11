import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from matplotlib.colors import ListedColormap
# -*- coding: utf-8 -*-

# 加载数据集
iris = load_iris()
X = iris.data[:, :2]  # 只使用前两个特征以便于可视化
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建随机森林分类器实例
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练模型
rf_clf.fit(X_train, y_train)

# 预测与评估
y_pred = rf_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")  # 原"模型准确率"改为"Model Accuracy"

# ========================
# 1. Feature Importance Visualization
# ========================
plt.figure(figsize=(6, 4))
plt.barh(iris.feature_names[:2], rf_clf.feature_importances_[:2])
plt.title("Feature Importance")  # 原"特征重要性"改为"Feature Importance"
plt.xlabel("Importance")  # 原"重要性"改为"Importance"
plt.ylabel("Features")  # 原"特征"改为"Features"
plt.show()

# ========================
# 2. Decision Boundary Visualization
# ========================
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    cmap_light = ListedColormap(['#FFAAAA','#AAFFAA','#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000','#00FF00','#0000FF'])

    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=50)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Random Forest Classifier Decision Boundary")  # 原"随机森林分类器决策边界"改为"Random Forest Classifier Decision Boundary"
    plt.xlabel("Sepal Length (cm)")  # 原"花萼长度 (cm)"改为"Sepal Length (cm)"
    plt.ylabel("Sepal Width (cm)")  # 原"花萼宽度 (cm)"改为"Sepal Width (cm)"
    plt.show()

plot_decision_boundary(rf_clf, X, y)

# ========================
# 3. Confusion Matrix Visualization
# ========================
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")  # 原"混淆矩阵"改为"Confusion Matrix"
plt.show()