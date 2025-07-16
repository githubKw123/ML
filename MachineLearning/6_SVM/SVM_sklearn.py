import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_blobs

# 生成线性可分数据用于硬间隔SVM
X_hard, y_hard = make_blobs(n_samples=100, centers=2, cluster_std=1.0, random_state=42)
# 生成带噪声数据用于软间隔SVM
X_soft, y_soft = make_blobs(n_samples=100, centers=2, cluster_std=2.5, random_state=42)

# 训练硬间隔SVM (C设置非常大，接近硬间隔)
svm_hard = SVC(kernel='linear', C=1e10)
svm_hard.fit(X_hard, y_hard)

# 训练软间隔SVM (C较小，允许一定误分类)
svm_soft = SVC(kernel='linear', C=1.0)
svm_soft.fit(X_soft, y_soft)


# 可视化函数
def plot_svm(X, y, model, title, ax):
    h = .02  # 网格步长
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)

    # 绘制数据点
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    # 绘制支持向量
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
               s=100, facecolors='none', edgecolors='k', label='Support Vectors')

    # 绘制决策边界和间隔
    w = model.coef_[0]
    b = model.intercept_[0]
    x0 = np.linspace(x_min, x_max, 100)
    decision_boundary = -(w[0] * x0 + b) / w[1]
    margin = 1 / np.sqrt(np.sum(model.coef_ ** 2))
    ax.plot(x0, decision_boundary, 'k-', label='Decision Boundary')
    ax.plot(x0, decision_boundary + margin, 'k--')
    ax.plot(x0, decision_boundary - margin, 'k--')

    ax.set_title(title)
    ax.legend()


# 绘制结果
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
plot_svm(X_hard, y_hard, svm_hard, 'Hard Margin SVM', ax1)
plot_svm(X_soft, y_soft, svm_soft, 'Soft Margin SVM', ax2)
plt.tight_layout()
plt.show()