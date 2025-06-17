import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False   #这两行需要手动设置

class GaussianDiscriminantAnalysis:
    def __init__(self):
        self.phi = None  # 先验概率 P(y=1)
        self.mu0 = None  # 类别0的均值
        self.mu1 = None  # 类别1的均值
        self.sigma = None  # 共享协方差矩阵

    def fit(self, X, y):
        """
        训练高斯判别分析模型

        参数:
        X: 特征矩阵 (n_samples, n_features)
        y: 标签向量 (n_samples,) - 应该是0或1
        """
        # m为样本个数,n为维度
        m, n = X.shape

        # 计算先验概率 φ = P(y=1)
        self.phi = np.sum(y) / m

        # 计算各类别的均值，类别为1的样本的均值和类别为0的样本的均值
        self.mu0 = np.mean(X[y == 0], axis=0)
        self.mu1 = np.mean(X[y == 1], axis=0)

        # 计算共享协方差矩阵
        sigma0 = np.zeros((n, n))
        sigma1 = np.zeros((n, n))

        # 计算类别0的协方差贡献
        X0_centered = X[y == 0] - self.mu0
        sigma0 = (X0_centered.T @ X0_centered)

        # 计算类别1的协方差贡献
        X1_centered = X[y == 1] - self.mu1
        sigma1 = (X1_centered.T @ X1_centered)

        # 共享协方差矩阵
        self.sigma = (sigma0 + sigma1) / m

    def predict_proba(self, X):
        """
        预测概率

        参数:
        X: 特征矩阵 (n_samples, n_features)

        返回:
        概率数组 (n_samples, 2) - [P(y=0|x), P(y=1|x)]
        """
        # 计算 P(x|y=0) 和 P(x|y=1) 一组符合多维高斯的概率值
        p_x_given_y0 = multivariate_normal.pdf(X, self.mu0, self.sigma)
        p_x_given_y1 = multivariate_normal.pdf(X, self.mu1, self.sigma)

        # 计算后验概率 P(y|x) 使用贝叶斯定理
        p_y0_given_x = (1 - self.phi) * p_x_given_y0
        p_y1_given_x = self.phi * p_x_given_y1

        # 归一化， 后面意思其实是就是这两个后验那个大就是那个类别
        total_prob = p_y0_given_x + p_y1_given_x
        p_y0_given_x /= total_prob
        p_y1_given_x /= total_prob

        return np.column_stack([p_y0_given_x, p_y1_given_x])

    def predict(self, X):
        """
        预测类别

        参数:
        X: 特征矩阵 (n_samples, n_features)

        返回:
        预测标签 (n_samples,)
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)

    def score(self, X, y):
        """
        计算准确率
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)




if __name__ == "__main__":
    # 生成示例数据
    X, y = make_classification(n_samples=500, n_features=2, n_redundant=0,
                               n_informative=2, n_clusters_per_class=1,
                               random_state=19)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=19)

    # 创建并训练模型
    gda = GaussianDiscriminantAnalysis()
    gda.fit(X_train, y_train)

    # 预测
    train_accuracy = gda.score(X_train, y_train)
    test_accuracy = gda.score(X_test, y_test)

    print(f"训练集准确率: {train_accuracy:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f}")
    print(f"先验概率 φ = P(y=1): {gda.phi:.4f}")
    print(f"类别0均值: {gda.mu0}")
    print(f"类别1均值: {gda.mu1}")

    # 可视化结果（仅适用于2D数据）
    if X.shape[1] == 2:
        plt.figure(figsize=(12, 5))

        # 绘制训练数据
        plt.subplot(1, 2, 1)
        plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
                    c='red', marker='o', alpha=0.7, label='类别 0')
        plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
                    c='blue', marker='s', alpha=0.7, label='类别 1')
        plt.scatter(gda.mu0[0], gda.mu0[1], c='red', marker='x', s=200,
                    linewidth=3, label='μ₀')
        plt.scatter(gda.mu1[0], gda.mu1[1], c='blue', marker='x', s=200,
                    linewidth=3, label='μ₁')
        plt.title('训练数据和类别中心')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 绘制决策边界
        plt.subplot(1, 2, 2)
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))

        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = gda.predict_proba(mesh_points)[:, 1]
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap='RdYlBu')
        plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
        plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1],
                    c='red', marker='o', alpha=0.8, label='测试 - 类别 0')
        plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1],
                    c='blue', marker='s', alpha=0.8, label='测试 - 类别 1')
        plt.colorbar(label='P(y=1|x)')
        plt.title('决策边界和测试数据')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()