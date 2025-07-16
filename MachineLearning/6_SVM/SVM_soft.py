import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import cvxpy as cp


class SoftMarginSVM:
    def __init__(self, C=1.0, kernel='linear', gamma=1.0):
        """
        软间隔SVM实现

        参数:
        C: 正则化参数，控制间隔和误分类的权衡
        kernel: 核函数类型 ('linear', 'rbf')
        gamma: RBF核的参数
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.alpha = None
        self.support_vectors = None
        self.support_labels = None
        self.b = None

    def _kernel_function(self, X1, X2):
        """计算核函数"""
        if self.kernel == 'linear':
            return np.dot(X1, X2.T)
        elif self.kernel == 'rbf':
            # RBF核: K(x,y) = exp(-gamma * ||x-y||^2)
            sqdist = np.sum(X1 ** 2, axis=1).reshape(-1, 1) + \
                     np.sum(X2 ** 2, axis=1) - 2 * np.dot(X1, X2.T)
            return np.exp(-self.gamma * sqdist)

    def fit(self, X, y):
        """训练SVM模型"""
        n_samples, n_features = X.shape

        # 计算核矩阵
        K = self._kernel_function(X, X)

        # 设置二次规划问题
        # 目标函数: min 1/2 * alpha^T * Q * alpha - e^T * alpha
        # 约束: 0 <= alpha_i <= C, sum(alpha_i * y_i) = 0

        # 构建Q矩阵
        Q = np.outer(y, y) * K

        # 使用CVXPY求解二次规划
        alpha = cp.Variable(n_samples)

        # 目标函数
        objective = cp.Minimize(0.5 * cp.quad_form(alpha, Q) - cp.sum(alpha))

        # 约束条件
        constraints = [
            alpha >= 0,
            alpha <= self.C,
            cp.sum(cp.multiply(alpha, y)) == 0
        ]

        # 求解
        prob = cp.Problem(objective, constraints)
        prob.solve()

        # 获取拉格朗日乘数
        self.alpha = alpha.value

        # 找到支持向量（alpha > 0的点）
        sv_idx = np.where(self.alpha > 1e-5)[0]
        self.support_vectors = X[sv_idx]
        self.support_labels = y[sv_idx]
        self.alpha_sv = self.alpha[sv_idx]

        # 计算偏置项b
        # 使用边界支持向量（0 < alpha < C）计算b
        margin_sv_idx = np.where((self.alpha > 1e-5) & (self.alpha < self.C - 1e-5))[0]

        if len(margin_sv_idx) > 0:
            # 对于边界支持向量，约束条件严格成立
            K_sv = self._kernel_function(X[margin_sv_idx], self.support_vectors)
            self.b = np.mean(
                self.support_labels[margin_sv_idx] -
                np.sum(self.alpha_sv * self.support_labels * K_sv, axis=1)
            )
        else:
            # 如果没有边界支持向量，使用所有支持向量的平均值
            K_sv = self._kernel_function(sv_idx, self.support_vectors)
            self.b = np.mean(
                self.support_labels -
                np.sum(self.alpha_sv * self.support_labels * K_sv, axis=1)
            )

    def predict(self, X):
        """预测新样本"""
        if self.support_vectors is None:
            raise ValueError("模型未训练，请先调用fit方法")

        # 计算决策函数值
        K = self._kernel_function(X, self.support_vectors)
        decision = np.sum(self.alpha_sv * self.support_labels * K, axis=1) + self.b

        # 返回预测标签
        return np.sign(decision)

    def decision_function(self, X):
        """计算决策函数值"""
        if self.support_vectors is None:
            raise ValueError("模型未训练，请先调用fit方法")

        K = self._kernel_function(X, self.support_vectors)
        return np.sum(self.alpha_sv * self.support_labels * K, axis=1) + self.b


def plot_svm_decision_boundary(svm, X, y, title="SVM决策边界"):
    """绘制SVM决策边界"""
    plt.figure(figsize=(10, 8))

    # 创建网格
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # 预测网格点
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = svm.decision_function(grid_points)
    Z = Z.reshape(xx.shape)

    # 绘制决策边界和间隔
    plt.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['red', 'black', 'red'],
                linestyles=['--', '-', '--'], linewidths=[2, 3, 2])

    # 绘制数据点
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', s=50, alpha=0.8)

    # 高亮支持向量
    if hasattr(svm, 'support_vectors'):
        plt.scatter(svm.support_vectors[:, 0], svm.support_vectors[:, 1],
                    s=200, facecolors='none', edgecolors='black', linewidths=2)

    plt.title(title)
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.colorbar(scatter)
    plt.grid(True, alpha=0.3)
    plt.show()


# 演示示例
def demo_soft_margin_svm():
    """演示软间隔SVM"""
    # 生成示例数据
    np.random.seed(42)
    X, y = make_blobs(n_samples=100, centers=2, n_features=2,
                      center_box=(-2, 2), cluster_std=1.2, random_state=42)
    y = np.where(y == 0, -1, 1)  # 转换标签为-1和1

    # 添加一些噪声点使数据不完全线性可分
    noise_X = np.random.randn(10, 2) * 0.5
    noise_y = np.random.choice([-1, 1], 10)
    X = np.vstack([X, noise_X])
    y = np.hstack([y, noise_y])

    print("=== 软间隔SVM演示 ===")
    print(f"数据集大小: {X.shape[0]} 样本, {X.shape[1]} 特征")

    # 测试不同的C值
    C_values = [0.1, 1.0, 10.0]

    for C in C_values:
        print(f"\n训练SVM (C={C})...")

        # 训练模型
        svm = SoftMarginSVM(C=C, kernel='linear')
        svm.fit(X, y)

        # 预测
        y_pred = svm.predict(X)
        accuracy = np.mean(y_pred == y)

        print(f"支持向量数量: {len(svm.support_vectors)}")
        print(f"训练准确率: {accuracy:.3f}")

        # 绘制决策边界
        plot_svm_decision_boundary(svm, X, y, f"软间隔SVM (C={C})")

    # 演示RBF核
    print(f"\n训练RBF核SVM...")
    svm_rbf = SoftMarginSVM(C=1.0, kernel='rbf', gamma=0.5)
    svm_rbf.fit(X, y)

    y_pred_rbf = svm_rbf.predict(X)
    accuracy_rbf = np.mean(y_pred_rbf == y)

    print(f"支持向量数量: {len(svm_rbf.support_vectors)}")
    print(f"训练准确率: {accuracy_rbf:.3f}")

    plot_svm_decision_boundary(svm_rbf, X, y, "RBF核SVM")


if __name__ == "__main__":
    demo_soft_margin_svm()