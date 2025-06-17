import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False   #这两行需要手动设置


class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        """
        逻辑回归分类器

        参数:
        learning_rate: 学习率
        max_iterations: 最大迭代次数
        tolerance: 收敛容忍度
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.cost_history = []
        self.converged = False

    def sigmoid(self, z):
        """
        Sigmoid激活函数
        σ(z) = 1 / (1 + e^(-z))
        """
        # 防止数值溢出
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        训练逻辑回归模型

        参数:
        X: 训练数据特征，shape为(n_samples, n_features)
        y: 训练标签，shape为(n_samples,)，标签为0或1
        """
        n_samples, n_features = X.shape

        # 初始化参数
        self.weights = np.zeros(n_features)
        self.bias = 0

        # 梯度下降优化
        for i in range(self.max_iterations):
            # 前向传播
            z = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(z)

            # 计算损失函数（交叉熵）
            cost = self.compute_cost(y, predictions)
            self.cost_history.append(cost)

            # 计算梯度
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            # 更新参数
            new_weights = self.weights - self.learning_rate * dw
            new_bias = self.bias - self.learning_rate * db

            # 检查收敛
            if (np.abs(new_weights - self.weights).max() < self.tolerance and
                    np.abs(new_bias - self.bias) < self.tolerance):
                self.converged = True
                print(f"收敛于第 {i + 1} 次迭代")
                break

            self.weights = new_weights
            self.bias = new_bias

        if not self.converged:
            print(f"达到最大迭代次数 {self.max_iterations}")

        return self

    def compute_cost(self, y_true, y_pred):
        """
        计算交叉熵损失
        J = -1/m * Σ[y*log(h) + (1-y)*log(1-h)]
        """
        # 防止log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        cost = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return cost

    def predict_proba(self, X):
        """
        预测概率

        参数:
        X: 待预测数据，shape为(n_samples, n_features)

        返回:
        预测概率，shape为(n_samples,)
        """
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        """
        预测类别

        参数:
        X: 待预测数据，shape为(n_samples, n_features)
        threshold: 分类阈值

        返回:
        预测的类别标签（0或1）
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    def plot_decision_boundary(self, X, y, title="逻辑回归决策边界"):
        """
        绘制决策边界（仅适用于2D数据）
        """
        if X.shape[1] != 2:
            print("只能绘制2维数据的决策边界")
            return

        plt.figure(figsize=(12, 8))

        # 创建网格
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # 预测网格点
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = self.predict_proba(grid_points)
        Z = Z.reshape(xx.shape)

        # 绘制等高线
        contour = plt.contour(xx, yy, Z, levels=[0.5], colors='green', linewidths=2)
        plt.clabel(contour, inline=True, fontsize=12, fmt='决策边界')

        # 绘制概率等高线
        plt.contourf(xx, yy, Z, levels=50, alpha=0.3, cmap='RdYlBu')
        colorbar = plt.colorbar()
        colorbar.set_label('预测概率')

        # 绘制数据点
        colors = ['blue', 'red']
        labels = ['类别 0', '类别 1']
        for i in [0, 1]:
            mask = (y == i)
            plt.scatter(X[mask, 0], X[mask, 1], c=colors[i],
                        label=labels[i], alpha=0.7, s=50, edgecolors='black')

        plt.xlabel('特征 1')
        plt.ylabel('特征 2')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_cost_history(self):
        """绘制损失函数变化曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.cost_history) + 1), self.cost_history, 'b-', linewidth=2)
        plt.xlabel('迭代次数')
        plt.ylabel('交叉熵损失')
        plt.title('训练过程中的损失函数变化')
        plt.grid(True, alpha=0.3)
        plt.show()







# 主程序
if __name__ == "__main__":

    # 生成二分类数据
    np.random.seed(42)
    X, y = make_classification(n_samples=300, n_features=2, n_redundant=0,
                               n_informative=2, n_clusters_per_class=1,
                               n_classes=2, random_state=19)

    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 训练逻辑回归
    lr = LogisticRegression(learning_rate=0.1, max_iterations=1000)
    lr.fit(X_scaled, y)

    # 预测
    y_pred = lr.predict(X_scaled)
    y_proba = lr.predict_proba(X_scaled)

    # 计算准确率
    accuracy = np.mean(y_pred == y)
    print(f"训练准确率: {accuracy:.2%}")



    # 可视化结果
    lr.plot_decision_boundary(X_scaled, y, "逻辑回归决策边界 - 合成数据")
    lr.plot_cost_history()

