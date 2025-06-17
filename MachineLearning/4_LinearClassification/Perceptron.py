import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False   #这两行需要手动设置

class Perceptron:
    def __init__(self, learning_rate=0.1, max_iterations=1000):
        """
        初始化感知机

        参数:
        learning_rate: 学习率
        max_iterations: 最大迭代次数
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
        self.errors = []

    def fit(self, X, y):
        """
        训练感知机

        参数:
        X: 训练数据特征，shape为(n_samples, n_features)
        y: 训练数据标签，shape为(n_samples,)，标签为1或-1
        """
        n_samples, n_features = X.shape

        # 初始化权重和偏置
        self.weights = np.zeros(n_features)
        self.bias = 0

        # 训练过程
        for i in range(self.max_iterations):
            errors = 0

            for j in range(n_samples):
                # 计算预测值
                prediction = self.predict_single(X[j])

                # 如果预测错误，SDG更新权重和偏置
                if y[j] * prediction <= 0:
                    self.weights += self.learning_rate * y[j] * X[j]
                    self.bias += self.learning_rate * y[j]
                    errors += 1

            self.errors.append(errors)

            # 如果没有错误，说明训练完成
            if errors == 0:
                print(f"训练完成，迭代次数: {i + 1}")
                break
        else:
            print(f"达到最大迭代次数: {self.max_iterations}")

    def predict_single(self, x):
        """预测单个样本"""
        return np.dot(x, self.weights) + self.bias

    def predict(self, X):
        """预测多个样本"""
        return np.array([1 if self.predict_single(x) > 0 else -1 for x in X])

    def plot_decision_boundary(self, X, y):
        """绘制决策边界（仅适用于2D数据）"""
        if X.shape[1] != 2:
            print("只能绘制2维数据的决策边界")
            return

        plt.figure(figsize=(10, 8))

        # 绘制数据点
        positive_points = X[y == 1]
        negative_points = X[y == -1]

        plt.scatter(positive_points[:, 0], positive_points[:, 1],
                    c='red', marker='o', label='正类 (+1)', s=50)
        plt.scatter(negative_points[:, 0], negative_points[:, 1],
                    c='blue', marker='s', label='负类 (-1)', s=50)

        # 绘制决策边界
        if self.weights is not None:
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

            # 决策边界方程: w1*x1 + w2*x2 + b = 0
            # 即: x2 = -(w1*x1 + b) / w2
            if abs(self.weights[1]) > 1e-10:  # 避免除零
                x_boundary = np.linspace(x_min, x_max, 100)
                y_boundary = -(self.weights[0] * x_boundary + self.bias) / self.weights[1]
                plt.plot(x_boundary, y_boundary, 'g-', linewidth=2, label='决策边界')

            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)

        plt.xlabel('特征 1')
        plt.ylabel('特征 2')
        plt.title('感知机决策边界')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_errors(self):
        """绘制训练过程中的错误数量"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.errors) + 1), self.errors, 'b-o', linewidth=2, markersize=6)
        plt.xlabel('迭代次数')
        plt.ylabel('错误分类数量')
        plt.title('训练过程中的错误数量变化')
        plt.grid(True, alpha=0.3)
        plt.show()


# 示例使用
if __name__ == "__main__":
    # 生成线性可分的示例数据
    np.random.seed(42)

    # 正类数据
    positive_samples = np.random.randn(20, 2) + [2, 2]
    positive_labels = np.ones(20)

    # 负类数据
    negative_samples = np.random.randn(20, 2) + [-2, -2]
    negative_labels = -np.ones(20)

    # 合并数据
    X = np.vstack([positive_samples, negative_samples])
    y = np.hstack([positive_labels, negative_labels])

    # 创建和训练感知机
    perceptron = Perceptron(learning_rate=0.1, max_iterations=1000)
    perceptron.fit(X, y)

    # 打印结果
    print(f"最终权重: {perceptron.weights}")
    print(f"最终偏置: {perceptron.bias}")

    # 测试预测
    predictions = perceptron.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"训练准确率: {accuracy:.2%}")

    # 绘制结果
    perceptron.plot_decision_boundary(X, y)

    # 测试新数据点
    test_points = np.array([[1, 1], [-1, -1], [3, 3], [-3, -3]])
    test_predictions = perceptron.predict(test_points)

    print("\n测试新数据点:")
    for i, (point, pred) in enumerate(zip(test_points, test_predictions)):
        print(f"点 {point}: 预测类别 {pred}")