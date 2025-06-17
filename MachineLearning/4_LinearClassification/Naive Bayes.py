import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False   #这两行需要手动设置

class NaiveBayesClassifier:
    """
    朴素贝叶斯分类器实现
    基于高斯分布假设的连续特征朴素贝叶斯分类器
    """

    def __init__(self):
        self.class_priors = {}  # 类别先验概率
        self.feature_params = {}  # 各特征在不同类别下的均值和方差
        self.classes = None

    def fit(self, X, y):
        """
        训练朴素贝叶斯分类器

        参数:
        X: 特征矩阵 (n_samples, n_features)
        y: 标签向量 (n_samples,)
        """
        self.classes = np.unique(y)
        n_samples, n_features = X.shape

        # 计算类别先验概率 P(y)
        for c in self.classes:
            self.class_priors[c] = np.sum(y == c) / n_samples

        # 计算每个特征在各个类别下的参数（均值和方差）
        self.feature_params = {}
        for c in self.classes:
            self.feature_params[c] = {}
            class_samples = X[y == c]

            # 对每个特征，计算在当前类别下的均值和方差
            for feature_idx in range(n_features):
                feature_values = class_samples[:, feature_idx]
                self.feature_params[c][feature_idx] = {
                    'mean': np.mean(feature_values),
                    'var': np.var(feature_values) + 1e-9  # 添加小值避免除零
                }

    def _gaussian_probability(self, x, mean, var):
        """
        计算高斯分布的概率密度
        """
        return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-((x - mean) ** 2) / (2 * var))

    def _predict_sample(self, x):
        """
        预测单个样本
        """
        posteriors = {}

        for c in self.classes:
            # 计算后验概率 P(y|X) ∝ P(X|y) * P(y)
            posterior = np.log(self.class_priors[c])  # 使用log避免数值下溢

            # 朴素贝叶斯假设：特征之间相互独立
            # P(X|y) = P(x1|y) * P(x2|y) * ... * P(xn|y)
            for feature_idx, feature_value in enumerate(x):
                mean = self.feature_params[c][feature_idx]['mean']
                var = self.feature_params[c][feature_idx]['var']

                # 计算 P(xi|y) 并取对数
                likelihood = self._gaussian_probability(feature_value, mean, var)
                posterior += np.log(likelihood + 1e-10)  # 避免log(0)

            posteriors[c] = posterior

        # 返回后验概率最大的类别
        return max(posteriors, key=posteriors.get)

    def predict(self, X):
        """
        预测多个样本
        """
        predictions = []
        for x in X:
            predictions.append(self._predict_sample(x))
        return np.array(predictions)

    def predict_proba(self, X):
        """
        预测类别概率
        """
        probabilities = []
        for x in X:
            posteriors = {}

            for c in self.classes:
                posterior = np.log(self.class_priors[c])

                for feature_idx, feature_value in enumerate(x):
                    mean = self.feature_params[c][feature_idx]['mean']
                    var = self.feature_params[c][feature_idx]['var']
                    likelihood = self._gaussian_probability(feature_value, mean, var)
                    posterior += np.log(likelihood + 1e-10)

                posteriors[c] = posterior

            # 转换回概率（归一化）
            max_posterior = max(posteriors.values())
            exp_posteriors = {c: np.exp(p - max_posterior) for c, p in posteriors.items()}
            total = sum(exp_posteriors.values())
            probabilities.append([exp_posteriors[c] / total for c in self.classes])

        return np.array(probabilities)


def generate_sample_data():
    """
    生成示例数据用于演示
    """
    # 生成二分类数据集
    X, y = make_classification(
        n_samples=1000,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        random_state=42
    )
    return X, y






if __name__ == "__main__":
    print("=== 朴素贝叶斯二分类器演示 ===\n")

    X, y = make_classification(
        n_samples=1000,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        random_state=42
    )

    #划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


    # 3. 训练模型

    nb_classifier = NaiveBayesClassifier()
    nb_classifier.fit(X_train, y_train)


    y_pred = nb_classifier.predict(X_test)
    y_pred_proba = nb_classifier.predict_proba(X_test)


    accuracy = accuracy_score(y_test, y_pred)
    print(f"准确率: {accuracy:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))

    # 6. 可视化结果
    plt.figure(figsize=(15, 5))

    # 子图1: 原始数据分布
    plt.subplot(1, 3, 1)
    colors = ['red', 'blue']
    for i, c in enumerate(np.unique(y)):
        plt.scatter(X[y == c, 0], X[y == c, 1],
                    c=colors[i], alpha=0.6, label=f'类别 {c}')
    plt.title('原始数据分布')
    plt.xlabel('特征 1')
    plt.ylabel('特征 2')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图2: 预测结果
    plt.subplot(1, 3, 2)
    for i, c in enumerate(np.unique(y_test)):
        mask = y_test == c
        correct_mask = mask & (y_pred == c)
        wrong_mask = mask & (y_pred != c)

        plt.scatter(X_test[correct_mask, 0], X_test[correct_mask, 1],
                    c=colors[i], alpha=0.8, marker='o', s=50,
                    label=f'类别 {c} (正确)')
        plt.scatter(X_test[wrong_mask, 0], X_test[wrong_mask, 1],
                    c=colors[i], alpha=0.8, marker='x', s=100,
                    label=f'类别 {c} (错误)')

    plt.title(f'预测结果 (准确率: {accuracy:.3f})')
    plt.xlabel('特征 1')
    plt.ylabel('特征 2')
    plt.legend()
    plt.grid(True, alpha=0.3)


