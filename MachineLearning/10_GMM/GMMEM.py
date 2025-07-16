import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import warnings

warnings.filterwarnings('ignore')


class GaussianMixtureEM:
    """
    使用EM算法实现高斯混合模型
    """

    def __init__(self, n_components=2, max_iter=100, tol=1e-6, random_state=None):
        self.n_components = n_components  # 混合成分数量
        self.max_iter = max_iter  # 最大迭代次数
        self.tol = tol  # 收敛阈值
        self.random_state = random_state

    def _initialize_parameters(self, X):
        """初始化参数"""
        n_samples, n_features = X.shape

        # 设置随机种子
        if self.random_state:
            np.random.seed(self.random_state)

        # 初始化权重（混合系数）
        self.weights = np.ones(self.n_components) / self.n_components

        # 初始化均值（随机选择数据点作为初始均值）
        idx = np.random.choice(n_samples, self.n_components, replace=False)
        self.means = X[idx].copy()

        # 初始化协方差矩阵（单位矩阵）
        self.covariances = []
        for _ in range(self.n_components):
            self.covariances.append(np.eye(n_features))

        self.covariances = np.array(self.covariances)

    def _e_step(self, X):
        """E步：计算后验概率（责任度）"""
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))

        # 计算每个数据点在每个高斯分量下的概率密度，这里对应的就是求Q那一部分
        for k in range(self.n_components):
            try:
                responsibilities[:, k] = self.weights[k] * multivariate_normal.pdf(
                    X, self.means[k], self.covariances[k]
                )
            except np.linalg.LinAlgError:
                # 处理奇异协方差矩阵
                self.covariances[k] += np.eye(self.covariances[k].shape[0]) * 1e-6
                responsibilities[:, k] = self.weights[k] * multivariate_normal.pdf(
                    X, self.means[k], self.covariances[k]
                )

        # 归一化得到后验概率
        responsibilities_sum = responsibilities.sum(axis=1, keepdims=True)
        responsibilities_sum[responsibilities_sum == 0] = 1e-8  # 避免除零
        responsibilities /= responsibilities_sum

        return responsibilities

    def _m_step(self, X, responsibilities):
        """M步：更新参数"""
        n_samples, n_features = X.shape

        # 计算有效样本数
        Nk = responsibilities.sum(axis=0)
        Nk[Nk == 0] = 1e-8  # 避免除零

        # 更新权重
        self.weights = Nk / n_samples

        # 更新均值
        for k in range(self.n_components):
            self.means[k] = np.sum(responsibilities[:, k:k + 1] * X, axis=0) / Nk[k]

        # 更新协方差矩阵
        for k in range(self.n_components):
            diff = X - self.means[k]
            self.covariances[k] = np.dot(
                (responsibilities[:, k:k + 1] * diff).T, diff
            ) / Nk[k]

            # 添加正则化项防止奇异
            self.covariances[k] += np.eye(n_features) * 1e-6

    def _compute_log_likelihood(self, X):
        """计算对数似然"""
        n_samples = X.shape[0]
        log_likelihood = 0

        for i in range(n_samples):
            sample_likelihood = 0
            for k in range(self.n_components):
                try:
                    sample_likelihood += self.weights[k] * multivariate_normal.pdf(
                        X[i], self.means[k], self.covariances[k]
                    )
                except:
                    sample_likelihood += 1e-8

            log_likelihood += np.log(sample_likelihood + 1e-8)

        return log_likelihood

    def fit(self, X):
        """训练模型"""
        self._initialize_parameters(X)

        log_likelihoods = []

        for iteration in range(self.max_iter):
            # E步
            responsibilities = self._e_step(X)

            # M步
            self._m_step(X, responsibilities)

            # 计算对数似然
            log_likelihood = self._compute_log_likelihood(X)
            log_likelihoods.append(log_likelihood)

            # 检查收敛
            if iteration > 0:
                if abs(log_likelihoods[-1] - log_likelihoods[-2]) < self.tol:
                    print(f"收敛于第 {iteration + 1} 次迭代")
                    break

        self.log_likelihoods_ = log_likelihoods
        return self

    def predict(self, X):
        """预测类别"""
        responsibilities = self._e_step(X)
        return np.argmax(responsibilities, axis=1)

    def predict_proba(self, X):
        """预测概率"""
        return self._e_step(X)


# 生成示例数据
def generate_sample_data():
    """生成二维高斯混合数据"""
    np.random.seed(42)

    # 第一个高斯分量
    mean1 = [2, 3]
    cov1 = [[1, 0.5], [0.5, 1]]
    data1 = np.random.multivariate_normal(mean1, cov1, 150)

    # 第二个高斯分量
    mean2 = [6, 7]
    cov2 = [[1.5, -0.3], [-0.3, 1.2]]
    data2 = np.random.multivariate_normal(mean2, cov2, 100)

    # 第三个高斯分量
    mean3 = [4, 1]
    cov3 = [[0.8, 0.2], [0.2, 0.8]]
    data3 = np.random.multivariate_normal(mean3, cov3, 120)

    # 合并数据
    X = np.vstack([data1, data2, data3])
    true_labels = np.hstack([np.zeros(150), np.ones(100), np.full(120, 2)])

    return X, true_labels


# 可视化函数
def plot_results(X, true_labels, predicted_labels, gmm, title):
    """绘制结果"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 绘制真实分布
    colors = ['red', 'blue', 'green']
    for i in range(3):
        mask = true_labels == i
        ax1.scatter(X[mask, 0], X[mask, 1], c=colors[i], alpha=0.6, s=30)

    # 绘制真实的高斯分量中心
    true_means = [[2, 3], [6, 7], [4, 1]]
    for i, mean in enumerate(true_means):
        ax1.plot(mean[0], mean[1], 'k*', markersize=15, markeredgecolor='white')

    ax1.set_title('真实分布')
    ax1.set_xlabel('X1')
    ax1.set_ylabel('X2')
    ax1.grid(True, alpha=0.3)

    # 绘制预测结果
    for i in range(gmm.n_components):
        mask = predicted_labels == i
        ax2.scatter(X[mask, 0], X[mask, 1], c=colors[i], alpha=0.6, s=30)

    # 绘制学习到的高斯分量中心
    for i, mean in enumerate(gmm.means):
        ax2.plot(mean[0], mean[1], 'k*', markersize=15, markeredgecolor='white')

    ax2.set_title(f'EM算法预测结果 ({title})')
    ax2.set_xlabel('X1')
    ax2.set_ylabel('X2')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_convergence(gmm):
    """绘制收敛过程"""
    plt.figure(figsize=(10, 6))
    plt.plot(gmm.log_likelihoods_, 'b-', linewidth=2, marker='o')
    plt.title('EM算法收敛过程')
    plt.xlabel('迭代次数')
    plt.ylabel('对数似然')
    plt.grid(True, alpha=0.3)
    plt.show()


# 主程序
if __name__ == "__main__":
    print("EM算法 - 高斯混合模型示例")
    print("=" * 50)

    # 生成数据
    X, true_labels = generate_sample_data()
    print(f"生成数据: {X.shape[0]} 个样本, {X.shape[1]} 个特征")

    # 训练EM模型
    print("\n训练EM算法...")
    # 在实际应用时，高斯分量数，也就是k是要提前选定的，选定的好坏很容易影响模型的好坏，可以用AIC 或 BIC
    gmm = GaussianMixtureEM(n_components=3, max_iter=100, random_state=42)
    gmm.fit(X)

    # 预测
    predicted_labels = gmm.predict(X)
    probabilities = gmm.predict_proba(X)

    # 输出结果
    print(f"\n学习到的参数:")
    print(f"权重: {gmm.weights}")
    print(f"均值:")
    for i, mean in enumerate(gmm.means):
        print(f"  分量 {i + 1}: {mean}")

    print(f"\n协方差矩阵:")
    for i, cov in enumerate(gmm.covariances):
        print(f"  分量 {i + 1}:")
        print(f"    {cov}")

    # 计算准确率（需要注意标签可能不匹配）
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import confusion_matrix

    # 使用匈牙利算法匹配标签
    cm = confusion_matrix(true_labels, predicted_labels)
    row_ind, col_ind = linear_sum_assignment(-cm)

    # 重新映射预测标签
    label_mapping = dict(zip(col_ind, row_ind))
    mapped_predictions = np.array([label_mapping[label] for label in predicted_labels])

    accuracy = np.mean(true_labels == mapped_predictions)
    print(f"\n聚类准确率: {accuracy:.3f}")

    # 可视化结果
    plot_results(X, true_labels, predicted_labels, gmm, f"准确率: {accuracy:.3f}")
    plot_convergence(gmm)

    print(f"\n最终对数似然: {gmm.log_likelihoods_[-1]:.2f}")
    print(f"总迭代次数: {len(gmm.log_likelihoods_)}")