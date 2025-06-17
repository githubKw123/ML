import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False   #这两行需要手动设置
def generate_multivariate_gaussian(mean, cov, n_samples=1000):
    """
    生成多维高斯分布样本

    参数:
    mean: 均值向量 (numpy array)
    cov: 协方差矩阵 (numpy array)
    n_samples: 样本数量

    返回:
    samples: 生成的样本 (n_samples x dimensions)
    """
    return np.random.multivariate_normal(mean, cov, n_samples)


def plot_2d_gaussian(samples, mean, title="2D Gaussian Distribution"):
    """绘制2D高斯分布"""
    plt.figure(figsize=(10, 8))
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.6, s=20)
    plt.scatter(mean[0], mean[1], color='red', s=100, marker='x', label='Mean')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()

def plot_3d_gaussian(samples, mean, title="3D Gaussian Distribution"):
    """绘制3D高斯分布"""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], alpha=0.6, s=20)
    ax.scatter(mean[0], mean[1], mean[2], color='red', s=100, marker='x', label='Mean')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')
    ax.set_title(title)
    ax.legend()
    plt.show()


def generate_random_covariance_matrix(dim, condition_number=10):
    """
    生成随机的正定协方差矩阵

    参数:
    dim: 维度
    condition_number: 条件数，控制矩阵的病态程度
    """
    # 生成随机特征值
    eigenvals = np.random.uniform(1, condition_number, dim)
    # 生成随机正交矩阵
    Q, _ = np.linalg.qr(np.random.randn(dim, dim))
    # 构造协方差矩阵
    cov = Q @ np.diag(eigenvals) @ Q.T
    return cov

if __name__ == "__main__":
    # 随机生成四维
    dim = 4
    random_mean = np.zeros(dim)
    random_cov = generate_random_covariance_matrix(dim)
    random_samples = generate_multivariate_gaussian(random_mean, random_cov, 500)

    # 绘制二维
    mean_2d = np.array([2, 3])
    cov_2d = np.array([[1, 0.5], [0.5, 2]])
    samples_2d = generate_multivariate_gaussian(mean_2d, cov_2d, 1000)
    plot_2d_gaussian(samples_2d, mean_2d)

    # 绘制三维
    mean_3d = np.array([1, -1, 2])
    cov_3d = np.array([[2, 0.3, 0.1],
                       [0.3, 1.5, -0.2],
                       [0.1, -0.2, 1]])
    samples_3d = generate_multivariate_gaussian(mean_3d, cov_3d, 1000)
    plot_3d_gaussian(samples_3d, mean_3d)