import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import time

# 生成示例数据
np.random.seed(42)
X, y = make_classification(
    n_samples=500,
    n_features=6,
    n_informative=4,
    n_redundant=2,
    n_clusters_per_class=1,
    random_state=42
)


# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==================== 方法1: 传统特征值分解方法 ====================
def pca_with_evd(X, n_components=None):
    """使用特征值分解的传统PCA方法"""
    # EVD步骤1: 计算协方差矩阵
    n_samples = X.shape[0]
    cov_matrix = np.dot(X.T, X) / (n_samples - 1)

   # "EVD步骤2: 特征值分解"
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # 降序排列 特征值越大，表示其对应的投影方差越大，用其表征样本空间越清晰，所以取特征值最大的前几个值作为投影空间
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # 这里的n_components就是降维后的维数
    if n_components is None:
        n_components = X.shape[1]

    # 这里的components就是提取维数个的特征向量（列）
    components = eigenvectors[:, :n_components]
    explained_variance = eigenvalues[:n_components]
    explained_variance_ratio = explained_variance / np.sum(eigenvalues)

    # 原向量乘以特征向量就是降维后的数据
    X_transformed = np.dot(X, components)

    return X_transformed, components, explained_variance_ratio, explained_variance


# ==================== 方法2: 手动SVD方法 ====================
def pca_with_manual_svd(X, n_components=None):
    """
    使用SVD手动实现PCA

    SVD分解：X = U × Σ × V^T
    其中：
    - U: 左奇异向量矩阵 (n_samples, n_samples)
    - Σ: 奇异值对角矩阵 (min(n_samples, n_features))
    - V^T: 右奇异向量矩阵转置 (n_features, n_features)

    对于PCA：
    - 主成分 = V（右奇异向量）
    - 特征值 = (奇异值²) / (n_samples - 1)
    """


    # 直接对数据矩阵进行SVD分解
    U, singular_values, Vt = np.linalg.svd(X, full_matrices=False)



    # 计算特征值：eigenvalues = singular_values² / (n_samples - 1)
    n_samples = X.shape[0]
    eigenvalues = (singular_values ** 2) / (n_samples - 1)

    # 主成分矩阵是V^T的转置，即V
    components_matrix = Vt.T  # shape: (n_features, n_components)


    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)

    # 选择主成分并降维
    if n_components is None:
        n_components = min(X.shape)

    # 选择前n_components个主成分
    selected_components = components_matrix[:, :n_components]
    selected_eigenvalues = eigenvalues[:n_components]
    selected_var_ratio = explained_variance_ratio[:n_components]

    # 降维变换
    X_transformed = np.dot(X, selected_components)


    return X_transformed, selected_components, selected_var_ratio, selected_eigenvalues, U, singular_values, Vt


n_components = 3

# 方法1: EVD
start_time = time.time()
X_evd, comp_evd, var_ratio_evd, eigenvals_evd = pca_with_evd(X_scaled, n_components)
evd_time = time.time() - start_time

# 方法2: 手动SVD
start_time = time.time()
X_manual_svd, comp_manual_svd, var_ratio_manual_svd, eigenvals_manual_svd, U, S, Vt = pca_with_manual_svd(X_scaled,
                                                                                                          n_components)
manual_svd_time = time.time() - start_time

# 方法3: sklearn
start_time = time.time()
pca_sklearn = PCA(n_components=n_components, svd_solver='full')
X_sklearn_svd = pca_sklearn.fit_transform(X_scaled)
sklearn_time = time.time() - start_time

# ==================== 结果对比 ====================
print("\n【三种方法结果对比】")
print("=" * 40)
print(f"原数据   {X_scaled}")
print("1. 方差贡献率对比:")
print(f"EVD方法:      {var_ratio_evd}")
print(f"手动SVD:      {var_ratio_manual_svd}")
print(f"sklearn SVD:  {pca_sklearn.explained_variance_ratio_}")
print(X_evd)

print("\n2. 特征值对比:")
print(f"EVD方法:      {eigenvals_evd}")
print(f"手动SVD:      {eigenvals_manual_svd}")
print(f"sklearn SVD:  {pca_sklearn.explained_variance_}")
print(X_manual_svd)

print(f"\n3. 性能对比:")
print(f"EVD方法耗时:      {evd_time:.6f}秒")
print(f"手动SVD耗时:      {manual_svd_time:.6f}秒")
print(f"sklearn SVD耗时:  {sklearn_time:.6f}秒")
print(X_sklearn_svd)


