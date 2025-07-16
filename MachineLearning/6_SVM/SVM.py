import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False   #这两行需要手动设置
warnings.filterwarnings('ignore')


class SVM:
    def __init__(self, C=1.0, kernel='linear', gamma=1.0, degree=3):
        """
        SVM分类器（从零实现）
        参数:
        C: 正则化参数
        kernel: 核函数类型 ('linear', 'rbf', 'poly')
        gamma: RBF和多项式核参数
        degree: 多项式核的度数
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.alpha = None
        self.b = 0
        self.X_train = None
        self.y_train = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.support_vector_alphas = None

    def _kernel_function(self, x1, x2):
        """计算核函数"""
        # 线性核函数
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        # 径向核函数
        elif self.kernel == 'rbf':
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)
        # 多项式核函数
        elif self.kernel == 'poly':
            return (self.gamma * np.dot(x1, x2) + 1) ** self.degree
        else:
            raise ValueError("不支持的核函数类型")

    def _compute_kernel_matrix(self, X1, X2):
        """计算核矩阵，也就是对应内积那一部分"""
        n1, n2 = X1.shape[0], X2.shape[0]
        K = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                K[i, j] = self._kernel_function(X1[i], X2[j])
        return K

    def _objective_function(self, alpha):
        """SVM的目标函数（对偶问题）"""
        K = self._compute_kernel_matrix(self.X_train, self.X_train)

        # 这里求的是公式里的lambda 计算 W(alpha) = sum(alpha) - 0.5 * sum(alpha_i * alpha_j * y_i * y_j * K(x_i, x_j))
        first_term = np.sum(alpha)
        second_term = 0.5 * np.sum(alpha[:, np.newaxis] * alpha[np.newaxis, :] *
                                   self.y_train[:, np.newaxis] * self.y_train[np.newaxis, :] * K)

        return -(first_term - second_term)  # 因为minimize函数是最小化，所以加负号

    def _constraint_function(self, alpha):
        """约束条件：sum(alpha_i * y_i) = 0"""
        return np.sum(alpha * self.y_train)

    def fit(self, X, y):
        """训练SVM模型"""
        self.X_train = X
        self.y_train = y
        n_samples = X.shape[0]

        # 初始化拉格朗日乘数
        alpha_init = np.random.random(n_samples) * 0.01

        # 约束条件
        constraints = {'type': 'eq', 'fun': self._constraint_function}
        bounds = [(0, self.C) for _ in range(n_samples)]

        # 求解优化问题 先求出lambda
        result = minimize(
            fun=self._objective_function,
            x0=alpha_init,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )

        self.alpha = result.x

        # 找到支持向量（alpha > 1e-5的样本） lambda非零的值表示为支持相量
        support_vector_indices = self.alpha > 1e-5
        self.support_vectors = X[support_vector_indices]
        self.support_vector_labels = y[support_vector_indices]
        self.support_vector_alphas = self.alpha[support_vector_indices]

        # 计算偏置项 b
        if len(self.support_vector_alphas) > 0:
            # 选择一个支持向量来计算b
            sv_idx = 0
            sv_x = self.support_vectors[sv_idx]
            sv_y = self.support_vector_labels[sv_idx]

            self.b = sv_y
            for i in range(len(self.support_vectors)):
                self.b -= self.support_vector_alphas[i] * self.support_vector_labels[i] * \
                          self._kernel_function(self.support_vectors[i], sv_x)


    def predict(self, X):
        """预测"""
        if self.support_vectors is None:
            raise ValueError("模型尚未训练")

        predictions = []
        for x in X:
            prediction = 0
            for i in range(len(self.support_vectors)):
                # 这里其实相当于在计算wx项
                prediction += self.support_vector_alphas[i] * self.support_vector_labels[i] * \
                              self._kernel_function(self.support_vectors[i], x)
            # 这里计算wx+b，大于0则分类成功
            prediction += self.b
            predictions.append(1 if prediction >= 0 else -1)

        return np.array(predictions)

    def decision_function(self, X):
        """决策函数值"""
        if self.support_vectors is None:
            raise ValueError("模型尚未训练")

        scores = []
        for x in X:
            score = 0
            for i in range(len(self.support_vectors)):
                score += self.support_vector_alphas[i] * self.support_vector_labels[i] * \
                         self._kernel_function(self.support_vectors[i], x)
            score += self.b
            scores.append(score)

        return np.array(scores)

# 数据生成函数
def generate_data():
    """生成二分类数据"""
    np.random.seed(42)

    # 生成两个类别的数据
    class1 = np.random.multivariate_normal([2, 2], [[0.5, 0], [0, 0.5]], 50)
    class2 = np.random.multivariate_normal([-2, -2], [[0.5, 0], [0, 0.5]], 50)

    X = np.vstack([class1, class2])
    y = np.hstack([np.ones(50), -np.ones(50)])

    return X, y

def train_test_split(X, y, test_size=0.3, random_state=42):
    """简单的训练测试集划分"""
    np.random.seed(random_state)
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    n_test = int(n_samples * test_size)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

def accuracy_score(y_true, y_pred):
    """计算准确率"""
    return np.mean(y_true == y_pred)

def plot_decision_boundary(X, y, svm_model, title="SVM决策边界"):
    """绘制决策边界"""
    plt.figure(figsize=(10, 8))

    # 创建网格
    h = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # 预测网格点
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = svm_model.decision_function(grid_points)
    Z = Z.reshape(xx.shape)

    # 绘制决策边界和边界
    plt.contour(xx, yy, Z, levels=[-1, 0, 1], alpha=0.8,
                linestyles=['--', '-', '--'], colors=['red', 'black', 'red'])
    plt.contourf(xx, yy, Z, levels=50, alpha=0.3, cmap=plt.cm.RdYlBu)

    # 绘制数据点
    positive_idx = y == 1
    negative_idx = y == -1

    plt.scatter(X[positive_idx, 0], X[positive_idx, 1], c='blue',
                marker='o', s=50, label='正类 (+1)', edgecolors='black')
    plt.scatter(X[negative_idx, 0], X[negative_idx, 1], c='red',
                marker='s', s=50, label='负类 (-1)', edgecolors='black')

    # 标记支持向量
    if svm_model.support_vectors is not None:
        plt.scatter(svm_model.support_vectors[:, 0], svm_model.support_vectors[:, 1],
                    s=200, facecolors='none', edgecolors='green', linewidth=2,
                    label='支持向量')

    plt.xlabel('特征 1')
    plt.ylabel('特征 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()



if __name__ == "__main__":
# 生成数据
    X, y = generate_data()

    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


    # 测试不同核函数
    kernels = ['linear', 'rbf', 'poly']

    plt.figure(figsize=(15, 5))

    for i, kernel in enumerate(kernels):
        print(f"\n=== 测试 {kernel} 核函数 ===")

        # 创建和训练SVM
        if kernel == 'rbf':
            svm = SVM(C=1.0, kernel=kernel, gamma=0.5)
        elif kernel == 'poly':
            svm = SVM(C=1.0, kernel=kernel, gamma=1.0, degree=2)
        else:
            svm = SVM(C=1.0, kernel=kernel)

        svm.fit(X_train, y_train)

        # 预测
        y_pred = svm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"测试集准确率: {accuracy:.4f}")

        # 绘制决策边界
        plt.subplot(1, 3, i + 1)

        h = 0.1
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = svm.decision_function(grid_points)
        Z = Z.reshape(xx.shape)

        plt.contour(xx, yy, Z, levels=[-1, 0, 1], alpha=0.8,
                    linestyles=['--', '-', '--'], colors=['red', 'black', 'red'])
        plt.contourf(xx, yy, Z, levels=50, alpha=0.3, cmap=plt.cm.RdYlBu)

        positive_idx = y == 1
        negative_idx = y == -1

        plt.scatter(X[positive_idx, 0], X[positive_idx, 1], c='blue',
                    marker='o', s=30, edgecolors='black')
        plt.scatter(X[negative_idx, 0], X[negative_idx, 1], c='red',
                    marker='s', s=30, edgecolors='black')

        if svm.support_vectors is not None:
            plt.scatter(svm.support_vectors[:, 0], svm.support_vectors[:, 1],
                        s=100, facecolors='none', edgecolors='green', linewidth=2)

        plt.title(f'{kernel}核 (准确率: {accuracy:.3f})')
        plt.xlabel('特征 1')
        plt.ylabel('特征 2')

    plt.tight_layout()
    plt.show()

    # 详细测试线性核
    print(f"\n=== 详细测试线性核 ===")
    svm_linear = SVM(C=1.0, kernel='linear')
    svm_linear.fit(X_train, y_train)

    y_pred = svm_linear.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"线性核测试集准确率: {accuracy:.4f}")
    print(f"支持向量数量: {len(svm_linear.support_vectors)}")

    # 绘制详细的决策边界
    plot_decision_boundary(X, y, svm_linear, "线性核SVM决策边界")
