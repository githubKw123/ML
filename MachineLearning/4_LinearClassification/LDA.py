import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False   #这两行需要手动设置



class BinaryLDA:
    def __init__(self):
        """
        二分类线性判别分析
        通过投影方向将二维数据投影到一维，并确定分类阈值
        """
        self.w = None  # 投影方向向量
        self.threshold = None  # 分类阈值
        self.mean1 = None  # 类别1的均值
        self.mean2 = None  # 类别2的均值
        self.class_labels = None  # 类别标签

    def fit(self, X, y):
        """
        训练二分类LDA模型

        参数:
        X: 训练数据，shape为(n_samples, 2) - 二维数据
        y: 标签，shape为(n_samples,) - 二进制标签(0/1 或 -1/1)
        """
        # 确保是二维数据
        if X.shape[1] != 2:
            raise ValueError("此实现仅支持二维数据")

        # 获取类别标签
        self.class_labels = np.unique(y)
        if len(self.class_labels) != 2:
            raise ValueError("此实现仅支持二分类")

        # 分离两个类别的数据
        class1_mask = (y == self.class_labels[0])
        class2_mask = (y == self.class_labels[1])

        X1 = X[class1_mask]  # 类别1的数据
        X2 = X[class2_mask]  # 类别2的数据

        # 计算各类别的均值
        self.mean1 = np.mean(X1, axis=0)
        self.mean2 = np.mean(X2, axis=0)

        # 计算类内散布矩阵 S_W = S1 + S2
        # S1 = Σ(x - μ1)(x - μ1)^T for class 1
        # S2 = Σ(x - μ2)(x - μ2)^T for class 2
        S1 = np.zeros((2, 2))
        for x in X1:
            diff = (x - self.mean1).reshape(-1, 1)
            S1 += np.dot(diff, diff.T)

        S2 = np.zeros((2, 2))
        for x in X2:
            diff = (x - self.mean2).reshape(-1, 1)
            S2 += np.dot(diff, diff.T)

        S_W = S1 + S2

        # 添加正则化以避免奇异矩阵
        S_W += 1e-6 * np.eye(2)

        # 计算最优投影方向
        # w = S_W^(-1) * (μ1 - μ2)
        mean_diff = self.mean1 - self.mean2
        try:
            self.w = np.linalg.solve(S_W, mean_diff)
        except np.linalg.LinAlgError:
            # 如果求解失败，使用伪逆
            self.w = np.dot(np.linalg.pinv(S_W), mean_diff)

        # 归一化投影向量
        self.w = self.w / np.linalg.norm(self.w)

        # 计算两个类别在投影方向上的均值
        proj_mean1 = np.dot(self.mean1, self.w)
        proj_mean2 = np.dot(self.mean2, self.w)

        # 分类阈值为两个投影均值的中点
        self.threshold = (proj_mean1 + proj_mean2) / 2

        return self

    def project(self, X):
        """
        将数据投影到一维空间

        参数:
        X: 待投影数据，shape为(n_samples, 2)

        返回:
        投影后的一维数据，shape为(n_samples,)
        """
        return np.dot(X, self.w)

    def predict(self, X):
        """
        预测类别

        参数:
        X: 待预测数据，shape为(n_samples, 2)

        返回:
        预测的类别标签
        """
        projections = self.project(X)
        predictions = np.where(projections > self.threshold,
                               self.class_labels[0], self.class_labels[1])
        return predictions

    def plot_results(self, X, y, title="二分类LDA结果"):
        """
        可视化分类结果
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 左图：原始二维数据和决策边界
        colors = ['red', 'blue']
        for i, label in enumerate(self.class_labels):
            mask = (y == label)
            ax1.scatter(X[mask, 0], X[mask, 1], c=colors[i],
                        label=f'类别 {label}', alpha=0.7, s=50)

        # 绘制类别均值
        ax1.scatter(self.mean1[0], self.mean1[1], c='red', marker='x', s=200, linewidth=3)
        ax1.scatter(self.mean2[0], self.mean2[1], c='blue', marker='x', s=200, linewidth=3)

        # 绘制投影方向
        center = (self.mean1 + self.mean2) / 2
        scale = 3
        ax1.arrow(center[0], center[1],
                  self.w[0] * scale, self.w[1] * scale,
                  head_width=0.2, head_length=0.3, fc='green', ec='green', linewidth=2)
        ax1.arrow(center[0], center[1],
                  -self.w[0] * scale, -self.w[1] * scale,
                  head_width=0.2, head_length=0.3, fc='green', ec='green', linewidth=2)

        # 绘制决策边界（垂直于投影方向）
        # 决策边界方程：w^T * (x - x0) = 0，其中x0是阈值点
        if abs(self.w[1]) > 1e-10:  # 避免除零
            x_range = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
            # 找到阈值对应的点
            threshold_point = center + (self.threshold - np.dot(center, self.w)) * self.w / np.dot(self.w, self.w)

            # 决策边界的方向向量（垂直于投影方向）
            boundary_direction = np.array([-self.w[1], self.w[0]])

            # 绘制决策边界
            boundary_scale = 5
            x_boundary = [threshold_point[0] - boundary_direction[0] * boundary_scale,
                          threshold_point[0] + boundary_direction[0] * boundary_scale]
            y_boundary = [threshold_point[1] - boundary_direction[1] * boundary_scale,
                          threshold_point[1] + boundary_direction[1] * boundary_scale]

            ax1.plot(x_boundary, y_boundary, 'g--', linewidth=2, label='决策边界')

        ax1.set_xlabel('特征 1')
        ax1.set_ylabel('特征 2')
        ax1.set_title('原始数据和投影方向')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')

        # 右图：投影后的一维数据
        projections = self.project(X)

        for i, label in enumerate(self.class_labels):
            mask = (y == label)
            ax2.scatter(projections[mask], np.zeros(np.sum(mask)),
                        c=colors[i], label=f'类别 {label}', alpha=0.7, s=50)

        # 绘制阈值线
        ax2.axvline(x=self.threshold, color='green', linestyle='--', linewidth=2, label='分类阈值')

        # 绘制各类别在投影方向上的均值
        proj_mean1 = np.dot(self.mean1, self.w)
        proj_mean2 = np.dot(self.mean2, self.w)
        ax2.scatter(proj_mean1, 0, c='red', marker='x', s=200, linewidth=3)
        ax2.scatter(proj_mean2, 0, c='blue', marker='x', s=200, linewidth=3)

        ax2.set_xlabel('投影值')
        ax2.set_ylabel('')
        ax2.set_title('投影到一维空间')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-0.5, 0.5)

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    def get_projection_info(self):
        """获取投影信息"""
        if self.w is None:
            print("模型尚未训练")
            return

        print("=== LDA投影信息 ===")
        print(f"投影向量 w: [{self.w[0]:.4f}, {self.w[1]:.4f}]")
        print(f"投影向量长度: {np.linalg.norm(self.w):.4f}")
        print(f"投影角度: {np.arctan2(self.w[1], self.w[0]) * 180 / np.pi:.2f}°")
        print(f"分类阈值: {self.threshold:.4f}")

        proj_mean1 = np.dot(self.mean1, self.w)
        proj_mean2 = np.dot(self.mean2, self.w)
        print(f"类别 {self.class_labels[0]} 投影均值: {proj_mean1:.4f}")
        print(f"类别 {self.class_labels[1]} 投影均值: {proj_mean2:.4f}")
        print(f"投影均值间距: {abs(proj_mean1 - proj_mean2):.4f}")








# 主程序
if __name__ == "__main__":


    X, y = make_classification(n_samples=300, n_features=2, n_redundant=0,
                               n_informative=2, n_clusters_per_class=1,
                               n_classes=2, random_state=19)

    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 训练LDA
    lda = BinaryLDA()
    lda.fit(X_scaled, y)

    # 预测
    y_pred = lda.predict(X_scaled)
    accuracy = np.mean(y_pred == y)

    print(f"分类准确率: {accuracy:.2%}")

    # 显示投影信息
    lda.get_projection_info()

    # 可视化结果
    lda.plot_results(X_scaled, y, "二分类LDA - 合成数据")




