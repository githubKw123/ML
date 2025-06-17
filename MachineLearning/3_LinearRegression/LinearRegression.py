import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False   #这两行需要手动设置
# 设置随机种子以保证结果可重现
np.random.seed(42)

# 普通线性回归手动实现（正规方程）
def manual_linear_regression_2d(X, y):
    """
    二维线性回归手动实现
    y = θ0 + θ1*x1 + θ2*x2
    使用正规方程: θ = (X^T * X)^(-1) * X^T * y
    """
    # 添加偏置项（截距项）
    X_with_bias = np.column_stack([np.ones(len(X)), X])

    # 计算参数 θ = (X^T * X)^(-1) * X^T * y
    XTX = X_with_bias.T @ X_with_bias
    XTy = X_with_bias.T @ y
    theta = np.linalg.inv(XTX) @ XTy

    return theta  # [θ0, θ1, θ2]

# L2正则化线性回归手动实现（正规方程）
def manual_ridge_regression_2d(X, y, alpha=1.0):
    """
    二维L2正则化线性回归手动实现
    损失函数: J(θ) = ||Xθ - y||² + α * ||θ||²
    正规方程解: θ = (X^T * X + α * I)^(-1) * X^T * y
    """
    # 添加偏置项
    X_with_bias = np.column_stack([np.ones(len(X)), X])

    # 构建正则化矩阵（通常不对截距项进行正则化）
    I = np.eye(X_with_bias.shape[1])
    I[0, 0] = 0  # 不对截距项正则化

    # Ridge回归正规方程: θ = (X^T * X + α * I)^(-1) * X^T * y
    XTX_reg = X_with_bias.T @ X_with_bias + alpha * I
    XTy = X_with_bias.T @ y
    theta = np.linalg.inv(XTX_reg) @ XTy

    return theta

def sklearn_linear_regression(X_train, y_train):
    # sklearn普通线性回归
    sklearn_lr = LinearRegression()
    sklearn_lr.fit(X_train, y_train)
    return sklearn_lr



# sklearn Ridge回归
def sklearn_ridge_regression(X_train, y_train):
    sklearn_ridge = Ridge(alpha=1.0)
    sklearn_ridge.fit(X_train, y_train)
    return sklearn_ridge

# 预测函数
def predict_2d(X, theta):
    """使用拟合的参数进行预测"""
    X_with_bias = np.column_stack([np.ones(len(X)), X])
    return X_with_bias @ theta


if __name__ == '__main__':
    # 生成二维示例数据
    n_samples = 200
    X1 = np.random.randn(n_samples) * 2  # 第一个特征
    X2 = np.random.randn(n_samples) * 1.5  # 第二个特征
    X = np.column_stack([X1, X2])  # 合并为二维特征矩阵

    # 生成目标值：y = 3*x1 + 2*x2 + 1 + 噪声
    y = 3 * X1 + 2 * X2 + 1 + np.random.randn(n_samples) * 0.8

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")
    print(f"特征维度: {X_train.shape[1]}")

    # 普通回归
    theta_lr = manual_linear_regression_2d(X_train, y_train)
    y_train_pred_lr = predict_2d(X_train, theta_lr)
    y_test_pred_lr = predict_2d(X_test, theta_lr)
    train_mse_liner = mean_squared_error(y_train, y_train_pred_lr)
    test_mse_liner = mean_squared_error(y_test, y_test_pred_lr)

    # 岭回归
    theta_ridge = manual_ridge_regression_2d(X_train, y_train, alpha=1)
    y_train_pred = predict_2d(X_train, theta_ridge)
    y_test_pred = predict_2d(X_test, theta_ridge)
    train_mse_ridge = mean_squared_error(y_train, y_train_pred)
    test_mse_ridge = mean_squared_error(y_test, y_test_pred)

    sklearn_lr = sklearn_linear_regression(X_train, y_train)
    y_train_sklearn_lr = sklearn_lr.predict(X_train)
    y_test_sklearn_lr = sklearn_lr.predict(X_test)
    mse_train_sklearn_lr = mean_squared_error(y_train, y_train_sklearn_lr)
    mse_test_sklearn_lr = mean_squared_error(y_test, y_test_sklearn_lr)

    sklearn_ridge = sklearn_ridge_regression(X_train, y_train)
    y_train_sklearn_ridge = sklearn_ridge.predict(X_train)
    y_test_sklearn_ridge = sklearn_ridge.predict(X_test)
    mse_train_sklearn_ridge = mean_squared_error(y_train, y_train_sklearn_ridge)
    mse_test_sklearn_ridge = mean_squared_error(y_test, y_test_sklearn_ridge)

    print("方法                | 截距    | 系数1   | 系数2   | 训练MSE | 测试MSE")
    print("-" * 50)
    print(
        f"手动实现(普通回归)   | {theta_lr[0]:6.2f} | {theta_lr[1]:6.2f} | {theta_lr[2]:6.2f} | {train_mse_liner:6.2f} | {test_mse_liner:6.2f}")
    print(
        f"sklearn LinearReg   | {sklearn_lr.intercept_:6.2f} | {sklearn_lr.coef_[0]:6.2f} | {sklearn_lr.coef_[1]:6.2f} | {mse_train_sklearn_lr:6.2f} | {mse_test_sklearn_lr:6.2f}")
    print(
        f"手动实现(Ridge α=1) | {theta_ridge[0]:6.2f} | {theta_ridge[1]:6.2f} | {theta_ridge[2]:6.2f} | {train_mse_ridge:6.2f} | {test_mse_ridge:6.2f}")
    print(
        f"sklearn Ridge α=1   | {sklearn_ridge.intercept_:6.2f} | {sklearn_ridge.coef_[0]:6.2f} | {sklearn_ridge.coef_[1]:6.2f} | {mse_train_sklearn_ridge:6.2f} | {mse_test_sklearn_ridge:6.2f}")

    fig = plt.figure(figsize=(16, 12))

    #原始数据3D散点图
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.scatter(X_train[:, 0], X_train[:, 1], y_train, alpha=0.6, c='blue', s=20)
    ax1.set_xlabel('X1')
    ax1.set_ylabel('X2')
    ax1.set_zlabel('y')
    ax1.set_title('训练数据 3D 散点图')

    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    # 创建网格用于绘制平面
    x1_range = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 20)
    x2_range = np.linspace(X_train[:, 1].min(), X_train[:, 1].max(), 20)
    X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)


    theta = theta_lr
    Y_grid = theta[0] + theta[1] * X1_grid + theta[2] * X2_grid
    ax2.plot_surface(X1_grid, X2_grid, Y_grid, alpha=0.3, color='red')

    ax2.scatter(X_train[:, 0], X_train[:, 1], y_train, alpha=0.6, c='blue', s=20)
    ax2.set_xlabel('X1')
    ax2.set_ylabel('X2')
    ax2.set_zlabel('y')

    plt.show()





