import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')


class MaxEntropyFitter:
    """
    基于最大熵原则的指数族分布拟合器
    支持多种指数族分布：正态分布、指数分布、伽马分布、泊松分布等
    """

    def __init__(self):
        self.fitted_distributions = {}
        self.best_distribution = None
        self.best_params = None
        self.best_score = -np.inf

    def fit_normal(self, data):
        """拟合正态分布 (高斯分布)"""
        mu = np.mean(data)
        sigma = np.std(data, ddof=1)

        # 计算对数似然
        log_likelihood = np.sum(stats.norm.logpdf(data, mu, sigma))

        return {
            'distribution': 'normal',
            'params': {'mu': mu, 'sigma': sigma},
            'log_likelihood': log_likelihood,
            'aic': -2 * log_likelihood + 2 * 2,  # 2个参数
            'bic': -2 * log_likelihood + 2 * np.log(len(data))
        }

    def fit_exponential(self, data):
        """拟合指数分布"""
        if np.any(data <= 0):
            return None

        # 最大似然估计
        lambda_param = 1 / np.mean(data)

        # 计算对数似然
        log_likelihood = np.sum(stats.expon.logpdf(data, scale=1 / lambda_param))

        return {
            'distribution': 'exponential',
            'params': {'lambda': lambda_param},
            'log_likelihood': log_likelihood,
            'aic': -2 * log_likelihood + 2 * 1,  # 1个参数
            'bic': -2 * log_likelihood + 1 * np.log(len(data))
        }

    def fit_gamma(self, data):
        """拟合伽马分布"""
        if np.any(data <= 0):
            return None

        # 使用矩估计作为初始值
        sample_mean = np.mean(data)
        sample_var = np.var(data, ddof=1)

        # 防止数值问题
        if sample_var <= 0:
            return None

        # 矩估计
        beta_init = sample_mean / sample_var
        alpha_init = sample_mean * beta_init

        # 最大似然估计优化
        def neg_log_likelihood(params):
            alpha, beta = params
            if alpha <= 0 or beta <= 0:
                return np.inf
            try:
                return -np.sum(stats.gamma.logpdf(data, alpha, scale=1 / beta))
            except:
                return np.inf

        try:
            result = minimize(neg_log_likelihood, [alpha_init, beta_init],
                              method='L-BFGS-B', bounds=[(0.01, None), (0.01, None)])

            if result.success:
                alpha, beta = result.x
                log_likelihood = -result.fun

                return {
                    'distribution': 'gamma',
                    'params': {'alpha': alpha, 'beta': beta},
                    'log_likelihood': log_likelihood,
                    'aic': -2 * log_likelihood + 2 * 2,  # 2个参数
                    'bic': -2 * log_likelihood + 2 * np.log(len(data))
                }
        except:
            pass

        return None

    def fit_poisson(self, data):
        """拟合泊松分布（离散数据）"""
        # 检查是否为非负整数
        if not np.all(data >= 0) or not np.all(data == np.floor(data)):
            return None

        # 最大似然估计
        lambda_param = np.mean(data)

        # 计算对数似然
        log_likelihood = np.sum(stats.poisson.logpmf(data.astype(int), lambda_param))

        return {
            'distribution': 'poisson',
            'params': {'lambda': lambda_param},
            'log_likelihood': log_likelihood,
            'aic': -2 * log_likelihood + 2 * 1,  # 1个参数
            'bic': -2 * log_likelihood + 1 * np.log(len(data))
        }

    def fit_beta(self, data):
        """拟合Beta分布"""
        # 检查数据是否在[0,1]区间内
        if not (np.all(data >= 0) and np.all(data <= 1)):
            return None

        # 使用矩估计作为初始值
        sample_mean = np.mean(data)
        sample_var = np.var(data, ddof=1)

        if sample_var <= 0 or sample_mean <= 0 or sample_mean >= 1:
            return None

        # 矩估计
        common_factor = sample_mean * (1 - sample_mean) / sample_var - 1
        alpha_init = sample_mean * common_factor
        beta_init = (1 - sample_mean) * common_factor

        if alpha_init <= 0 or beta_init <= 0:
            return None

        # 最大似然估计优化
        def neg_log_likelihood(params):
            alpha, beta = params
            if alpha <= 0 or beta <= 0:
                return np.inf
            try:
                return -np.sum(stats.beta.logpdf(data, alpha, beta))
            except:
                return np.inf

        try:
            result = minimize(neg_log_likelihood, [alpha_init, beta_init],
                              method='L-BFGS-B', bounds=[(0.01, None), (0.01, None)])

            if result.success:
                alpha, beta = result.x
                log_likelihood = -result.fun

                return {
                    'distribution': 'beta',
                    'params': {'alpha': alpha, 'beta': beta},
                    'log_likelihood': log_likelihood,
                    'aic': -2 * log_likelihood + 2 * 2,  # 2个参数
                    'bic': -2 * log_likelihood + 2 * np.log(len(data))
                }
        except:
            pass

        return None

    def fit(self, data, criterion='aic'):
        """
        拟合多个指数族分布并选择最佳分布

        Parameters:
        -----------
        data : array-like
            输入数据
        criterion : str
            模型选择准则，'aic' 或 'bic'
        """
        data = np.array(data)

        # 尝试拟合各种分布
        distributions_to_try = [
            self.fit_normal,
            self.fit_exponential,
            self.fit_gamma,
            self.fit_poisson,
            self.fit_beta
        ]

        valid_fits = []

        for fit_func in distributions_to_try:
            try:
                result = fit_func(data)
                if result is not None:
                    valid_fits.append(result)
            except Exception as e:
                print(f"Warning: {fit_func.__name__} failed: {e}")
                continue

        if not valid_fits:
            raise ValueError("无法拟合任何分布")

        # 根据准则选择最佳分布
        self.fitted_distributions = valid_fits

        if criterion == 'aic':
            best_fit = min(valid_fits, key=lambda x: x['aic'])
        else:  # bic
            best_fit = min(valid_fits, key=lambda x: x['bic'])

        self.best_distribution = best_fit['distribution']
        self.best_params = best_fit['params']
        self.best_score = best_fit['log_likelihood']

        return self

    def predict_pdf(self, x):
        """预测概率密度函数值"""
        if self.best_distribution is None:
            raise ValueError("模型尚未拟合")

        if self.best_distribution == 'normal':
            return stats.norm.pdf(x, self.best_params['mu'], self.best_params['sigma'])
        elif self.best_distribution == 'exponential':
            return stats.expon.pdf(x, scale=1 / self.best_params['lambda'])
        elif self.best_distribution == 'gamma':
            return stats.gamma.pdf(x, self.best_params['alpha'], scale=1 / self.best_params['beta'])
        elif self.best_distribution == 'poisson':
            return stats.poisson.pmf(x.astype(int), self.best_params['lambda'])
        elif self.best_distribution == 'beta':
            return stats.beta.pdf(x, self.best_params['alpha'], self.best_params['beta'])

    def print_results(self):
        """打印拟合结果"""
        print("=" * 50)
        print("最大熵原则指数族分布拟合结果")
        print("=" * 50)

        print(f"最佳分布: {self.best_distribution}")
        print(f"参数: {self.best_params}")
        print(f"对数似然: {self.best_score:.4f}")

        print("\n所有拟合结果:")
        print("-" * 50)
        for fit in self.fitted_distributions:
            print(f"分布: {fit['distribution']}")
            print(f"  参数: {fit['params']}")
            print(f"  对数似然: {fit['log_likelihood']:.4f}")
            print(f"  AIC: {fit['aic']:.4f}")
            print(f"  BIC: {fit['bic']:.4f}")
            print()

    def plot_fit(self, data, bins=30):
        """绘制拟合结果"""
        if self.best_distribution is None:
            raise ValueError("模型尚未拟合")

        plt.figure(figsize=(12, 8))

        # 原始数据直方图
        plt.hist(data, bins=bins, density=True, alpha=0.7, color='skyblue',
                 label='原始数据', edgecolor='black')

        # 拟合的分布
        if self.best_distribution == 'poisson':
            # 离散分布
            x_range = np.arange(int(np.min(data)), int(np.max(data)) + 1)
            y_pred = self.predict_pdf(x_range)
            plt.plot(x_range, y_pred, 'ro-', linewidth=2, markersize=6,
                     label=f'拟合分布: {self.best_distribution}')
        else:
            # 连续分布
            x_range = np.linspace(np.min(data), np.max(data), 200)
            y_pred = self.predict_pdf(x_range)
            plt.plot(x_range, y_pred, 'r-', linewidth=2,
                     label=f'拟合分布: {self.best_distribution}')

        plt.xlabel('数值')
        plt.ylabel('概率密度')
        plt.title(f'最大熵原则拟合 - {self.best_distribution}分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# 演示函数
def demo():
    """演示不同类型的数据拟合"""
    np.random.seed(42)

    # 示例1: 正态分布数据
    print("示例1: 正态分布数据")
    normal_data = np.random.normal(5, 2, 1000)

    fitter1 = MaxEntropyFitter()
    fitter1.fit(normal_data)
    fitter1.print_results()
    fitter1.plot_fit(normal_data)

    # 示例2: 指数分布数据
    print("\n示例2: 指数分布数据")
    exp_data = np.random.exponential(2, 1000)

    fitter2 = MaxEntropyFitter()
    fitter2.fit(exp_data)
    fitter2.print_results()
    fitter2.plot_fit(exp_data)

    # 示例3: 泊松分布数据
    print("\n示例3: 泊松分布数据")
    poisson_data = np.random.poisson(3, 1000)

    fitter3 = MaxEntropyFitter()
    fitter3.fit(poisson_data)
    fitter3.print_results()
    fitter3.plot_fit(poisson_data)

    # 示例4: Beta分布数据
    print("\n示例4: Beta分布数据")
    beta_data = np.random.beta(2, 5, 1000)

    fitter4 = MaxEntropyFitter()
    fitter4.fit(beta_data)
    fitter4.print_results()
    fitter4.plot_fit(beta_data)


if __name__ == "__main__":
    demo()