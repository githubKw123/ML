import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

from ReinforcementLearning.DP.env import GridWorld



class ValueIteration:
    """价值迭代算法"""

    def __init__(self, env, gamma=0.9, theta=1e-6):
        self.env = env
        self.gamma = gamma  # 折扣因子
        self.theta = theta  # 收敛阈值

        # 初始化状态价值函数
        self.V = np.zeros(env.num_states)

        # 记录迭代历史
        self.value_history = []
        self.policy_history = []
        self.delta_history = []

    def bellman_optimality_update(self):
        """贝尔曼最优性更新：V(s) = max_a Σ p(s',r|s,a)[r + γV(s')]"""
        new_V = np.zeros(self.env.num_states)
        delta = 0

        for s in range(self.env.num_states):
            if s in self.env.terminal_states:
                new_V[s] = 0
                continue

            # 计算所有动作的期望回报
            action_values = np.zeros(self.env.num_actions)
            for a in range(self.env.num_actions):
                next_state = self.env.get_next_state(s, a)
                reward = self.env.get_reward(s, a, next_state)
                action_values[a] = reward + self.gamma * self.V[next_state]

            # 取最大值
            new_V[s] = np.max(action_values)
            delta = max(delta, abs(new_V[s] - self.V[s]))

        return new_V, delta

    def extract_policy(self):
        """从价值函数中提取最优策略"""
        policy = np.zeros(self.env.num_states, dtype=int)

        for s in range(self.env.num_states):
            if s in self.env.terminal_states:
                policy[s] = 0  # 终止状态动作不影响结果
                continue

            # 计算所有动作的期望回报
            action_values = np.zeros(self.env.num_actions)
            for a in range(self.env.num_actions):
                next_state = self.env.get_next_state(s, a)
                reward = self.env.get_reward(s, a, next_state)
                action_values[a] = reward + self.gamma * self.V[next_state]

            # 选择最优动作
            policy[s] = np.argmax(action_values)

        return policy

    def solve(self, max_iterations=1000):
        """执行价值迭代算法"""
        print("开始价值迭代...")
        print(f"收敛阈值: {self.theta}")
        print(f"最大迭代次数: {max_iterations}")

        for i in range(max_iterations):
            # 保存当前价值函数
            self.value_history.append(self.V.copy())

            # 贝尔曼最优性更新
            new_V, delta = self.bellman_optimality_update()

            # 记录变化量
            self.delta_history.append(delta)

            # 提取当前策略
            current_policy = self.extract_policy()
            self.policy_history.append(current_policy)

            print(f"迭代 {i + 1:3d}: δ = {delta:.6f}")

            # 更新价值函数
            self.V = new_V.copy()

            # 检查收敛
            if delta < self.theta:
                print(f"\n价值迭代在第 {i + 1} 次迭代后收敛!")
                print(f"最终δ = {delta:.8f}")
                break
        else:
            print(f"\n达到最大迭代次数 {max_iterations}，δ = {delta:.6f}")

        # 提取最终策略
        optimal_policy = self.extract_policy()

        return optimal_policy, self.V

    def get_action_values(self, state):
        """获取特定状态下所有动作的价值（Q值）"""
        if state in self.env.terminal_states:
            return np.zeros(self.env.num_actions)

        action_values = np.zeros(self.env.num_actions)
        for a in range(self.env.num_actions):
            next_state = self.env.get_next_state(state, a)
            reward = self.env.get_reward(state, a, next_state)
            action_values[a] = reward + self.gamma * self.V[next_state]

        return action_values

    def visualize_convergence(self):
        """可视化收敛过程"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # 绘制δ的变化
        ax1.plot(self.delta_history, 'b-', linewidth=2, marker='o', markersize=4)
        ax1.set_xlabel('迭代次数')
        ax1.set_ylabel('δ (最大价值变化)')
        ax1.set_title('价值迭代收敛过程')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=self.theta, color='r', linestyle='--', alpha=0.7, label=f'收敛阈值 ({self.theta})')
        ax1.legend()

        # 绘制几个关键状态的价值变化
        selected_states = [1, 5, 10, 14]  # 选择几个代表性状态
        for state in selected_states:
            if state < self.env.num_states:
                values = [v[state] for v in self.value_history]
                ax2.plot(values, label=f'状态 {state}', marker='o', markersize=3)

        ax2.set_xlabel('迭代次数')
        ax2.set_ylabel('状态价值')
        ax2.set_title('关键状态价值变化')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def visualize_policy(self, iteration=None):
        """可视化策略"""
        if iteration is not None and iteration < len(self.policy_history):
            policy_to_show = self.policy_history[iteration]
            values_to_show = self.value_history[iteration]
            title = f"策略和价值 (第 {iteration + 1} 次迭代)"
        else:
            policy_to_show = self.extract_policy()
            values_to_show = self.V
            title = "最优策略和价值"

        fig, ax = plt.subplots(figsize=(10, 8))

        for s in range(self.env.num_states):
            row, col = self.env.state_to_coord(s)

            # 根据价值设置颜色
            if s in self.env.terminal_states:
                color = 'lightgreen'
            else:
                # 价值越高颜色越深
                normalized_value = (values_to_show[s] - values_to_show.min()) / (
                            values_to_show.max() - values_to_show.min() + 1e-8)
                color = plt.cm.Blues(normalized_value * 0.8 + 0.2)

            # 绘制网格
            rect = Rectangle((col, self.env.height - 1 - row), 1, 1,
                             linewidth=2, edgecolor='black', facecolor=color)
            ax.add_patch(rect)

            # 绘制状态编号和价值
            ax.text(col + 0.1, self.env.height - 1 - row + 0.85, f's{s}',
                    fontsize=10, fontweight='bold')
            ax.text(col + 0.1, self.env.height - 1 - row + 0.1, f'{values_to_show[s]:.2f}',
                    fontsize=9, color='darkblue')

            # 绘制动作箭头
            if s not in self.env.terminal_states:
                action = policy_to_show[s]
                arrow = self.env.actions[action]
                ax.text(col + 0.5, self.env.height - 1 - row + 0.5, arrow,
                        fontsize=24, ha='center', va='center', color='red', fontweight='bold')
            else:
                ax.text(col + 0.5, self.env.height - 1 - row + 0.5, 'T',
                        fontsize=16, ha='center', va='center', color='darkgreen', fontweight='bold')

        ax.set_xlim(0, self.env.width)
        ax.set_ylim(0, self.env.height)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

        plt.tight_layout()
        plt.show()

    def visualize_values(self, iteration=None):
        """可视化状态价值函数热力图"""
        if iteration is not None and iteration < len(self.value_history):
            values_to_show = self.value_history[iteration]
            title = f"状态价值函数 (第 {iteration + 1} 次迭代)"
        else:
            values_to_show = self.V
            title = "最优状态价值函数"

        # 重新整理为网格形式
        value_grid = values_to_show.reshape(self.env.height, self.env.width)

        plt.figure(figsize=(8, 6))
        sns.heatmap(value_grid, annot=True, fmt='.2f', cmap='viridis',
                    cbar_kws={'label': '状态价值'}, square=True)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('列')
        plt.ylabel('行')
        plt.tight_layout()
        plt.show()

    def print_action_values(self, state):
        """打印特定状态的动作价值"""
        if state >= self.env.num_states:
            print(f"状态 {state} 不存在")
            return

        row, col = self.env.state_to_coord(state)
        print(f"\n状态 {state} ({row},{col}) 的动作价值:")

        if state in self.env.terminal_states:
            print("  终止状态，无需选择动作")
            return

        action_values = self.get_action_values(state)
        for a, value in enumerate(action_values):
            marker = " *" if a == np.argmax(action_values) else "  "
            print(f"  {self.env.actions[a]}: {value:.3f}{marker}")


# 演示代码和比较分析
def compare_algorithms():
    """比较策略迭代和价值迭代"""
    print("=== 算法比较：策略迭代 vs 价值迭代 ===\n")

    # 创建环境
    env = GridWorld(4, 4)

    # 价值迭代
    print("1. 价值迭代算法:")
    vi_solver = ValueIteration(env, gamma=0.9, theta=1e-6)
    vi_policy, vi_values = vi_solver.solve()
    vi_iterations = len(vi_solver.delta_history)

    print(f"\n价值迭代结果:")
    print(f"  - 迭代次数: {vi_iterations}")
    print(f"  - 最终最大价值变化: {vi_solver.delta_history[-1]:.8f}")

    return vi_solver, vi_policy, vi_values


if __name__ == "__main__":
    # 创建环境
    print("创建 4x4 网格世界环境...")
    env = GridWorld(4, 4)

    # 创建价值迭代求解器
    print("初始化价值迭代算法...")
    solver = ValueIteration(env, gamma=0.9, theta=1e-6)

    # 求解
    optimal_policy, optimal_values = solver.solve()

    # 显示结果
    print("\n=== 最终结果 ===")
    print("最优策略:")
    for s in range(env.num_states):
        row, col = env.state_to_coord(s)
        if s in env.terminal_states:
            print(f"状态 {s} ({row},{col}): 终止状态")
        else:
            action = optimal_policy[s]
            print(f"状态 {s} ({row},{col}): {env.actions[action]}")

    print(f"\n最优状态价值函数:")
    for s in range(env.num_states):
        row, col = env.state_to_coord(s)
        print(f"V*({s}) = {optimal_values[s]:.3f}")

    # 显示某个状态的动作价值
    solver.print_action_values(5)  # 状态5的动作价值
    solver.print_action_values(10)  # 状态10的动作价值

    # 可视化结果
    print("\n可视化结果:")

    # 收敛过程
    solver.visualize_convergence()

    # 最优策略和价值
    solver.visualize_policy()
    solver.visualize_values()

    # 显示几个中间迭代的结果
    if len(solver.policy_history) > 5:
        print("\n第1次迭代的结果:")
        solver.visualize_policy(0)

        print("\n第5次迭代的结果:")
        solver.visualize_policy(4)

    # 算法比较
    print("\n" + "=" * 50)
    compare_algorithms()

    print(f"\n价值迭代总迭代次数: {len(solver.delta_history)}")
    print("价值迭代算法完成！")