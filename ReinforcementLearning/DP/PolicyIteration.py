import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

from ReinforcementLearning.DP.env import GridWorld


class PolicyIteration:
    """策略迭代算法"""

    def __init__(self, env, gamma=0.9, theta=1e-6):
        self.env = env
        self.gamma = gamma  # 折扣因子
        self.theta = theta  # 收敛阈值

        # 初始化随机策略（每个状态下的策略都是随机的，这里是相当于一种确定性策略的，在固定状态下它的下一步动作是确定的）
        self.policy = np.random.randint(0, env.num_actions, env.num_states)
        # 终止状态的策略设为0（不影响结果）
        for terminal in env.terminal_states:
            self.policy[terminal] = 0

        # 初始化状态价值函数
        self.V = np.zeros(env.num_states)

        # 记录迭代历史
        self.policy_history = []
        self.value_history = []

    def policy_evaluation(self):
        """策略评估：计算当前策略下的状态价值函数"""
        iteration = 0
        while True:
            delta = 0
            new_V = np.zeros(self.env.num_states)

            for s in range(self.env.num_states):
                if s in self.env.terminal_states:
                    new_V[s] = 0
                    continue

                action = self.policy[s]
                next_state = self.env.get_next_state(s, action)
                reward = self.env.get_reward(s, action, next_state)
                # 这里因为动作策略是确定的，状态转移也是确定的，所以这个新的价值函数就可以这样定义
                new_V[s] = reward + self.gamma * self.V[next_state]

                delta = max(delta, abs(new_V[s] - self.V[s]))

            self.V = new_V.copy()
            iteration += 1

            if delta < self.theta:
                break

        return iteration

    def policy_improvement(self):
        """策略改进：基于当前价值函数改进策略"""
        policy_stable = True

        for s in range(self.env.num_states):
            if s in self.env.terminal_states:
                continue

            old_action = self.policy[s]

            # 计算所有动作的动作价值
            action_values = np.zeros(self.env.num_actions)
            for a in range(self.env.num_actions):
                next_state = self.env.get_next_state(s, a)
                reward = self.env.get_reward(s, a, next_state)
                action_values[a] = reward + self.gamma * self.V[next_state]

            # 选择最优动作 更新策略
            self.policy[s] = np.argmax(action_values)

            # 看看旧动作是否相同，如果全部相同，就不用改了
            if old_action != self.policy[s]:
                policy_stable = False

        return policy_stable

    def solve(self, max_iterations=100):
        """执行策略迭代算法"""
        print("开始策略迭代...")

        for i in range(max_iterations):
            print(f"\n=== 第 {i + 1} 次策略迭代 ===")

            # 策略评估
            eval_iterations = self.policy_evaluation()
            print(f"策略评估收敛用了 {eval_iterations} 次迭代")

            # 记录历史
            self.policy_history.append(self.policy.copy())
            self.value_history.append(self.V.copy())

            # 策略改进
            policy_stable = self.policy_improvement()

            print(f"策略是否稳定: {policy_stable}")

            if policy_stable:
                print(f"\n策略迭代在第 {i + 1} 次迭代后收敛!")
                break

        return self.policy, self.V

    def visualize_policy(self, iteration=None):
        """可视化策略"""
        if iteration is not None and iteration < len(self.policy_history):
            policy_to_show = self.policy_history[iteration]
            title = f"policy ( {iteration + 1} )"
        else:
            policy_to_show = self.policy
            title = "final policy"

        fig, ax = plt.subplots(figsize=(8, 8))

        for s in range(self.env.num_states):
            row, col = self.env.state_to_coord(s)

            # 绘制网格
            rect = Rectangle((col, self.env.height - 1 - row), 1, 1,
                             linewidth=2, edgecolor='black', facecolor='lightblue')
            ax.add_patch(rect)

            # 绘制状态编号
            ax.text(col + 0.1, self.env.height - 1 - row + 0.8, str(s),
                    fontsize=10, fontweight='bold')

            # 绘制动作箭头
            if s not in self.env.terminal_states:
                action = policy_to_show[s]
                arrow = self.env.actions[action]
                ax.text(col + 0.5, self.env.height - 1 - row + 0.4, arrow,
                        fontsize=20, ha='center', va='center', color='red')
            else:
                ax.text(col + 0.5, self.env.height - 1 - row + 0.4, 'T',
                        fontsize=16, ha='center', va='center', color='green')

        ax.set_xlim(0, self.env.width)
        ax.set_ylim(0, self.env.height)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

        plt.tight_layout()
        plt.show()

    def visualize_values(self, iteration=None):
        """可视化状态价值函数"""
        if iteration is not None and iteration < len(self.value_history):
            values_to_show = self.value_history[iteration]
            title = f"value ( {iteration + 1} )"
        else:
            values_to_show = self.V
            title = "final value"

        # 重新整理为网格形式
        value_grid = values_to_show.reshape(self.env.height, self.env.width)

        plt.figure(figsize=(8, 6))
        sns.heatmap(value_grid, annot=True, fmt='.2f', cmap='viridis',
                    cbar_kws={'label': 'value'})
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('col')
        plt.ylabel('raw')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # 创建环境
    print("创建 4x4 网格世界环境...")
    env = GridWorld(5,6)

    # 创建策略迭代求解器
    print("初始化策略迭代算法...")
    solver = PolicyIteration(env, gamma=0.9, theta=1e-6)

    # 显示初始随机策略
    print("\n初始随机策略:")
    solver.visualize_policy()

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


    for s in range(env.num_states):
        row, col = env.state_to_coord(s)
        print(f"V*({s}) = {optimal_values[s]:.3f}")


    solver.visualize_policy()
    solver.visualize_values()

    # 显示迭代过程
    print(f"\n总共进行了 {len(solver.policy_history)} 次策略迭代")

    # 可视化第一次迭代的结果
    if len(solver.policy_history) > 1:
        print("\n第1次迭代后的策略:")
        solver.visualize_policy(0)
        solver.visualize_values(0)