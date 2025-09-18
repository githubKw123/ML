import numpy as np
from collections import defaultdict
import random
import gymnasium as gym


def monte_carlo_control(env, num_episodes, gamma=1.0, epsilon=0.1):
    """
    On-Policy Monte Carlo Control algorithm
    Parameters:
        env: Gymnasium environment with state and action spaces
        num_episodes: Number of episodes to run
        gamma: Discount factor
        epsilon: Exploration rate for epsilon-greedy policy
    Returns:
        Q: Action-value function
        policy: Optimal policy
    """
    # 初始化动作价值函数
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)

    # 定义贪心策略
    def epsilon_greedy_policy(state, Q, epsilon):
        state = tuple(state) if isinstance(state, (list, tuple)) else state
        if random.random() < epsilon:
            return random.randint(0, env.action_space.n - 1)
        else:
            return np.argmax(Q[state])


    for episode in range(num_episodes):
        # 收集数据
        episode_data = []
        state, _ = env.reset()
        state = tuple(state)
        done = False

        # 循环一轮收集数据
        while not done:
            action = epsilon_greedy_policy(state, Q, epsilon)
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = tuple(next_state)
            episode_data.append((state, action, reward))
            state = next_state
            done = done or truncated

        # 计算回报更新Q值
        G = 0
        visited_state_action = set()

        # 后序遍历收集
        for state, action, reward in reversed(episode_data):
            G = gamma * G + reward
            state_action = (state, action)

            # 第一次观测到
            if state_action not in visited_state_action:
                visited_state_action.add(state_action)
                returns_sum[state_action] += G
                returns_count[state_action] += 1
                Q[state][action] = returns_sum[state_action] / returns_count[state_action]

    # 贪心策略更新
    policy = {}
    for state in Q:
        policy[state] = np.argmax(Q[state])

    return Q, policy


# Example usage with Blackjack-v1
if __name__ == "__main__":
    env = gym.make('Blackjack-v1')
    Q, policy = monte_carlo_control(env, num_episodes=10000, gamma=0.9, epsilon=0.1)

    '''
    Blackjack-v1环境是单局的21点游戏
    状态（玩家手牌数、庄家点数、玩家是否有A）
    动作 （要牌，停牌）
    奖励 （输赢平）
    这是一个很典型的状态转移函数完全不可知的环境，只能通过MC方法进行控制
    '''
    # Print sample policy for a few states
    for state in list(policy.keys())[:5]:
        print(f"State (player_sum, dealer_card, usable_ace): {state}, Best action: {policy[state]}")