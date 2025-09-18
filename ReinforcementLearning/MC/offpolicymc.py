import numpy as np
from collections import defaultdict
import random
import gymnasium as gym


def monte_carlo_off_policy_control(env, num_episodes, gamma=1.0, epsilon=0.1):
    """
    Off-Policy Monte Carlo Control algorithm using Weighted Importance Sampling
    Parameters:
        env: Gymnasium environment with state and action spaces
        num_episodes: Number of episodes to run
        gamma: Discount factor
        epsilon: Exploration rate for behavior policy (epsilon-soft)
    Returns:
        Q: Action-value function (target policy)
        policy: Optimal greedy policy
    """
    # 初始化动作价值函数和重要性
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    C = defaultdict(lambda: np.zeros(env.action_space.n))  # Cumulative weights

    # 行动策略用于收集数据，为贪婪策略
    def behavior_policy(state, Q, epsilon):
        state = tuple(state) if isinstance(state, (list, tuple)) else state
        if random.random() < epsilon:
            return random.randint(0, env.action_space.n - 1)
        else:
            return np.argmax(Q[state])

    # 目标策略用于更新
    def target_policy(state, Q):
        state = tuple(state) if isinstance(state, (list, tuple)) else state
        return np.argmax(Q[state])

    # 循环
    for episode in range(num_episodes):

        episode_data = []
        state, _ = env.reset()
        state = tuple(state)
        done = False

        # 行动策略收集数据，这里行动策略必须是软性策略
        while not done:
            action = behavior_policy(state, Q, epsilon)
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = tuple(next_state)
            episode_data.append((state, action, reward))
            state = next_state
            done = done or truncated


        G = 0.0
        W = 1.0  # Importance sampling ratio

        # Process episode in reverse order (first-visit)
        for state, action, reward in reversed(episode_data):
            G = gamma * G + reward
            state_action = (state, action)

            # 增量更新权重和Q值
            C[state][action] += W
            Q[state][action] += (W / C[state][action]) * (G - Q[state][action])

            # Update importance ratio for next step
            target_action = target_policy(state, Q)
            if action != target_action:
                break  # 正常来说b用于采样可以使用的前提是这个策略pi也可以产生，如果动作与目标动作不等的话，就说明这组序列是不可用的

            # 这里求的是pi\b
            num_actions = env.action_space.n
            prob_b = epsilon / num_actions + (1 - epsilon) if action == target_action else epsilon / num_actions
            W = W / prob_b  # W = W * (pi / b)

    # Derive optimal greedy policy from Q
    policy = {}
    for state in Q:
        policy[state] = np.argmax(Q[state])

    return Q, policy


# Example usage with Blackjack-v1
if __name__ == "__main__":
    env = gym.make('Blackjack-v1')
    Q, policy = monte_carlo_off_policy_control(env, num_episodes=10000, gamma=0.9, epsilon=0.1)

    # Print sample policy for a few states
    for state in list(policy.keys())[:5]:
        print(f"State (player_sum, dealer_card, usable_ace): {state}, Best action: {policy[state]}")