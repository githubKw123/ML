import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


# Epsilon-greedy policy
def epsilon_greedy(Q, state, epsilon, n_actions):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    else:
        return np.argmax(Q[state])


# SARSA
def sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    n_actions = env.action_space.n
    Q = np.zeros((env.observation_space.n, n_actions))
    rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        action = epsilon_greedy(Q, state, epsilon, n_actions)
        total_reward = 0
        done = False

        while not done:
            next_state, reward, done, truncated, _ = env.step(action)
            # 这里贪心策略选择下一步动作，形成一组SARSA
            next_action = epsilon_greedy(Q, next_state, epsilon, n_actions)
            Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])
            state, action = next_state, next_action
            total_reward += reward
            if truncated:
                done = True

        rewards.append(total_reward)

    return Q, rewards


# Q-Learning
def q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    n_actions = env.action_space.n
    Q = np.zeros((env.observation_space.n, n_actions))
    rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = epsilon_greedy(Q, state, epsilon, n_actions)
            next_state, reward, done, truncated, _ = env.step(action)
            # 下一步动作取最优的那个来更新
            best_next_action = np.argmax(Q[next_state])
            Q[state, action] += alpha * (reward + gamma * Q[next_state, best_next_action] - Q[state, action])
            state = next_state
            total_reward += reward
            if truncated:
                done = True

        rewards.append(total_reward)

    return Q, rewards


# Expected SARSA
def expected_sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    n_actions = env.action_space.n
    Q = np.zeros((env.observation_space.n, n_actions))
    rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = epsilon_greedy(Q, state, epsilon, n_actions)
            next_state, reward, done, truncated, _ = env.step(action)

            # 这里就是求一个动作价值的期望
            next_Q = Q[next_state]
            pi = np.ones(n_actions) * epsilon / n_actions
            best_action = np.argmax(next_Q)
            pi[best_action] += 1 - epsilon
            expected = np.sum(pi * next_Q)

            Q[state, action] += alpha * (reward + gamma * expected - Q[state, action])
            state = next_state
            total_reward += reward
            if truncated:
                done = True

        rewards.append(total_reward)

    return Q, rewards


# Unified function to switch between algorithms
def unified_rl(env, episodes, algorithm='q_learning', alpha=0.1, gamma=0.99, epsilon=0.1):
    if algorithm == 'sarsa':
        return sarsa(env, episodes, alpha, gamma, epsilon)
    elif algorithm == 'q_learning':
        return q_learning(env, episodes, alpha, gamma, epsilon)
    elif algorithm == 'expected_sarsa':
        return expected_sarsa(env, episodes, alpha, gamma, epsilon)
    else:
        raise ValueError("Unknown algorithm")


# Function to compute moving average
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


# Run experiment and plot performance
def run_experiment(episodes=500, runs=10, window_size=50):
    env = gym.make('CliffWalking-v0')
    '''
    CliffWalking-v0为4*12的网格世界，目标是从(3, 0)走到(3, 11)，其中(3,1-10)为悬崖
    状态：位置（48个格子对应的数）；动作：上下左右；奖励：悬崖-100，普通移动-1
    '''

    algorithms = ['sarsa', 'q_learning', 'expected_sarsa']
    all_rewards = {alg: [] for alg in algorithms}

    for alg in algorithms:
        for _ in range(runs):
            _, rewards = unified_rl(env, episodes, algorithm=alg, alpha=0.1, gamma=0.99, epsilon=0.1)
            all_rewards[alg].append(rewards)

    # Compute average rewards across runs
    avg_rewards = {alg: np.mean(all_rewards[alg], axis=0) for alg in algorithms}
    smoothed_rewards = {alg: moving_average(avg_rewards[alg], window_size) for alg in algorithms}

    # Plot performance
    plt.figure(figsize=(10, 6))
    for alg in algorithms:
        plt.plot(range(window_size - 1, episodes), smoothed_rewards[alg], label=alg.capitalize())
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward (Smoothed)')
    plt.title('Performance Comparison of SARSA, Q-Learning, and Expected SARSA')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print final average rewards (last 100 episodes)
    for alg in algorithms:
        final_avg = np.mean(avg_rewards[alg][-100:])
        print(f"{alg.capitalize()} final average reward (last 100 episodes): {final_avg:.2f}")


# Run the experiment
if __name__ == "__main__":
    run_experiment(episodes=500, runs=10, window_size=50)