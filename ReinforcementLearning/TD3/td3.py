import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from collections import deque
import random
import matplotlib.pyplot as plt

# 设置随机种子
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# 超参数
BUFFER_SIZE = 100000
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
POLICY_DELAY = 2
POLICY_NOISE = 0.2
NOISE_CLIP = 0.5
EXPLORATION_NOISE = 0.1


# Actor网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x * self.max_action


# Critic网络
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1
        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

        # Q2
        self.fc4 = nn.Linear(state_dim + action_dim, 400)
        self.fc5 = nn.Linear(400, 300)
        self.fc6 = nn.Linear(300, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = torch.relu(self.fc1(sa))
        q1 = torch.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        q2 = torch.relu(self.fc4(sa))
        q2 = torch.relu(self.fc5(q2))
        q2 = self.fc6(q2)

        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = torch.relu(self.fc1(sa))
        q1 = torch.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        return q1


# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def size(self):
        return len(self.buffer)


# TD3算法
class TD3:
    def __init__(self, state_dim, action_dim, max_action):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.max_action = max_action
        self.total_it = 0

    def select_action(self, state, noise=0):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        if noise != 0:
            action = action + np.random.normal(0, noise, size=action.shape)
        return action.clip(-self.max_action, self.max_action)

    def train(self, replay_buffer):
        self.total_it += 1

        # 从经验池采样
        states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).reshape(-1, 1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).reshape(-1, 1).to(self.device)

        with torch.no_grad():
            # 选择下一个动作并添加噪声
            noise = (torch.randn_like(actions) * POLICY_NOISE).clamp(-NOISE_CLIP, NOISE_CLIP)
            next_actions = (self.actor_target(next_states) + noise).clamp(-self.max_action, self.max_action)

            # 计算目标Q值
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * GAMMA * target_q

        # 更新Critic
        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 延迟更新Actor
        if self.total_it % POLICY_DELAY == 0:
            actor_loss = -self.critic.Q1(states, self.actor(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 软更新目标网络
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)


# 训练函数
def train_td3():

    env = gym.make('Pendulum-v1')
    state_dim = env.observation_space.shape[0]  # 3维状态空间 (cos(theta), sin(theta), theta_dot)
    action_dim = env.action_space.shape[0]  # 1维连续动作空间（扭矩）
    max_action = float(env.action_space.high[0])  # 动作的最大值



    agent = TD3(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    episodes = 200
    max_steps = 200
    reward_history = []

    # 初始随机探索
    print("开始初始化经验池...")
    state, _ = env.reset(seed=SEED)
    for _ in range(1000):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.add(state, action, reward, next_state, float(done))
        state = next_state if not done else env.reset(seed=SEED)[0]
    print(f"初始化完成，经验池大小: {replay_buffer.size()}")
    print("=" * 60)

    for episode in range(episodes):
        state, _ = env.reset(seed=SEED + episode)
        episode_reward = 0

        for step in range(max_steps):
            # 选择动作（添加探索噪声）
            action = agent.select_action(state, noise=EXPLORATION_NOISE)

            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 存储经验
            replay_buffer.add(state, action, reward, next_state, float(done))

            # 训练
            agent.train(replay_buffer)

            state = next_state
            episode_reward += reward

            if done:
                break

        reward_history.append(episode_reward)

        # 计算最近10个episode的平均奖励
        avg_reward = np.mean(reward_history[-10:])

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{episodes}, Reward: {episode_reward:.2f}, Avg(10): {avg_reward:.2f}")

    env.close()
    return agent, reward_history


# 测试函数
def test_td3(agent, episodes=10, render=False):
    if render:
        env = gym.make('Pendulum-v1', render_mode='human')
    else:
        env = gym.make('Pendulum-v1')

    total_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0

        for _ in range(200):
            # 不添加探索噪声
            action = agent.select_action(state, noise=0)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state
            episode_reward += reward
            steps += 1

            if done:
                break

        total_rewards.append(episode_reward)
        print(f"Test Episode {episode + 1}, Reward: {episode_reward:.2f}, Steps: {steps}")

    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"\n平均测试奖励: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Pendulum-v1解决标准: 平均奖励 > -200")


    env.close()
    return total_rewards


# 绘制训练曲线
def plot_rewards(reward_history):
    plt.figure(figsize=(10, 5))
    plt.plot(reward_history, alpha=0.6, label='Episode Reward')

    # 绘制移动平均
    window = 10
    if len(reward_history) >= window:
        moving_avg = np.convolve(reward_history, np.ones(window) / window, mode='valid')
        plt.plot(range(window - 1, len(reward_history)), moving_avg,
                 linewidth=2, label=f'Moving Average ({window})')

    plt.axhline(y=-200, color='r', linestyle='--', label='Solved Threshold')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('TD3 Training on Pendulum-v1')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # 训练
    agent, reward_history = train_td3()

    print("\n" + "=" * 60)
    print("训练完成！开始测试...")
    print("=" * 60 + "\n")

    # 测试
    test_rewards = test_td3(agent, episodes=20, render=False)

    # 绘制训练曲线
    plot_rewards(reward_history)