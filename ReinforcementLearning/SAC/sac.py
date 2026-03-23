import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
TAU = 0.005  # 软更新系数
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
LR_ALPHA = 3e-4
ALPHA = 0.2  # 温度参数初始值（自动调整）
TARGET_ENTROPY = None  # 目标熵（会自动设置）


# Actor网络（随机策略）
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        self.max_action = max_action

        # 初始化参数
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)  # 限制log_std范围
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        # 重参数化技巧
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action

        # 计算log_prob
        log_prob = normal.log_prob(x_t)
        # 修正tanh变换的log_prob
        log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        mean = torch.tanh(mean) * self.max_action

        return action, log_prob, mean


# Critic网络（Q函数）
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

        # Q2
        self.fc4 = nn.Linear(state_dim + action_dim, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 1)

        # 初始化参数
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        q2 = F.relu(self.fc4(sa))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)

        return q1, q2


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


# SAC算法
class SAC:
    def __init__(self, state_dim, action_dim, max_action):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # Actor网络
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        # Critic网络（双Q网络）
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        # 自动调整温度参数alpha
        self.target_entropy = -action_dim  # 启发式目标熵
        self.log_alpha = torch.tensor(np.log(ALPHA), requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LR_ALPHA)

        self.max_action = max_action

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if deterministic:
                _, _, action = self.actor.sample(state)
            else:
                action, _, _ = self.actor.sample(state)

        return action.cpu().numpy()[0]

    def train(self, replay_buffer):
        # 从经验池采样
        states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # 更新Critic
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * GAMMA * (target_q - self.log_alpha.exp() * next_log_probs)

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 更新Actor
        new_actions, log_probs, _ = self.actor.sample(states)
        q1, q2 = self.critic(states, new_actions)
        q = torch.min(q1, q2)

        actor_loss = (self.log_alpha.exp() * log_probs - q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新温度参数alpha
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # 软更新目标网络
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        return critic_loss.item(), actor_loss.item(), self.log_alpha.exp().item()


# 训练函数
def train_sac():
    env = gym.make('Pendulum-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    print(f"状态维度: {state_dim}")
    print(f"动作维度: {action_dim}")
    print(f"最大动作值: {max_action}")
    print("=" * 60)

    agent = SAC(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    max_episodes = 200
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

    total_steps = 0
    for episode in range(max_episodes):
        state, _ = env.reset(seed=SEED + episode)
        episode_reward = 0

        for step in range(max_steps):
            # 选择动作
            action = agent.select_action(state)

            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 存储经验
            replay_buffer.add(state, action, reward, next_state, float(done))

            # 训练
            if replay_buffer.size() >= BATCH_SIZE:
                critic_loss, actor_loss, alpha = agent.train(replay_buffer)

            state = next_state
            episode_reward += reward
            total_steps += 1

            if done:
                break

        reward_history.append(episode_reward)

        # 计算平均奖励
        avg_reward = np.mean(reward_history[-10:])

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{max_episodes}, Reward: {episode_reward:.2f}, "
                  f"Avg(10): {avg_reward:.2f}, Alpha: {alpha:.3f}")

    env.close()
    return agent, reward_history


# 测试函数
def test_sac(agent, episodes=10, render=False):
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
            # 使用确定性策略
            action = agent.select_action(state, deterministic=True)
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

    if avg_reward > -200:
        print("✓ 环境已解决！")
    else:
        print("✗ 还需要继续训练")

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
    plt.title('SAC Training on Pendulum-v1')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('sac_pendulum_training.png', dpi=150)
    print("\n训练曲线已保存为 'sac_pendulum_training.png'")
    plt.show()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SAC算法训练 - Pendulum-v1环境")
    print("=" * 60 + "\n")

    # 训练
    agent, reward_history = train_sac()

    print("\n" + "=" * 60)
    print("训练完成！开始测试...")
    print("=" * 60 + "\n")

    # 测试
    test_rewards = test_sac(agent, episodes=20, render=False)

    # 绘制训练曲线
    plot_rewards(reward_history)