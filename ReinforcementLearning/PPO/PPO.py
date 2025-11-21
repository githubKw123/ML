import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from collections import deque
import matplotlib.pyplot as plt

# 设置随机种子
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# 超参数（针对Pendulum优化）
GAMMA = 0.99
LAMBDA = 0.95  # GAE参数
EPSILON = 0.2  # PPO裁剪参数
ENTROPY_COEF = 0.001  # 降低熵系数，减少随机性
VALUE_COEF = 0.5  # 价值函数损失系数
LR_ACTOR = 3e-4  # Actor学习率
LR_CRITIC = 1e-3  # Critic学习率（提高）
EPOCHS = 20  # 增加训练轮数
BATCH_SIZE = 64
BUFFER_SIZE = 2048  # 收集经验的步数
MAX_GRAD_NORM = 0.5  # 梯度裁剪
FIXED_STD = 0.5  # 固定的标准差


# Actor网络（策略网络）- 固定方差版本
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.mean = nn.Linear(64, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = 2.0 * torch.tanh(self.mean(x))  # 缩放到[-2, 2]
        return mean


# Critic网络（价值网络）- 改进版
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.value = nn.Linear(64, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        value = self.value(x)
        return value


# PPO算法
class PPO:
    def __init__(self, state_dim, action_dim, action_bound):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.action_bound = action_bound

        # 存储经验
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            mean = self.actor(state)

            if deterministic:
                action = mean
            else:
                # 使用固定的标准差
                std = torch.ones_like(mean) * FIXED_STD
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)

                self.log_probs.append(log_prob.cpu().numpy()[0])

            value = self.critic(state)
            self.values.append(value.cpu().numpy()[0, 0])

        # 动作已经在网络中缩放到[-2, 2]，直接使用
        action = action.cpu().numpy()[0]
        action = np.clip(action, -self.action_bound, self.action_bound)

        return action

    def store_transition(self, state, action, reward, done):
        self.states.append(state)
        self.actions.append(action)  # 直接存储原始动作
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_gae(self, next_value):
        """计算广义优势估计(GAE)"""
        advantages = []
        returns = []
        gae = 0

        # 从后往前计算
        values = self.values + [next_value]
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + GAMMA * values[step + 1] * (1 - self.dones[step]) - values[step]
            gae = delta + GAMMA * LAMBDA * (1 - self.dones[step]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])

        return advantages, returns

    def update(self, next_state):
        # 计算下一个状态的价值
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            next_value = self.critic(next_state).cpu().numpy()[0, 0]

        # 计算优势和回报
        advantages, returns = self.compute_gae(next_value)

        # 转换为张量
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        advantages = torch.FloatTensor(np.array(advantages)).to(self.device)
        returns = torch.FloatTensor(np.array(returns)).to(self.device)

        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO更新
        dataset_size = len(self.states)
        indices = np.arange(dataset_size)

        for _ in range(EPOCHS):
            np.random.shuffle(indices)

            for start in range(0, dataset_size, BATCH_SIZE):
                end = start + BATCH_SIZE
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # 计算新的动作分布（使用固定标准差）
                mean = self.actor(batch_states)
                std = torch.ones_like(mean) * FIXED_STD
                dist = torch.distributions.Normal(mean, std)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                # 计算比率和裁剪目标
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - EPSILON, 1 + EPSILON) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - ENTROPY_COEF * entropy

                # 更新Actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD_NORM)
                self.actor_optimizer.step()

                # 计算价值损失
                values = self.critic(batch_states).squeeze()
                critic_loss = nn.MSELoss()(values, batch_returns)

                # 更新Critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM)
                self.critic_optimizer.step()

        # 清空缓冲区
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()


# 训练函数
def train_ppo():
    env = gym.make('Pendulum-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = float(env.action_space.high[0])

    print(f"状态维度: {state_dim}")
    print(f"动作维度: {action_dim}")
    print(f"动作范围: [-{action_bound}, {action_bound}]")
    print("=" * 60)

    agent = PPO(state_dim, action_dim, action_bound)

    max_episodes = 3000
    max_steps = 200
    reward_history = []
    update_counter = 0

    state, _ = env.reset(seed=SEED)
    episode_reward = 0
    episode = 0

    print("开始训练...\n")

    for total_steps in range(1, max_episodes * max_steps + 1):
        # 选择动作
        action = agent.select_action(state)

        # 执行动作
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # 存储转换
        agent.store_transition(state, action, reward, done)

        episode_reward += reward
        state = next_state

        # 更新策略
        if total_steps % BUFFER_SIZE == 0:
            agent.update(next_state)
            update_counter += 1
            avg_reward = np.mean(reward_history[-20:]) if len(reward_history) >= 20 else (
                np.mean(reward_history) if reward_history else 0)
            print(f"Update {update_counter}, Total Steps: {total_steps}, Avg Reward(20): {avg_reward:.2f}")

        # Episode结束
        if done:
            reward_history.append(episode_reward)
            episode += 1

            # 计算平均奖励
            avg_reward = np.mean(reward_history[-20:]) if len(reward_history) >= 20 else np.mean(reward_history)

            if episode % 20 == 0:
                print(f"  Episode {episode}, Reward: {episode_reward:.2f}, Avg(20): {avg_reward:.2f}")

            # 重置环境
            state, _ = env.reset(seed=SEED + episode)
            episode_reward = 0

            if episode >= max_episodes:
                break

    env.close()
    return agent, reward_history


# 测试函数
def test_ppo(agent, episodes=10, render=False):
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
    plt.title('PPO Training on Pendulum-v1')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ppo_pendulum_training.png', dpi=150)
    print("\n训练曲线已保存为 'ppo_pendulum_training.png'")
    plt.show()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PPO算法训练 - Pendulum-v1环境")
    print("=" * 60 + "\n")

    # 训练
    agent, reward_history = train_ppo()

    print("\n" + "=" * 60)
    print("训练完成！开始测试...")
    print("=" * 60 + "\n")

    # 测试
    test_rewards = test_ppo(agent, episodes=20, render=False)

    # 绘制训练曲线
    plot_rewards(reward_history)