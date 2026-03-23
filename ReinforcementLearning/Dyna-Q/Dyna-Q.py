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
LR_Q = 1e-3
LR_MODEL = 1e-3
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
N_PLANNING_STEPS = 10  # Dyna-Q的规划步数
UPDATE_TARGET_FREQ = 10  # 目标网络更新频率


# Q网络
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q


# 环境模型（预测下一状态和奖励）
class ModelNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ModelNetwork, self).__init__()
        # 预测下一状态
        self.state_fc1 = nn.Linear(state_dim + action_dim, 256)
        self.state_fc2 = nn.Linear(256, 256)
        self.state_fc3 = nn.Linear(256, state_dim)

        # 预测奖励
        self.reward_fc1 = nn.Linear(state_dim + action_dim, 256)
        self.reward_fc2 = nn.Linear(256, 256)
        self.reward_fc3 = nn.Linear(256, 1)

        # 预测是否终止
        self.done_fc1 = nn.Linear(state_dim + action_dim, 256)
        self.done_fc2 = nn.Linear(256, 256)
        self.done_fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)

        # 预测下一状态
        next_state = F.relu(self.state_fc1(x))
        next_state = F.relu(self.state_fc2(next_state))
        next_state = self.state_fc3(next_state)

        # 预测奖励
        reward = F.relu(self.reward_fc1(x))
        reward = F.relu(self.reward_fc2(reward))
        reward = self.reward_fc3(reward)

        # 预测终止
        done = F.relu(self.done_fc1(x))
        done = F.relu(self.done_fc2(done))
        done = torch.sigmoid(self.done_fc3(done))

        return next_state, reward, done


# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def size(self):
        return len(self.buffer)


# Dyna-Q算法
class DynaQ:
    def __init__(self, state_dim, action_dim, action_bound):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound

        # Q网络和目标Q网络
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.q_target = QNetwork(state_dim, action_dim).to(self.device)
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=LR_Q)

        # 环境模型
        self.model = ModelNetwork(state_dim, action_dim).to(self.device)
        self.model_optimizer = optim.Adam(self.model.parameters(), lr=LR_MODEL)

        self.epsilon = EPSILON_START
        self.update_counter = 0

        # 用于生成动作候选的离散化
        self.n_action_samples = 20  # 在连续空间中采样的动作数

    def select_action(self, state, deterministic=False):
        if not deterministic and random.random() < self.epsilon:
            # 随机探索
            action = np.random.uniform(-self.action_bound, self.action_bound, size=self.action_dim)
        else:
            # 贪婪策略：从多个候选动作中选择Q值最大的
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            # 生成动作候选
            action_samples = torch.FloatTensor(
                np.random.uniform(-self.action_bound, self.action_bound,
                                  size=(self.n_action_samples, self.action_dim))
            ).to(self.device)

            # 扩展状态以匹配动作样本数
            state_expanded = state_tensor.repeat(self.n_action_samples, 1)

            # 计算每个动作的Q值
            with torch.no_grad():
                q_values = self.q_net(state_expanded, action_samples)

            # 选择Q值最大的动作
            best_idx = q_values.argmax()
            action = action_samples[best_idx].cpu().numpy()

        return action

    def update_q_network(self, states, actions, rewards, next_states, dones):
        """更新Q网络"""
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # 计算当前Q值
        current_q = self.q_net(states, actions)

        # 计算目标Q值
        with torch.no_grad():
            # 为下一状态生成动作候选
            batch_size = next_states.shape[0]
            next_action_samples = torch.FloatTensor(
                np.random.uniform(-self.action_bound, self.action_bound,
                                  size=(batch_size, self.n_action_samples, self.action_dim))
            ).to(self.device)

            # 为每个下一状态找到最优动作
            max_next_q = []
            for i in range(batch_size):
                next_state_expanded = next_states[i].unsqueeze(0).repeat(self.n_action_samples, 1)
                q_values = self.q_target(next_state_expanded, next_action_samples[i])
                max_next_q.append(q_values.max())

            max_next_q = torch.stack(max_next_q).unsqueeze(1)
            target_q = rewards + (1 - dones) * GAMMA * max_next_q

        # 计算损失并更新
        q_loss = F.mse_loss(current_q, target_q)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.q_optimizer.step()

        return q_loss.item()

    def update_model(self, states, actions, rewards, next_states, dones):
        """更新环境模型"""
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # 预测
        pred_next_states, pred_rewards, pred_dones = self.model(states, actions)

        # 计算损失
        state_loss = F.mse_loss(pred_next_states, next_states)
        reward_loss = F.mse_loss(pred_rewards, rewards)
        done_loss = F.binary_cross_entropy(pred_dones, dones)

        model_loss = state_loss + reward_loss + done_loss

        self.model_optimizer.zero_grad()
        model_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.model_optimizer.step()

        return model_loss.item()

    def planning(self, replay_buffer, n_steps):
        """使用学习到的模型进行规划"""
        if replay_buffer.size() < BATCH_SIZE:
            return

        for _ in range(n_steps):
            # 从经验池随机采样状态和动作
            states, actions, _, _, _ = replay_buffer.sample(BATCH_SIZE)

            states = torch.FloatTensor(states).to(self.device)
            actions = torch.FloatTensor(actions).to(self.device)

            # 使用模型预测下一状态和奖励
            with torch.no_grad():
                pred_next_states, pred_rewards, pred_dones = self.model(states, actions)

            # 使用模拟经验更新Q网络
            self.update_q_network(
                states.cpu().numpy(),
                actions.cpu().numpy(),
                pred_rewards.squeeze().cpu().numpy(),
                pred_next_states.cpu().numpy(),
                pred_dones.squeeze().cpu().numpy()
            )

    def train(self, replay_buffer):
        """训练步骤"""
        if replay_buffer.size() < BATCH_SIZE:
            return 0, 0

        # 采样真实经验
        states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

        # 更新Q网络（真实经验）
        q_loss = self.update_q_network(states, actions, rewards, next_states, dones)

        # 更新环境模型
        model_loss = self.update_model(states, actions, rewards, next_states, dones)

        # Dyna-Q规划步骤（使用模型生成的模拟经验）
        self.planning(replay_buffer, N_PLANNING_STEPS)

        # 更新目标网络
        self.update_counter += 1
        if self.update_counter % UPDATE_TARGET_FREQ == 0:
            self.q_target.load_state_dict(self.q_net.state_dict())

        # 衰减epsilon
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

        return q_loss, model_loss


# 训练函数
def train_dynaq():
    env = gym.make('Pendulum-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = float(env.action_space.high[0])

    print(f"状态维度: {state_dim}")
    print(f"动作维度: {action_dim}")
    print(f"动作范围: [-{action_bound}, {action_bound}]")
    print(f"规划步数: {N_PLANNING_STEPS}")
    print("=" * 60)

    agent = DynaQ(state_dim, action_dim, action_bound)
    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    max_episodes = 300
    max_steps = 200
    reward_history = []

    print("开始训练...\n")

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

            # 训练（包括真实学习和模型规划）
            if replay_buffer.size() >= BATCH_SIZE:
                q_loss, model_loss = agent.train(replay_buffer)

            state = next_state
            episode_reward += reward

            if done:
                break

        reward_history.append(episode_reward)

        # 计算平均奖励
        avg_reward = np.mean(reward_history[-20:]) if len(reward_history) >= 20 else np.mean(reward_history)

        if (episode + 1) % 20 == 0:
            print(f"Episode {episode + 1}/{max_episodes}, Reward: {episode_reward:.2f}, "
                  f"Avg(20): {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    env.close()
    return agent, reward_history


# 测试函数
def test_dynaq(agent, episodes=10, render=False):
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
            # 使用确定性策略（贪婪）
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
    window = 20
    if len(reward_history) >= window:
        moving_avg = np.convolve(reward_history, np.ones(window) / window, mode='valid')
        plt.plot(range(window - 1, len(reward_history)), moving_avg,
                 linewidth=2, label=f'Moving Average ({window})')

    plt.axhline(y=-200, color='r', linestyle='--', label='Solved Threshold')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Dyna-Q Training on Pendulum-v1')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('dynaq_pendulum_training.png', dpi=150)
    print("\n训练曲线已保存为 'dynaq_pendulum_training.png'")
    plt.show()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Dyna-Q算法训练 - Pendulum-v1环境")
    print("=" * 60 + "\n")

    # 训练
    agent, reward_history = train_dynaq()

    print("\n" + "=" * 60)
    print("训练完成！开始测试...")
    print("=" * 60 + "\n")

    # 测试
    test_rewards = test_dynaq(agent, episodes=20, render=False)

    # 绘制训练曲线
    plot_rewards(reward_history)