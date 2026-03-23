import torch
import minari
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import deque
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# 1. 增强型配置类（带维度校验）
class SafeConfig:
    # 训练参数
    batch_size = 1024
    lr = 3e-5
    tau = 0.007
    gamma = 0.99
    total_epochs = 500

    # 网络架构
    hidden_dim = 768
    num_layers = 3
    dropout_rate = 0.1
    activation_fn = 'Mish'  # 支持Mish/SiLU/ReLU

    # 正则化参数
    conservative_init = 2.5
    conservative_decay = 0.995
    min_conservative = 0.3
    reward_scale = 4.0

    # 探索参数
    noise_scale = 0.2
    noise_clip = 0.5
    candidate_samples = 400
    imitation_ratio = 0.15



# 2. 安全数据加载系统
class SafeDataset(Dataset):
    def __init__(self, dataset_name):
        # 加载原始数据
        dataset = minari.load_dataset(dataset_name, download=True)

        # 获取维度信息
        first_ep = dataset[0]
        self.state_dim = first_ep.observations[0].shape[0]
        self.action_dim = first_ep.actions[0].shape[0]

        # 数据存储
        self.obs, self.acts, self.rews, self.dones, self.next_obs = [], [], [], [], []
        for ep in dataset:
            self._store_episode(
                ep.observations[:-1],
                ep.actions,
                ep.rewards,
                np.logical_or(ep.terminations, ep.truncations),
                ep.observations[1:]
            )

        # 标准化
        self._normalize()
        self.priorities = np.ones(len(self.obs)) * 1e-5

    def _store_episode(self, obs, acts, rews, dones, next_obs):
        self.obs.extend(obs)
        self.acts.extend(acts)
        self.rews.extend(rews)
        self.dones.extend(dones)
        self.next_obs.extend(next_obs)

    def _normalize(self):
        # 状态标准化
        self.obs_mean = np.mean(self.obs, axis=0)
        self.obs_std = np.std(self.obs, axis=0) + 1e-8
        self.obs = (self.obs - self.obs_mean) / self.obs_std
        self.next_obs = (self.next_obs - self.obs_mean) / self.obs_std

        # 动作标准化
        self.act_mean = np.mean(self.acts, axis=0)
        self.act_std = np.std(self.acts, axis=0) + 1e-8
        self.acts = (self.acts - self.act_mean) / self.act_std

    def update_priorities(self, indices, priorities):
        self.priorities[indices] = np.abs(priorities.flatten()) + 1e-5

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return (
            idx,
            torch.FloatTensor(self.obs[idx]),
            torch.FloatTensor(self.acts[idx]),
            torch.FloatTensor(self.next_obs[idx]),
            torch.FloatTensor([self.rews[idx]]),
            torch.FloatTensor([bool(self.dones[idx])])
        )



# 3. 维度安全网络架构
class SafeQNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.input_dim = state_dim + action_dim  # 关键动态计算

        # 主网络
        self.feature_net = self._build_network()
        self.q1 = torch.nn.Linear(SafeConfig.hidden_dim, 1)
        self.q2 = torch.nn.Linear(SafeConfig.hidden_dim, 1)

        # 目标网络
        self.target_net = self._build_network()
        self.target_q1 = torch.nn.Linear(SafeConfig.hidden_dim, 1)
        self.target_q2 = torch.nn.Linear(SafeConfig.hidden_dim, 1)

        # 初始化
        self._init_weights()
        self._update_target(1.0)

    def _build_network(self):
        layers = []
        input_dim = self.input_dim  # 使用动态计算值
        for _ in range(SafeConfig.num_layers):
            layers.extend([
                torch.nn.Linear(input_dim, SafeConfig.hidden_dim),
                torch.nn.LayerNorm(SafeConfig.hidden_dim),
                self._activation(),
                torch.nn.Dropout(SafeConfig.dropout_rate),
            ])
            input_dim = SafeConfig.hidden_dim
        return torch.nn.Sequential(*layers)

    def _activation(self):
        return {
            'Mish': torch.nn.Mish(),
            'SiLU': torch.nn.SiLU(),
            'ReLU': torch.nn.ReLU()
        }[SafeConfig.activation_fn]

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight)
                torch.nn.init.normal_(m.bias, 0, 0.1)

    def forward(self, state, action):
        # 维度校验
        assert state.shape[-1] == self.state_dim, f"State dim error: {state.shape[-1]} vs {self.state_dim}"
        assert action.shape[-1] == self.action_dim, f"Action dim error: {action.shape[-1]} vs {self.action_dim}"

        x = torch.cat([state, action], dim=1)
        features = self.feature_net(x)
        return self.q1(features), self.q2(features)

    def target_forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        features = self.target_net(x)
        return self.target_q1(features), self.target_q2(features)

    def _update_target(self, tau):
        with torch.no_grad():
            for t_param, param in zip(self.target_net.parameters(), self.feature_net.parameters()):
                t_param.data.copy_(tau * param.data + (1 - tau) * t_param.data)
            for t_param, param in zip(self.target_q1.parameters(), self.q1.parameters()):
                t_param.data.copy_(tau * param.data + (1 - tau) * t_param.data)
            for t_param, param in zip(self.target_q2.parameters(), self.q2.parameters()):
                t_param.data.copy_(tau * param.data + (1 - tau) * t_param.data)



# 4. 安全训练系统
class SafeTrainer:
    def __init__(self, dataset_name):
        # 数据系统
        self.dataset = SafeDataset(dataset_name)
        self.state_dim = self.dataset.state_dim
        self.action_dim = self.dataset.action_dim

        # 网络系统
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = SafeQNetwork(self.state_dim, self.action_dim).to(self.device)

        # 优化系统
        self.optimizer = torch.optim.AdamW(
            self.q_net.parameters(),
            lr=SafeConfig.lr,
            weight_decay=1e-3
        )
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=100,
            eta_min=1e-6
        )

        # 数据加载
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=SafeConfig.batch_size,
            sampler=WeightedRandomSampler(
                self.dataset.priorities,
                num_samples=len(self.dataset),
                replacement=True
            ),
            collate_fn=lambda b: {
                'indices': torch.LongTensor([x[0] for x in b]),
                'states': torch.stack([x[1] for x in b]),
                'actions': torch.stack([x[2] for x in b]),
                'next_states': torch.stack([x[3] for x in b]),
                'rewards': torch.stack([x[4] for x in b]),
                'dones': torch.stack([x[5] for x in b])
            },
            num_workers=0
        )

        # 训练状态
        self.conservative_weight = SafeConfig.conservative_init
        self.loss_history = deque(maxlen=100)

    def train_epoch(self, epoch):
        self.q_net.train()
        total_loss = 0.0

        for batch in self.dataloader:
            # 数据准备
            states = batch['states'].to(self.device)
            actions = batch['actions'].to(self.device)
            next_states = batch['next_states'].to(self.device)
            rewards = batch['rewards'].to(self.device) * SafeConfig.reward_scale
            dones = batch['dones'].to(self.device)

            # 目标Q值计算
            with torch.no_grad():
                # 带噪声的动作生成
                noise = torch.randn_like(actions) * SafeConfig.noise_scale
                noise = torch.clamp(noise, -SafeConfig.noise_clip, SafeConfig.noise_clip)
                noisy_actions = actions + noise

                # 双Q学习
                target_q1, target_q2 = self.q_net.target_forward(next_states, noisy_actions)
                target_q = torch.min(target_q1, target_q2).squeeze(-1)
                y = rewards.squeeze(-1) + (1 - dones.squeeze(-1)) * SafeConfig.gamma * target_q

            # 当前Q值预测
            current_q1, current_q2 = self.q_net(states, actions)
            current_q1 = current_q1.squeeze(-1).clamp(-10.0, 50.0)
            current_q2 = current_q2.squeeze(-1).clamp(-10.0, 50.0)

            # 损失计算
            bellman_loss = 0.5 * (
                    torch.nn.functional.huber_loss(current_q1, y, delta=1.0) +
                    torch.nn.functional.huber_loss(current_q2, y, delta=1.0)
            )

            # 保守正则项
            rand_acts = torch.randn_like(actions) * SafeConfig.noise_scale
            q1_rand, q2_rand = self.q_net(states, rand_acts)
            conservative_loss = (q1_rand + q2_rand).mean() - (current_q1 + current_q2).mean()

            # 总损失
            loss = bellman_loss + self.conservative_weight * conservative_loss

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 2.0)
            self.optimizer.step()

            # 更新目标网络
            self.q_net._update_target(SafeConfig.tau)

            # 更新优先级
            td_errors = (current_q1 - y).detach().cpu().numpy()
            self.dataset.update_priorities(batch['indices'].numpy(), td_errors)

            total_loss += loss.item()

        # 调整保守权重
        self.conservative_weight = max(
            self.conservative_weight * SafeConfig.conservative_decay,
            SafeConfig.min_conservative
        )

        # 学习率调度
        self.scheduler.step()

        return total_loss / len(self.dataloader)

    def get_action(self, state):
        self.q_net.eval()
        state_norm = (state - self.dataset.obs_mean) / self.dataset.obs_std
        state_tensor = torch.FloatTensor(state_norm).unsqueeze(0).to(self.device)

        # 候选动作生成
        num_imitation = int(SafeConfig.candidate_samples * SafeConfig.imitation_ratio)
        imitation_idx = np.random.choice(len(self.dataset), num_imitation)
        imitation_acts = self.dataset.acts[imitation_idx]
        noise_acts = np.random.randn(SafeConfig.candidate_samples - num_imitation, self.action_dim)
        candidates = np.concatenate([imitation_acts, noise_acts])
        candidates = (candidates * self.dataset.act_std) + self.dataset.act_mean

        # 选择最优动作
        with torch.no_grad():
            state_batch = state_tensor.repeat(SafeConfig.candidate_samples, 1)
            candidate_tensor = torch.FloatTensor(candidates).to(self.device)
            candidate_norm = (candidate_tensor - self.dataset.act_mean) / self.dataset.act_std
            q_values, _ = self.q_net(state_batch, candidate_norm)
            best_idx = torch.argmax(q_values)

        return candidates[best_idx.cpu().item()]


# 5. 训练执行
if __name__ == "__main__":
    trainer = SafeTrainer("D4RL/door/human-v2")
    print(f"初始化维度检查: state={trainer.state_dim}, action={trainer.action_dim}")

    try:
        for epoch in range(SafeConfig.total_epochs):
            loss = trainer.train_epoch(epoch)

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1:04d} | Loss: {loss:.2f} | "
                      f"Conserv: {trainer.conservative_weight:.2f} | "
                      f"LR: {trainer.scheduler.get_last_lr()[0]:.1e}")

    except KeyboardInterrupt:
        print("\n训练中断，保存检查点...")
        torch.save(trainer.q_net.state_dict(), "interrupted.pth")

    print("训练完成...")
