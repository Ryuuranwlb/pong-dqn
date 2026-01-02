import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
from glob import glob
from collections import namedtuple, deque
from utils.process_obs_tool import ObsProcessTool

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义网络结构，输入为一张84*84的灰度图片，输出为各个动作的Q值，并采用2D卷积
class DQN(nn.Module):
    def __init__(self, state_size, action_size, skip_frame=4, horizon=4, clip=False, left=False, dueling=False):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(state_size[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        fc_input_dims = self.calculate_conv_output_dims(state_size)

        self.fc1 = nn.Linear(fc_input_dims, 512)
        self.dueling = dueling
        if self.dueling:
            self.value_head = nn.Linear(512, 1)
            self.advantage_head = nn.Linear(512, action_size)
        else:
            self.fc2 = nn.Linear(512, action_size)

        self.obs_process_tool = ObsProcessTool(skip_frame=skip_frame, horizon=horizon, clip=clip, flip=left)
        self.pre_action = 2


    def calculate_conv_output_dims(self, input_dims):
        state = torch.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, state):
        layer = F.relu(self.conv1(state))
        layer = F.relu(self.conv2(layer))
        layer = F.relu(self.conv3(layer))
        # conv3 shape = Batch Size X n_filters X H X W
        layer = layer.view(layer.size()[0], -1)
        layer = F.relu(self.fc1(layer))
        if self.dueling:
            advantage = self.advantage_head(layer)
            value = self.value_head(layer)
            advantage_mean = advantage.mean(dim=1, keepdim=True)
            layer = value + advantage - advantage_mean  # dueling aggregation: V(s)+A(s,a)-mean_a A(s,a)
        else:
            layer = self.fc2(layer)

        return layer
    
    def act(self, obs):
        code, state = self.obs_process_tool.process(obs)
        if code == -1:
            return self.pre_action
        else:
            state = torch.from_numpy(np.float32(state)).unsqueeze(0).to(device)
            with torch.no_grad():
                q_val = self.forward(state)
                act = q_val.max(1)[1].item()
            self.pre_action = act
            return act


# 定义代理类
class DQNAgent:
    def __init__(self, state_size, action_size, batch_size=64, gamma=0.99, lr=0.0001, memory_size=20000, skip_frame=4, horizon=4, clip=False, left=False, loss_type="mse", double=False, dueling=False, n_step: int = 1):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.memory_size = memory_size
        self.loss_type = loss_type.lower()
        self.double = bool(double)
        self.dueling = bool(dueling)
        self.n_step = int(n_step)
        if self.n_step < 1:
            raise ValueError("n_step must be >= 1")

        if not isinstance(self.double, bool): raise TypeError("double must be a boolean value")
        if not isinstance(self.dueling, bool): raise TypeError("dueling must be a boolean value")

        if self.loss_type == "mse":
            self.loss_fn = nn.MSELoss(reduction="mean")
        elif self.loss_type == "huber":
            self.loss_fn = nn.SmoothL1Loss(reduction="mean")
        else:
            raise ValueError("Unsupported loss_type: {}".format(loss_type))

        # 创建两个网络
        self.dqn_net = DQN(self.state_size, self.action_size, skip_frame=skip_frame, horizon=horizon, clip=clip, left=left, dueling=self.dueling).to(device)
        self.target_net = DQN(self.state_size, self.action_size, skip_frame=skip_frame, horizon=horizon, clip=clip, left=left, dueling=self.dueling).to(device)
        self.optimizer = optim.Adam(self.dqn_net.parameters(), lr=self.lr)

        # 创建记忆库
        self.memory = deque(maxlen=memory_size)
        # n-step buffer keeps decision-step transitions until they can be folded into an n-step return
        self.nstep_buffer = deque()

        self.epsilon_max = 1.0
        self.epsilon_min = 0.005
        self.epsilon_decay = 0.00001
        self.target_update_freq = 1000
        self.skip_frame = skip_frame
        self.gamma_decision = self.gamma ** self.skip_frame
        self.algo_variant = self._build_algo_variant()

    def set_left(self, left: bool):
        self.dqn_net.obs_process_tool.flip = left
        self.target_net.obs_process_tool.flip = left

    def select_action(self, state, eps):
        self.dqn_net.eval()
        if random.random() > eps:
            act = self.dqn_net.act(state)
        else:
            code, state = self.dqn_net.obs_process_tool.process(state)
            if code == -1:
                act = self.dqn_net.pre_action
            else:
                act = random.randrange(self.action_size)
                self.dqn_net.pre_action = act
        return act

    def memory_push(self, state, action, next_state, reward, done):
        """
        Combine incoming 1-step decision transitions into n-step transitions before storing.
        """
        self.nstep_buffer.append((state, action, next_state, reward, done))

        if done:
            # Episode ended: emit as many full n-step transitions as possible, then flush tail
            while len(self.nstep_buffer) >= self.n_step:
                self._append_nstep_transition()
            self._flush_nstep_buffer()
            return

        if len(self.nstep_buffer) < self.n_step:
            return

        # Normal path: buffer is full, emit the oldest n-step transition
        self._append_nstep_transition()

    def memory_sample(self, batch_size):
        idxs = np.random.choice(len(self.memory), batch_size, False)
        states, actions, next_states, rewards, dones = zip(*[self.memory[i] for i in idxs])
        return (np.array(states), np.array(actions), np.array(next_states),
                np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8))

    def update(self, step):
        if len(self.memory) < self.batch_size:
            return None

        self.dqn_net.train()
        # 更新target_net
        self.update_target_net(step)

        self.optimizer.zero_grad()

        # 从记忆库中随机采样
        states, actions, next_states, rewards, dones = self.memory_sample(self.batch_size)

        states = torch.from_numpy(np.float32(states)).to(device)
        actions = torch.from_numpy(actions).to(device)
        next_states = torch.from_numpy(np.float32(next_states)).to(device)
        rewards = torch.from_numpy(rewards).to(device)
        dones = torch.from_numpy(dones).to(device).float()


        q_vals = self.dqn_net(states)

        if actions.dtype != torch.int64:
            actions = actions.long()
        q_val = q_vals.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            target_next_q = self.target_net(next_states)
            if self.double:
                online_next_q = self.dqn_net(next_states)
                # Double DQN bootstrap: online argmax + target gather
                next_actions = online_next_q.argmax(dim=1)
                nxt_q_val = target_next_q.gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
            else:
                nxt_q_val = target_next_q.max(1)[0]  # DQN/Dueling: target max over actions

            gamma_decision = self.gamma_decision
            g_n = gamma_decision ** self.n_step
            exp_q_val = rewards + g_n * nxt_q_val * (1 - dones)

        loss = self.loss_fn(q_val, exp_q_val.detach())
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            td_error = (q_val - exp_q_val).detach().abs().cpu().numpy()
            q_max = q_vals.max(1)[0].detach().cpu().numpy()
            loss_item = loss.item()

        return {
            "loss": loss_item,
            "td_errors": td_error,
            "q_max": q_max,
        }


    def save_model(self, episode, step, path):
        eval_name = 'eval_checkpoint_ep{}_step{}.pth'.format(episode, step)
        target_name = 'target_checkpoint_ep{}_step{}.pth'.format(episode, step)
        torch.save(self.dqn_net.state_dict(), os.path.join(path, eval_name))
        torch.save(self.target_net.state_dict(), os.path.join(path, target_name))

    def load_model(self, step, path):
        legacy_eval = os.path.join(path, 'eval_checkpoint_{}.pth'.format(step))
        legacy_target = os.path.join(path, 'target_checkpoint_{}.pth'.format(step))

        if os.path.exists(legacy_eval) and os.path.exists(legacy_target):
            self.dqn_net.load_state_dict(torch.load(legacy_eval))
            self.target_net.load_state_dict(torch.load(legacy_target))
            return

        eval_matches = sorted(glob(os.path.join(path, 'eval_checkpoint_ep*_step{}.pth'.format(step))))
        target_matches = sorted(glob(os.path.join(path, 'target_checkpoint_ep*_step{}.pth'.format(step))))
        if not eval_matches or not target_matches:
            raise FileNotFoundError('checkpoint for step {} not found in {}'.format(step, path))

        self.dqn_net.load_state_dict(torch.load(eval_matches[-1]))
        self.target_net.load_state_dict(torch.load(target_matches[-1]))

    def update_target_net(self, step):
        if step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.dqn_net.state_dict())

    def update_epsilon(self, step):
        eps = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(-1 * ((step + 1) * self.epsilon_decay))
        return eps
        # self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
    
    def reset(self):
        self.dqn_net.obs_process_tool.reset()
        self.target_net.obs_process_tool.reset()
        self.nstep_buffer.clear()

    def _build_algo_variant(self):
        if self.double and self.dueling:
            return "D3QN"
        elif self.double:
            return "DoubleDQN"
        elif self.dueling:
            return "DuelingDQN"
        else:
            return "DQN"

    def _append_nstep_transition(self):
        """
        Build n-step return from the current buffer head and push into replay.
        """
        reward_sum, next_state, done_flag = self._compute_nstep_from_buffer(self.n_step)
        state, action, _, _, _ = self.nstep_buffer[0]
        self.memory.append((state, action, next_state, reward_sum, done_flag))
        self.nstep_buffer.popleft()

    def _flush_nstep_buffer(self):
        """
        Flush remaining buffered transitions at episode end; uses shorter k-step returns with done=True.
        """
        while len(self.nstep_buffer) > 0:
            k = len(self.nstep_buffer)
            reward_sum, next_state, _ = self._compute_nstep_from_buffer(k)
            state, action, _, _, _ = self.nstep_buffer[0]
            self.memory.append((state, action, next_state, reward_sum, True))
            self.nstep_buffer.popleft()

    def _compute_nstep_from_buffer(self, k):
        """
        Compute k-step return (k<=n_step) from the buffer head using decision-step discounting.
        """
        reward_sum = 0.0
        discount = 1.0
        for idx in range(k):
            _, _, _, rew, _ = self.nstep_buffer[idx]
            reward_sum += discount * rew
            discount *= self.gamma_decision
        next_state = self.nstep_buffer[k - 1][2]
        done_flag = self.nstep_buffer[k - 1][4]
        return reward_sum, next_state, done_flag
