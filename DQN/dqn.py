import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
from collections import namedtuple, deque
from utils.process_obs_tool import ObsProcessTool

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义网络结构，输入为一张84*84的灰度图片，输出为各个动作的Q值，并采用2D卷积
class DQN(nn.Module):
    def __init__(self, state_size, action_size, skip_frame=4, horizon=4, clip=False, left=False):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(state_size[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        fc_input_dims = self.calculate_conv_output_dims(state_size)

        self.fc1 = nn.Linear(fc_input_dims, 512)
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
        layer = self.fc2(layer)

        return layer
    
    def act(self, obs):
        code, state = self.obs_process_tool.process(obs)
        if code == -1:
            return self.pre_action
        else:
            state = torch.from_numpy(np.float32(state)).unsqueeze(0).to(device)
            q_val = self.forward(state)
            act = q_val.max(1)[1].item()
            self.pre_action = act
            return act


# 定义代理类
class DQNAgent:
    def __init__(self, state_size, action_size, batch_size=64, gamma=0.99, lr=0.0001, memory_size=20000, skip_frame=4, horizon=4, clip=False, left=False):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr

        # 创建两个网络
        self.dqn_net = DQN(self.state_size, self.action_size, skip_frame=skip_frame, horizon=horizon, clip=clip, left=left).to(device)
        self.target_net = DQN(self.state_size, self.action_size, skip_frame=skip_frame, horizon=horizon, clip=clip, left=left).to(device)
        self.optimizer = optim.Adam(self.dqn_net.parameters(), lr=self.lr)

        # 创建记忆库
        self.memory = deque(maxlen=memory_size)

        self.epsilon_max = 1.0
        self.epsilon_min = 0.005
        self.epsilon_decay = 0.00001
        self.set_flip(left, reset=False)

    def set_flip(self, flip: bool, reset: bool = True):
        self.dqn_net.obs_process_tool.flip = bool(flip)
        self.target_net.obs_process_tool.flip = bool(flip)
        if reset:
            self.reset()

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
        # e = self.experience(state, action, next_state, reward, done)
        self.memory.append((state, action, next_state, reward, done))

    def memory_sample(self, batch_size):
        idxs = np.random.choice(len(self.memory), batch_size, False)
        states, actions, next_states, rewards, dones = zip(*[self.memory[i] for i in idxs])
        return (np.array(states), np.array(actions), np.array(next_states),
                np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8))

    def update(self, step):
        if len(self.memory) < self.batch_size:
            return

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
        dones = torch.from_numpy(dones).to(device)

        q_vals = self.dqn_net(states)
        nxt_q_vals = self.target_net(next_states)

        if actions.dtype != torch.int64:
            actions = actions.long()
        # print(actions.dtype)
        # exit()
        q_val = q_vals.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        nxt_q_val = nxt_q_vals.max(1)[0]
        exp_q_val = rewards + self.gamma * nxt_q_val * (1 - dones)

        loss = (q_val - exp_q_val.data.to(device)).pow(2).mean()
        loss.backward()
        self.optimizer.step()


    def save_model(self, episode, path):
        torch.save(self.dqn_net.state_dict(), os.path.join(path, 'eval_checkpoint_{}.pth'.format(episode)))
        torch.save(self.target_net.state_dict(), os.path.join(path, 'target_checkpoint_{}.pth'.format(episode)))

    def load_model(self, episode, path):
        self.dqn_net.load_state_dict(torch.load(os.path.join(path, 'eval_checkpoint_{}.pth'.format(episode))))
        self.target_net.load_state_dict(torch.load(os.path.join(path, 'target_checkpoint_{}.pth'.format(episode))))

    def update_target_net(self, step):
        if step % 1000 == 0:
            self.target_net.load_state_dict(self.dqn_net.state_dict())

    def update_epsilon(self, step):
        eps = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(-1 * ((step + 1) * self.epsilon_decay))
        return eps
        # self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
    
    def reset(self):
        self.dqn_net.obs_process_tool.reset()
        self.target_net.obs_process_tool.reset()
