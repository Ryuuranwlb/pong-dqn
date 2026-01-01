import retro
import matplotlib.pyplot as plt
import gym
from IPython import display
import gym.spaces
import numpy as np
import cv2
from tqdm import tqdm
import argparse
import os
output_dir = 'videos/test/'
os.makedirs(output_dir, exist_ok=True)

from config import *

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from gym import make, ObservationWrapper, Wrapper
from gym.spaces import Box
from collections import deque
from utils.process_obs_tool import ObsProcessTool

import signal


CONFIG = {
    'model_dir': 'checkpoints',
    'video_dir': 'videos',
}


total_rews = []
steps_list = []
eps_list = []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-test_mode", action="store_true", default=False)
    parser.add_argument("-memo_1", type=str, default='teacher_original')
    parser.add_argument("-memo_2", type=str, default='test')

    parser.add_argument("-agent_1", type=str, default='DQN')
    parser.add_argument("-agent_2", type=str, default='DQN')

    parser.add_argument("-start_episode_1", type=int, default=0)
    parser.add_argument("-start_episode_2", type=int, default=0)
    parser.add_argument("-total_episode", type=int, default=400)

    parser.add_argument("-horizon", type=int, default=4)
    parser.add_argument("-player", type=int, default=1)
    parser.add_argument("-skip_frame", type=int, default=4)
    
    return parser.parse_args()


def PongDiscretizer(env, players=1):
    """
    Discretize Retro Pong-Atari2600 environment
    """
    return Discretizer(env, buttons=env.unwrapped.buttons, combos=[['DOWN'], ['UP'], ['BUTTON'],], players=players)
    

class Discretizer(gym.ActionWrapper):
    """
    Wrap a gym environment and make it use discrete actions.
    based on https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
    Args:
        buttons: ordered list of buttons, corresponding to each dimension of the MultiBinary action space
        combos: ordered list of lists of valid button combinations
    """

    def __init__(self, env, buttons, combos, players=1):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)

        self.players = players
        self._decode_discrete_action = []
        self._decode_discrete_action2 = []
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        if self.players == 2:
            # pkayer 2 : 7: DOWN, 6: 'UP', 15:'BUTTOM'
            arr = np.array([False] * env.action_space.n)
            arr[7] = True
            self._decode_discrete_action2.append(arr)
            
            arr = np.array([False] * env.action_space.n)
            arr[6] = True
            self._decode_discrete_action2.append(arr)
            
            arr = np.array([False] * env.action_space.n)
            arr[15] = True
            self._decode_discrete_action2.append(arr)
        
        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def action(self, act1, act2):
        act1_v = self._decode_discrete_action[act1].copy()
        if self.players == 1:
            return act1_v.copy()
        else:
            act2_v = self._decode_discrete_action2[act2].copy()
            return np.logical_or(act1_v, act2_v).copy()
    
    def step(self, act1, act2=None):
        return self.env.step(self.action(act1, act2))


def traverse_imgs(writer, images):
    # 遍历所有图片，并且让writer抓取视频帧
    with tqdm(total=len(images), desc='traverse_imgs', leave=False) as pbar:
        for img in images:
            plt.imshow(img)
            writer.grab_frame()
            plt.pause(0.01)
            plt.clf()
            pbar.update(1)
        plt.close()


def plot_learning_curve(x, scores, epsilon, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, label='1')
    ax2 = fig.add_subplot(111, label='2', frame_on=False)

    ax.plot(x, epsilon, color='C0')
    ax.set_xlabel('Training Steps', color='C0')
    ax.set_ylabel('Epsilon', color='C0')
    ax.tick_params(axis='x', colors='C0')
    ax.tick_params(axis='y', colors='C0')

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-100):(t+1)])

    ax2.scatter(x,running_avg, color='C1')
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color='C1')
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors='C1')

    plt.savefig(filename)


def train(agent_1, agent_2=None, players=1, skip_frame=2, horizon=2, max_steps=2500, start_episode=0, total_episode=1000):
    global CONFIG

    env = PongDiscretizer(retro.make(game='Pong-Atari2600', players=players), players=players)
    env.reset()

    global total_rews
    global steps_list
    global eps_list
    best_avg_rew = -np.inf
    best_rew = -np.inf
    pre_action_1 = 2
    steps = 0

    for i in range(start_episode, total_episode):
        done = False
        total_rew = 0.0
        skip_rew = 0.0
        obs = env.reset()
        
        if players == 2:
            agent_1.reset()
            agent_2.reset()
        else:
            agent_1.reset()

        while not done:
            if players == 2 and (obs == 236).sum() < 12504:
                env.step(2, 2)

            # 更新epsilon
            eps = agent_1.update_epsilon(steps)

            # 右侧板
            action_1 = agent_1.select_action(obs, eps)

            # 左侧板
            if players == 2 and agent_2 is None:
                raise ValueError("agent_2 is None")
            if players == 2 and agent_2 is not None:
                action_2 = agent_2.select_action(obs, eps=0.0)
            else:
                action_2 = None

            # 训左侧板暂时修改
            nxt_obs, rew, done, info = env.step(action_1, action_2)
            
            obs = nxt_obs
            if players == 2:
                rew = rew[0]
            skip_rew += rew
            total_rew += rew

            if agent_1.dqn_net.obs_process_tool.frame_cnt == 0:
                agent_1.memory_push(agent_1.dqn_net.obs_process_tool.pre_obs, pre_action_1, agent_1.dqn_net.obs_process_tool.obs, skip_rew, done)
                agent_1.update(steps)
                pre_action_1 = action_1
                steps += 1
                skip_rew = 0.0
            

        total_rews.append(total_rew)
        steps_list.append(steps)
        eps_list.append(eps)

        if total_rew > best_rew:
            best_rew = total_rew

        avg_total_rew = np.mean(total_rews[-100:])

        if avg_total_rew > best_avg_rew:
            best_avg_rew = avg_total_rew
            agent_1.save_model(i, CONFIG['model_dir'])
            # if args.player == 2:
            #     agent_2.save_model(i, CONFIG['model_dir'])

        print('episode: %d, total step = %d, total reward = %.2f, avg reward = %.6f, best reward = %.2f, best avg reward = %.6f, epsilon = %.6f' % (i, steps, total_rew, avg_total_rew, best_rew, best_avg_rew, eps))

        if i % 25 == 0:
            # 测试agent
            test(agent_1, agent_2, players=players, skip_frame=skip_frame, horizon=horizon, max_steps=max_steps, episode=i, env=env)

    plot_learning_curve(steps_list, total_rews, eps_list, os.path.join(CONFIG['model_dir'], 'pong.png'))


def test(agent_1, agent_2=None, players=1, skip_frame=2, horizon=2, max_steps=2500, episode=0, env=None):
    global CONFIG

    if env is None:
        env = PongDiscretizer(retro.make(game='Pong-Atari2600', players=players), players=players)
        env.reset()

    done = False
    steps = 0
    images = []
    obs = env.reset()

    if players == 2:
        agent_1.reset()
        agent_2.reset()
    else:
        agent_1.reset()

    while not done:
        if players == 2 and (obs == 236).sum() < 12504:
            env.step(2, 2)

        # 右侧板
        action_1 = agent_1.select_action(obs, eps=0.0)

        # 左侧板
        if players == 2 and agent_2 is None:
            raise ValueError("agent_2 is None")
        if players == 2 and agent_2 is not None:
            action_2 = agent_2.select_action(obs, eps=0.0)
        else:
            action_2 = None

        nxt_obs, rew, done, info = env.step(action_1, action_2)

        obs = nxt_obs
            
        if agent_1.dqn_net.obs_process_tool.frame_cnt == 0:
            steps += 1
            if steps % 8 == 0:
                images.append(env.render(mode='rgb_array'))

        if steps > max_steps:
            break


    # 创建video writer, 设置好相应参数，fps
    metadata = dict(title='01', artist='Matplotlib',comment='depth prediiton')
    writer = FFMpegWriter(fps=10, metadata=metadata)

    figure = plt.figure(figsize=(10.8, 7.2))
    plt.ion()                                   # 为了可以动态显示
    plt.tight_layout()                          # 尽量减少窗口的留白
    with writer.saving(figure, os.path.join(CONFIG['video_dir'], 'episode_%d.mp4' % episode), 100): 
        traverse_imgs(writer, images)

    return info


def main(args):
    global CONFIG
    # 检查目录是否存在，如果不存在则创建，存在则停止运行，test_mode不需要创建

    if args.player == 2 and not os.path.exists(os.path.join(CONFIG['model_dir'], args.memo_2)):
        raise ValueError("agent_2 model is not exists")

    if args.player == 1:
        CONFIG['model_dir'] = os.path.join(CONFIG['model_dir'], args.memo_1)
        CONFIG['video_dir'] = os.path.join(CONFIG['video_dir'], args.memo_1)

    if not args.test_mode and args.memo_1 != 'test' and args.player != 2 and os.path.exists(CONFIG['model_dir']):
        raise ValueError("memo is already exists")

    if args.player == 1:
        agent_1 = AGENT[args.agent_1](state_size=(args.horizon, 84, 84), action_size=3, skip_frame=args.skip_frame, horizon=args.horizon, clip=False, left=False)
        agent_2 = None
    elif args.player == 2:
        agent_1 = AGENT[args.agent_1](state_size=(args.horizon, 84, 84), action_size=3, skip_frame=args.skip_frame, horizon=args.horizon, clip=False, left=False)
        agent_2 = AGENT[args.agent_2](state_size=(args.horizon, 84, 84), action_size=3, skip_frame=args.skip_frame, horizon=args.horizon, clip=False, left=False)

    if args.test_mode:
        if args.player == 2:
            agent_1.load_model(args.start_episode_1, os.path.join(CONFIG['model_dir'], args.memo_1))
            agent_2.load_model(args.start_episode_2, os.path.join(CONFIG['model_dir'], args.memo_2))

            CONFIG['model_dir'] = os.path.join(CONFIG['model_dir'], args.memo_1 + '_' + args.memo_2)
            CONFIG['video_dir'] = os.path.join(CONFIG['video_dir'], args.memo_1 + '_' + args.memo_2)

            if not os.path.exists(CONFIG['model_dir']):
                os.makedirs(CONFIG['model_dir'])
            if not os.path.exists(CONFIG['video_dir']):
                os.makedirs(CONFIG['video_dir'])
        else:
            if not os.path.exists(CONFIG['model_dir']):
                raise ValueError("model dir is not exists")
            if not os.path.exists(CONFIG['video_dir']):
                raise ValueError("video dir is not exists")
            
            agent_1.load_model(args.start_episode_1, CONFIG['model_dir'])
        
        # 测试agent
        info = test(agent_1, agent_2, players=args.player, skip_frame=args.skip_frame, horizon=args.horizon, max_steps=2500, episode=args.start_episode_1)
        print(info)
    else:
        if args.player == 2:
            agent_1.load_model(args.start_episode_1, os.path.join(CONFIG['model_dir'], args.memo_1))
            agent_2.load_model(args.start_episode_2, os.path.join(CONFIG['model_dir'], args.memo_2))


            CONFIG['model_dir'] = os.path.join(CONFIG['model_dir'], args.memo_1 + '_' + args.memo_2)
            CONFIG['video_dir'] = os.path.join(CONFIG['video_dir'], args.memo_1 + '_' + args.memo_2)

            if not os.path.exists(CONFIG['model_dir']):
                os.makedirs(CONFIG['model_dir'])
            if not os.path.exists(CONFIG['video_dir']):
                os.makedirs(CONFIG['video_dir'])

        else:
            if args.memo_1 != 'test':
                if os.path.exists(CONFIG['model_dir']):
                    agent_1.load_model(args.start_episode_1 - 1, CONFIG['model_dir'])
                else:
                    os.makedirs(CONFIG['model_dir'])

                if os.path.exists(CONFIG['video_dir']):
                    os.makedirs(CONFIG['video_dir'])

        # 训练agent
        steps_list, total_rews, eps_list, data = train(agent_1, agent_2, players=args.player, skip_frame=args.skip_frame, horizon=args.horizon, max_steps=2500, start_episode=args.start_episode_1, total_episode=args.total_episode)

        return steps_list, total_rews, eps_list, data


def int_handler(signum, frame):
    plot_learning_curve(steps_list, total_rews, eps_list, os.path.join(CONFIG['model_dir'], 'pong.png'))
    exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, int_handler)

    args = parse_args()    
    main(args)
