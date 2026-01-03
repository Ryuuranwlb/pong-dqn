import retro
import matplotlib.pyplot as plt
import gym
from IPython import display
import gym.spaces
import numpy as np
import random
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
from utils.logger import RunLogger

import signal


CONFIG = {
    'model_dir': 'checkpoints',
    'video_dir': 'videos',
}

active_run_logger = None


total_rews = []
steps_list = []
eps_list = []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-test_mode", action="store_true", default=False)
    parser.add_argument("-memo_1", type=str, default='test')
    parser.add_argument("-memo_2", type=str, default='test')
    parser.add_argument("-seed", type=int, default=0)

    parser.add_argument("-agent_1", type=str, default='DQN')
    parser.add_argument("-agent_2", type=str, default='DQN')

    parser.add_argument("-start_step_1", type=int, default=0)
    parser.add_argument("-start_step_2", type=int, default=0)
    parser.add_argument("-total_episode", type=int, default=400)

    parser.add_argument("-horizon", type=int, default=4)
    parser.add_argument("-player", type=int, default=1)
    parser.add_argument("-skip_frame", type=int, default=4)
    parser.add_argument("-run_root", type=str, default='runs')
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "huber"])
    parser.add_argument("--n_step", type=int, default=1)
    parser.add_argument("-eval_episodes", type=int, default=2)
    parser.add_argument("-train_left", action="store_true", default=False)
    parser.add_argument("-flip2", action="store_true", default=False)
    
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


def train(agent_1, agent_2=None, players=1, skip_frame=2, horizon=2, max_steps=2500, start_step=0, total_episode=1000, logger=None, seed=0, eval_episodes=1, eval_epsilon=0.0, eval_interval_episodes=25):
    global CONFIG


    env = PongDiscretizer(retro.make(game='Pong-Atari2600', players=players), players=players)
    env.reset()

    global total_rews
    global steps_list
    global eps_list
    best_avg_rew = -np.inf
    best_rew = -np.inf
    steps = start_step
    eval_idx = 0

    for i in range(total_episode):
        done = False
        total_rew = 0.0
        reward_acc = 0.0
        last_state = None
        last_action = None
        last_done = False
        obs = env.reset()
        last_info = None
        episode_decision_steps = 0
        losses = []
        td_errors = []
        q_max_values = []
        eps = agent_1.update_epsilon(steps)
        
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
            if agent_1.dqn_net.obs_process_tool.frame_cnt == 0:
                current_state = agent_1.dqn_net.obs_process_tool.obs
                if last_state is not None:
                    agent_1.memory_push(last_state, last_action, current_state, reward_acc, last_done)
                    metrics = agent_1.update(steps)
                    if metrics is not None:
                        losses.append(metrics["loss"])
                        if metrics.get("td_errors") is not None:
                            td_errors.extend(np.asarray(metrics["td_errors"]).tolist())
                        if metrics.get("q_max") is not None:
                            q_max_values.extend(np.asarray(metrics["q_max"]).tolist())
                    steps += 1
                    episode_decision_steps += 1
                    reward_acc = 0.0
                last_state = current_state
                last_action = action_1
                last_done = False

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
            last_info = info
            if players == 2:
                rew = rew[0]
            total_rew += rew
            reward_acc += rew
            last_done = done
            
        if last_state is not None and last_done:
            # 尝试用 terminal 的 obs 构造更合理的 next_state
            code, terminal_state = agent_1.dqn_net.obs_process_tool.process(obs)
            if code == 0:
                next_state = terminal_state
            else:
                # 没到决策边界（skip_frame 未对齐）时，退化为当前缓存里的状态
                next_state = agent_1.dqn_net.obs_process_tool.obs

            agent_1.memory_push(last_state, last_action, next_state, reward_acc, True)
            metrics = agent_1.update(steps)
            if metrics is not None:
                losses.append(metrics["loss"])
                if metrics.get("td_errors") is not None:
                    td_errors.extend(np.asarray(metrics["td_errors"]).tolist())
                if metrics.get("q_max") is not None:
                    q_max_values.extend(np.asarray(metrics["q_max"]).tolist())
            steps += 1
            episode_decision_steps += 1


        total_rews.append(total_rew)
        steps_list.append(steps)
        eps_list.append(eps)

        if total_rew > best_rew:
            best_rew = total_rew

        avg_total_rew = np.mean(total_rews[-100:])

        if avg_total_rew > best_avg_rew:
            best_avg_rew = avg_total_rew
            agent_1.save_model(i, steps, CONFIG['model_dir'])
            # if args.player == 2:
            #     agent_2.save_model(i, CONFIG['model_dir'])

        loss_mean = float(np.mean(losses)) if len(losses) > 0 else np.nan
        td_error_p95 = float(np.percentile(np.asarray(td_errors), 95)) if len(td_errors) > 0 else np.nan
        q_max_mean = float(np.mean(q_max_values)) if len(q_max_values) > 0 else np.nan
        train_score_diff = np.nan
        if last_info is not None and 'score1' in last_info and 'score2' in last_info:
            # agent1 分数是 score2 也就是右侧板
            train_score_diff = float(last_info['score2'] - last_info['score1'])

        print('episode: %d, total step = %d, total reward = %.2f, avg reward = %.6f, best reward = %.2f, best avg reward = %.6f, epsilon = %.6f' % (i, steps, total_rew, avg_total_rew, best_rew, best_avg_rew, eps))

        if logger is not None:
            logger.log_train_episode({
                "run_id": logger.run_id,
                "seed": seed,
                "episode": i,
                "global_step": steps,
                "episode_len": episode_decision_steps,
                "train_return": total_rew,
                "train_score_diff": train_score_diff,
                "epsilon": eps,
                "loss_mean": loss_mean,
                "td_error_p95": td_error_p95,
                "q_max_mean": q_max_mean,
            })

        if (i % eval_interval_episodes == 0) or (i == total_episode - 1):
            # 测试agent
            eval_returns = []
            eval_lengths = []
            eval_score_diffs = []
            eval_wins = []
            eval_draws = []
            eval_losses = []
            for eval_round in range(eval_episodes):
                save_video = (eval_round == 0)
                eval_result = test(agent_1, agent_2, players=players, skip_frame=skip_frame, horizon=horizon, max_steps=max_steps, episode=i, step_id=steps, env=env, eps=eval_epsilon, eval_round=eval_round if eval_episodes > 1 else None, save_video=save_video)
                if eval_result:
                    eval_returns.append(eval_result.get("episode_return", np.nan))
                    eval_lengths.append(eval_result.get("episode_len", np.nan))
                    eval_score_diffs.append(eval_result.get("score_diff", np.nan))
                    eval_wins.append(eval_result.get("win", 0))
                    eval_draws.append(eval_result.get("draw", 0))
                    eval_losses.append(eval_result.get("loss", 0))

            return_mean = float(np.mean(eval_returns)) if len(eval_returns) > 0 else np.nan
            return_std = float(np.std(eval_returns)) if len(eval_returns) > 0 else np.nan
            avg_episode_len = float(np.mean(eval_lengths)) if len(eval_lengths) > 0 else np.nan
            score_diff_mean = float(np.mean(eval_score_diffs)) if len(eval_score_diffs) > 0 else np.nan
            score_diff_std = float(np.std(eval_score_diffs)) if len(eval_score_diffs) > 0 else np.nan
            total_eval = max(len(eval_wins), 1)
            win_rate = float(np.sum(eval_wins) / total_eval) if total_eval > 0 else np.nan
            draw_rate = float(np.sum(eval_draws) / total_eval) if total_eval > 0 else np.nan
            loss_rate = float(np.sum(eval_losses) / total_eval) if total_eval > 0 else np.nan

            if logger is not None:
                logger.log_eval({
                    "run_id": logger.run_id,
                    "seed": seed,
                    "eval_idx": eval_idx,
                    "global_step": steps,
                    "eval_episodes": eval_episodes,
                    "win_rate": win_rate,
                    "draw_rate": draw_rate,
                    "loss_rate": loss_rate,
                    "score_diff_mean": score_diff_mean,
                    "score_diff_std": score_diff_std,
                    "return_mean": return_mean,
                    "return_std": return_std,
                    "avg_episode_len": avg_episode_len,
                })
                eval_idx += 1

    plot_learning_curve(steps_list, total_rews, eps_list, os.path.join(CONFIG['model_dir'], 'pong.png'))
    return steps_list, total_rews, eps_list, None


def test(agent_1, agent_2=None, players=1, skip_frame=2, horizon=2, max_steps=2500, episode=0, step_id=0, env=None, eps=0.0, eval_round=None, save_video=True):
    global CONFIG

    if env is None:
        env = PongDiscretizer(retro.make(game='Pong-Atari2600', players=players), players=players)
        env.reset()

    done = False
    steps = 0
    images = []
    obs = env.reset()
    episode_return = 0.0
    last_info = None

    if players == 2:
        agent_1.reset()
        agent_2.reset()
    else:
        agent_1.reset()

    while not done:
        if players == 2 and (obs == 236).sum() < 12504:
            env.step(2, 2)

        # 右侧板
        action_1 = agent_1.select_action(obs, eps=eps)

        # 左侧板
        if players == 2 and agent_2 is None:
            raise ValueError("agent_2 is None")
        if players == 2 and agent_2 is not None:
            action_2 = agent_2.select_action(obs, eps=0.0)
        else:
            action_2 = None

        nxt_obs, rew, done, info = env.step(action_1, action_2)
        if players == 2:
            rew = rew[0]
        episode_return += rew
        last_info = info

        obs = nxt_obs
            
        if agent_1.dqn_net.obs_process_tool.frame_cnt == 0:
            steps += 1
            if save_video and steps % 8 == 0:
                images.append(env.render(mode='rgb_array'))

        if steps > max_steps:
            break


    if save_video:
        # 创建video writer, 设置好相应参数，fps
        metadata = dict(title='01', artist='Matplotlib',comment='depth prediiton')
        writer = FFMpegWriter(fps=10, metadata=metadata)

        figure = plt.figure(figsize=(10.8, 7.2))
        plt.ion()                                   # 为了可以动态显示
        plt.tight_layout()                          # 尽量减少窗口的留白
        video_name = 'ep_%d_step_%d.mp4' % (episode, step_id) if eval_round is None else 'ep_%d_step_%d_eval%d.mp4' % (episode, step_id, eval_round)
        with writer.saving(figure, os.path.join(CONFIG['video_dir'], video_name), 100): 
            traverse_imgs(writer, images)

    score_diff = np.nan
    score1 = None
    score2 = None
    if last_info is not None and 'score1' in last_info and 'score2' in last_info:
        score1 = last_info['score1']
        score2 = last_info['score2']
        # agent1 分数是 score2 也就是右侧板
        score_diff = float(score2 - score1)

    win = draw = loss = 0
    if not np.isnan(score_diff):
        if score_diff > 0:
            win = 1
        elif score_diff < 0:
            loss = 1
        else:
            draw = 1

    return {
        "episode_len": steps,
        "episode_return": episode_return,
        "score1": score1,
        "score2": score2,
        "score_diff": score_diff,
        "win": win,
        "draw": draw,
        "loss": loss,
        "raw_info": info,
    }


def main(args):
    global CONFIG
    np.random.seed(args.seed)
    random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except ImportError:
        pass
    base_checkpoint_root = CONFIG['model_dir']
    base_video_root = CONFIG['video_dir']
    agent1_load_dir = os.path.join(base_checkpoint_root, args.memo_1)
    agent2_load_dir = os.path.join(base_checkpoint_root, args.memo_2)
    exp_name = args.memo_1 if args.player == 1 else args.memo_1 + '_' + args.memo_2

    if args.player == 2 and not os.path.exists(agent2_load_dir):
        raise ValueError("agent_2 model is not exists")

    # agent_1 = AGENT[args.agent_1](state_size=(args.horizon, 84, 84), action_size=3, skip_frame=args.skip_frame, horizon=args.horizon, clip=False, left=args.train_left, loss_type=args.loss, n_step=args.n_step)
    if args.player == 1:
        agent_1 = AGENT[args.agent_1](state_size=(args.horizon, 84, 84), action_size=3, skip_frame=args.skip_frame, horizon=args.horizon, clip=False, left=args.train_left, loss_type=args.loss, n_step=args.n_step)
        agent_2 = None
    elif args.player == 2:
        agent_1 = AGENT[args.agent_1](state_size=(args.horizon, 84, 84), action_size=3, skip_frame=args.skip_frame, horizon=args.horizon, clip=False, left=args.train_left, loss_type=args.loss, n_step=args.n_step)
        agent_2 = AGENT[args.agent_2](state_size=(args.horizon, 84, 84), action_size=3, skip_frame=args.skip_frame, horizon=args.horizon, clip=False, left=args.flip2)

    if args.test_mode:
        if args.player == 2:
            agent_1.load_model(args.start_step_1, agent1_load_dir)
            agent_2.load_model(args.start_step_2, agent2_load_dir)
            agent_2.set_left(args.flip2)
            

            CONFIG['model_dir'] = os.path.join(base_checkpoint_root, exp_name)
            CONFIG['video_dir'] = os.path.join(base_video_root, exp_name)

            if not os.path.exists(CONFIG['model_dir']):
                os.makedirs(CONFIG['model_dir'])
            if not os.path.exists(CONFIG['video_dir']):
                os.makedirs(CONFIG['video_dir'])
        else:
            if not os.path.exists(agent1_load_dir):
                raise ValueError("model dir is not exists")
            if not os.path.exists(os.path.join(base_video_root, args.memo_1)):
                raise ValueError("video dir is not exists")
            
            agent_1.load_model(args.start_step_1, agent1_load_dir)
            CONFIG['model_dir'] = agent1_load_dir
            CONFIG['video_dir'] = os.path.join(base_video_root, args.memo_1)
        
        # 测试agent
        info = test(agent_1, agent_2, players=args.player, skip_frame=args.skip_frame, horizon=args.horizon, max_steps=2500, episode=0, step_id=args.start_step_1)
        print(info)
    else:
        if args.player == 2:
            if args.start_step_1 > 0:
                agent_1.load_model(args.start_step_1, agent1_load_dir)
            if args.start_step_2 > 0:
                agent_2.load_model(args.start_step_2, agent2_load_dir)
                agent_2.set_left(args.flip2)
        else:
            if args.start_step_1 > 0 and os.path.exists(agent1_load_dir):
                agent_1.load_model(args.start_step_1, agent1_load_dir)

        assert args.start_step_2 > 0

        run_id = RunLogger.default_run_id(exp_name, args.seed)
        run_dir = os.path.join(args.run_root, exp_name, f"seed{args.seed}", run_id)
        CONFIG['model_dir'] = os.path.join(run_dir, 'checkpoints')
        CONFIG['video_dir'] = os.path.join(run_dir, 'videos')
        os.makedirs(CONFIG['model_dir'], exist_ok=True)
        os.makedirs(CONFIG['video_dir'], exist_ok=True)

        eval_episodes = args.eval_episodes
        # p = 0.0625
        # eval_interval_episodes = max(1, int(args.total_episode * p))
        eval_interval_episodes = max(1, args.total_episode // 25)
        eval_epsilon = 0.0

        run_logger = RunLogger(run_dir=run_dir, run_id=run_id, seed=args.seed)
        global active_run_logger
        active_run_logger = run_logger
        config_dict = {
            "exp_name": exp_name,
            "run_id": run_id,
            "seed": args.seed,
            "env_id": "Pong-Atari2600",
            "game": "Pong",
            "obs_mode": "stacked_frames",
            "frame_skip": args.skip_frame,
            "frame_stack": args.horizon,
            "action_repeat": None,
            "players": args.player,
            "algo": "DQN",
            "algo_variant": agent_1.algo_variant,
            "n_step": args.n_step,
            "loss": args.loss,
            "loss_type": args.loss,
            "gamma": agent_1.gamma,
            "gamma_decision": agent_1.gamma_decision,
            "gamma_n": agent_1.gamma_decision ** agent_1.n_step,
            "lr": agent_1.lr,
            "batch_size": agent_1.batch_size,
            "replay_size": agent_1.memory_size,
            "target_update_freq": agent_1.target_update_freq,
            "action_size": agent_1.action_size,
            "epsilon_schedule": {
                "epsilon_max": agent_1.epsilon_max,
                "epsilon_min": agent_1.epsilon_min,
                "epsilon_decay": agent_1.epsilon_decay,
            },
            "train_freq": 1,
            "double": agent_1.double,
            "dueling": agent_1.dueling,
            "max_episodes": args.total_episode,
            "eval_episodes": eval_episodes,
            "eval_epsilon": eval_epsilon,
            "eval_interval_episodes": eval_interval_episodes,
            "skip_frame_gamma_exponent": args.skip_frame,
            "global_step_definition": "decision_steps_only (obs_process_tool.frame_cnt == 0 updates)",
            "gamma_decision_definition": "gamma ** skip_frame (decision-step discount)",
            "gamma_n_definition": "gamma_decision ** n_step (discount for n-step bootstrap)",
            "score_diff_definition": "score2 - score1 from env.info at episode end; win if score_diff > 0, draw if ==0, loss otherwise",
            "run_dir": run_dir,
            "checkpoint_dir": CONFIG['model_dir'],
            "video_dir": CONFIG['video_dir'],
        }
        run_logger.log_config(config_dict)

        try:
            # 训练agent
            steps_list, total_rews, eps_list, data = train(agent_1, agent_2, players=args.player, skip_frame=args.skip_frame, horizon=args.horizon, max_steps=2500, start_step=args.start_step_1, total_episode=args.total_episode, logger=run_logger, seed=args.seed, eval_episodes=eval_episodes, eval_epsilon=eval_epsilon, eval_interval_episodes=eval_interval_episodes)
        finally:
            run_logger.close()
            active_run_logger = None

        return steps_list, total_rews, eps_list, data


def int_handler(signum, frame):
    global active_run_logger
    plot_learning_curve(steps_list, total_rews, eps_list, os.path.join(CONFIG['model_dir'], 'pong.png'))
    if active_run_logger is not None:
        try:
            active_run_logger.close()
        except Exception:
            pass
    exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, int_handler)

    args = parse_args()    
    main(args)
