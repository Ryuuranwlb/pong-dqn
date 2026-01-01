import numbers
from typing import Any

import gym
import numpy as np


class RewardShapingWrapper(gym.Wrapper):
    """
    Env wrapper that optionally adds score-delta based shaping.

    shaped_reward = base_reward + shaping_bonus, where bonus only fires
    when scores change (score2 = our agent on the right; score1 = opponent).
    """

    def __init__(self, env: gym.Env, mode: str = "none", alpha: float = 0.2, beta: float = 0.2):
        super().__init__(env)
        if mode not in {"none", "score_delta"}:
            raise ValueError(f"Unsupported reward shaping mode: {mode}")
        self.mode = mode
        self.alpha = alpha
        self.beta = beta
        self.prev_score1 = None
        self.prev_score2 = None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.prev_score1 = None
        self.prev_score2 = None
        return obs

    def _apply_bonus(self, reward: Any, bonus: float):
        """
        Apply bonus while preserving reward structure and avoiding errors for
        array-like rewards (players=2).
        """
        if bonus == 0:
            return reward
        if isinstance(reward, np.ndarray):
            shaped = reward.copy()
            if shaped.size > 0:
                shaped[0] = shaped[0] + bonus
            return shaped
        if isinstance(reward, list):
            if len(reward) == 0:
                return reward
            shaped = reward.copy()
            shaped[0] = shaped[0] + bonus
            return shaped
        if isinstance(reward, tuple):
            if len(reward) == 0:
                return reward
            shaped_list = list(reward)
            shaped_list[0] = shaped_list[0] + bonus
            return tuple(shaped_list)
        if isinstance(reward, numbers.Number):
            return reward + bonus
        try:
            return reward + bonus
        except Exception:
            return reward

    def step(self, *args, **kwargs):
        obs, rew, done, info = self.env.step(*args, **kwargs)
        base_reward = rew
        bonus = 0.0

        if self.mode == "score_delta" and isinstance(info, dict) and "score1" in info and "score2" in info:
            score1 = info["score1"]
            score2 = info["score2"]
            if self.prev_score1 is None or self.prev_score2 is None:
                self.prev_score1 = score1
                self.prev_score2 = score2
                bonus = 0.0
            else:
                delta_score2 = score2 - self.prev_score2
                delta_score1 = score1 - self.prev_score1
                if delta_score2 > 0:
                    bonus += self.alpha
                if delta_score1 > 0:
                    bonus -= self.beta
                self.prev_score1 = score1
                self.prev_score2 = score2
        shaped_reward = self._apply_bonus(base_reward, bonus)

        if isinstance(info, dict):
            info = dict(info)
            info["base_reward"] = base_reward
            info["shaping_bonus"] = bonus
            info["shaped_reward"] = shaped_reward

        return obs, shaped_reward, done, info
