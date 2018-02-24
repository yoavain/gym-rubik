from enum import Enum

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces

from gym_rubik.envs.cube import Actions, Cube


class DebugLevel(Enum):
    WARNING = 0,
    INFO = 1,
    VERBOSE = 2


class RubikEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.cube = Cube(3, whiteplastic=False)
        self.action_space = spaces.Discrete(len(ACTION_LOOKUP))
        self.status = self.cube.score()
        self.scoreAfterLastReward = self.status
        self.fig = None

        self.debugLevel = DebugLevel.WARNING
        self.renderCube = False
        self.scrambleSize = 1

        self.config()

    def config(self, debug_level=DebugLevel.WARNING, render_cube=False, scramble_size=1):
        self.debugLevel = debug_level
        self.renderCube = render_cube
        self.scrambleSize = scramble_size

        if self.renderCube:
            plt.ion()
            plt.show()

    def step(self, action):
        """

        Parameters
        ----------
        action :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        self._take_action(action)
        self.status = self.cube.score()
        reward = self._get_reward()
        if self.debugLevel == DebugLevel.VERBOSE:
            print("status = {0} ; score after last reward = {1} ; reward = {2}"
                  .format(str(self.status),
                          str(self.scoreAfterLastReward),
                          str(reward)))
        observation = self._get_state()
        episode_over = self.cube.solved(self.status)
        if episode_over:
            if self.debugLevel == DebugLevel.INFO:
                print("S-O-L-V-E-D !!!")
        return observation, reward, episode_over, {}

    def reset(self):
        self.cube = Cube(3, whiteplastic=False)
        if self.scrambleSize > 0:
            if self.debugLevel == DebugLevel.INFO:
                print("scramble " + str(self.scrambleSize) + " moves")
            self.cube.randomize(self.scrambleSize)
        self.status = self.cube.score()
        self.scoreAfterLastReward = self.status
        self.observation_space = spaces.Box(low=0, high=5, shape=(6, 3, 3), dtype=np.int32)
        return self._get_state()

    def render(self, mode='human', close=False):
        if self.renderCube:
            self.fig = self.cube.render(self.fig, flat=False)
            plt.pause(0.001)

    def _take_action(self, action):
        self.cube.move_by_action(ACTION_LOOKUP[action])

    def _get_reward(self):
        reward = self.status - self.scoreAfterLastReward
        if reward > 0:
            self.scoreAfterLastReward = self.status
            return reward - 1
        return -1

    def _get_state(self):
        return self.cube.get_state()


ACTION_LOOKUP = {
    0: Actions.U,
    1: Actions.U_1,
    2: Actions.D,
    3: Actions.D_1,
    4: Actions.F,
    5: Actions.F_1,
    6: Actions.B,
    7: Actions.B_1,
    8: Actions.R,
    9: Actions.R_1,
    10: Actions.L,
    11: Actions.L_1
}