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
        self.observation_space = spaces.Box(low=0, high=5, shape=(6, 3, 3), dtype=np.int32)
        self.fig = None

        self.scramble = []

        self.debugLevel = DebugLevel.WARNING
        self.renderViews = True
        self.renderFlat = True
        self.renderCube = False
        self.scrambleSize = 1

        self.config()

    def config(self, debug_level=DebugLevel.WARNING, render_cube=False, scramble_size=1, render_views=True,
               render_flat=True):
        self.debugLevel = debug_level
        self.renderCube = render_cube
        self.scrambleSize = scramble_size

        self.renderViews = render_views
        self.renderFlat = render_flat

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
        observation, reward, episode_over, info : tuple
            observation (object) :
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
        self.scramble = []
        if self.scrambleSize > 0:
            if self.debugLevel == DebugLevel.INFO:
                print("scramble " + str(self.scrambleSize) + " moves")
            self.randomize(self.scrambleSize)
        self.status = self.cube.score()
        self.scoreAfterLastReward = self.status
        return self._get_state()

    def render(self, mode='human', close=False):
        if self.renderCube:
            self.fig = self.cube.render(self.fig, views=self.renderViews, flat=self.renderFlat)
            plt.pause(0.001)

    def _take_action(self, action):
        self.cube.move_by_action(ACTION_LOOKUP[action])

    @staticmethod
    def action_name(action):
        return ACTION_LOOKUP[action].name

    def get_scramble(self):
        return self.scramble

    def valid_scramble_action(self, action, previous_actions):
        num_previous_actions = len(previous_actions)
        if num_previous_actions > 2 \
                and previous_actions[num_previous_actions - 1] == previous_actions[num_previous_actions - 2] \
                and action.name == previous_actions[num_previous_actions - 1]:
            return False
        if num_previous_actions > 1 \
                and self.cube.opposite_actions(previous_actions[num_previous_actions - 1], action):
            return False
        return True

    def randomize(self, number):
        t = 0
        while t < number:
            action = ACTION_LOOKUP[np.random.randint(len(ACTION_LOOKUP.keys()))]
            if self.valid_scramble_action(action, self.scramble):
                self.scramble.append(action.name)
                self.cube.move_by_action(action)
                t += 1

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
