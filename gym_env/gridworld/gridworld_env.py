import gym
import numpy as np
from gym import spaces


class GridworldEnv(gym.Env):
    def __init__(self, row, col, n_action):
        super(GridworldEnv, self).__init__()

        self.row = row
        self.col = col
        self.target_loc = np.array([1, 0], dtype=np.uint16)

        self.observation_space = spaces.Box(low=-1., high=1., shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(n_action)

    def step(self, action):
        self._take_action(action)

        next_observation = self._get_normalize_loc()
        reward = 1 if np.array_equal(self.agent_loc, self.target_loc) is True else 0
        done = True if np.array_equal(self.agent_loc, self.target_loc) is True else False

        return (next_observation, reward, done, {})

    def reset(self):
        self._init_env()
        return self._get_normalize_loc()

    def render(self):
        raise NotImplementedError()

    def _init_env(self):
        self.agent_loc = np.array([0, self.col // 2], dtype=np.uint16)

    def _get_normalize_loc(self):
        normalized_loc = np.array([
            self.agent_loc[0] / float(self.row),
            self.agent_loc[1] / float(self.col)])
        return normalized_loc

    def _take_action(self, action):
        action = np.argmax(action)
        if action == 0:
            new_loc = self.agent_loc[0] - 1
            if new_loc < 0:
                new_loc = 0  # Out of bound
            self.agent_loc = np.array([new_loc, self.agent_loc[1]])
    
        elif action == 1:
            new_loc = self.agent_loc[0] + 1
            if new_loc >= self.row:
                new_loc = self.row - 1  # Out of bound
            self.agent_loc = np.array([new_loc, self.agent_loc[1]])
    
        elif action == 2:
            new_loc = self.agent_loc[1] + 1
            if new_loc >= self.col:
                new_loc = self.col - 1  # Out of bound
            self.agent_loc = np.array([self.agent_loc[0], new_loc])
    
        elif action == 3:
            new_loc = self.agent_loc[1] - 1
            if new_loc < 0: 
                new_loc = 0  # Out of bound
            self.agent_loc = np.array([self.agent_loc[0], new_loc])
    
        else:
            raise ValueError("Wrong action")
