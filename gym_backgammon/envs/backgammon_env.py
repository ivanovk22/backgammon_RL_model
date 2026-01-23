import gymnasium as gym
from gymnasium import spaces # Use gymnasium spaces
import numpy as np
from random import randint
from gym_backgammon.envs.backgammon import Backgammon as Game, WHITE, BLACK
from gym_backgammon.envs.rendering import Viewer

STATE_W = 96
STATE_H = 96

SCREEN_W = 800
SCREEN_H = 600


class BackgammonEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array', 'state_pixels'], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.game = Game()
        self.current_agent = None
        self.render_mode = render_mode

        # 2. Observation Space
        low = np.zeros((198, 1), dtype=np.float32)
        high = np.ones((198, 1), dtype=np.float32)
        for i in range(3, 97, 4):
            high[i] = 6.0
        high[96] = 7.5

        for i in range(101, 195, 4):
            high[i] = 6.0
        high[194] = 7.5
        self.observation_space = spaces.Box(low=low.flatten(), high=high.flatten(), dtype=np.float32)

        # 3. FIX: Add the missing Action Space
        # Backgammon move space is complex, but Gymnasium requires a definition.
        # Since moves are usually list-based, we use a placeholder or Discrete.
        self.action_space = spaces.Discrete(1000)  # Placeholder: adjust to your needs

        self.counter = 0
        self.max_length_episode = 10000
        self.viewer = None

    def step(self, action):
        self.game.execute_play(self.current_agent, action)
        # self.current_roll = None
        # get the board representation from the opponent player perspective (the current player has already performed the move)
        observation = np.array(self.game.get_board_features(self.game.get_opponent(self.current_agent)),
                               dtype=np.float32)
        reward = 0
        terminated = False
        truncated = False

        winner = self.game.get_winner()

        if winner is not None:
            if winner == WHITE:
                reward = 1
            terminated = True

        if self.counter > self.max_length_episode:
            truncated = True

        self.counter += 1

        return observation, reward, terminated, truncated, {"winner": winner}

    def reset(self, seed=None, options=None):
        # 5. Handle seed for Gymnasium compatibility
        super().reset(seed=seed)

        roll = randint(1, 6), randint(1, 6)
        while roll[0] == roll[1]:
            roll = randint(1, 6), randint(1, 6)

        if roll[0] > roll[1]:
            self.current_agent = WHITE
            roll = (-roll[0], -roll[1])
        else:
            self.current_agent = BLACK

        self.game = Game()
        self.counter = 0

        observation = np.array(self.game.get_board_features(self.current_agent), dtype=np.float32)
        # 6. Return observation and info dict
        info = {"current_agent": self.current_agent, "roll": roll}
        self.current_roll = roll
        return observation, info

    def render(self, mode='human'):
        # assert mode in ['human', 'rgb_array', 'state_pixels'], print(mode)
        mode = self.render_mode
        if mode == 'human':
            self.game.render()
            return None
        else:
            if self.viewer is None:
                self.viewer = Viewer(SCREEN_W, SCREEN_H)

            if mode == 'rgb_array':
                width = SCREEN_W
                height = SCREEN_H

            else:
                assert mode == 'state_pixels', print(mode)
                width = STATE_W
                height = STATE_H
            return self.viewer.render(board=self.game.board, bar=self.game.bar, off=self.game.off, state_w=width, state_h=height, roll=getattr(self, 'current_roll', None))

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def get_valid_actions(self, roll):
        return self.game.get_valid_plays(self.current_agent, roll)

    def get_opponent_agent(self):
        self.current_agent = self.game.get_opponent(self.current_agent)
        return self.current_agent


class BackgammonEnvPixel(BackgammonEnv):

    def __init__(self, render_mode='state_pixels'):
        super(BackgammonEnvPixel, self).__init__(render_mode=render_mode)
        self.observation_space = spaces.Box(low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8)

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)

        pixel_observation = self.render()  # Uses self.render_mode set in __init__

        return pixel_observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)

        pixel_observation = self.render()

        return pixel_observation, info