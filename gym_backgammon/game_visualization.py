import torch
import torch.nn as nn
import gymnasium as gym
import time
from itertools import count
import random
import numpy as np
from gym_backgammon.envs.backgammon import WHITE, BLACK, COLORS, TOKEN

"""
This file is used for visualizing the game + playing against a trained model
"""

class BackgammonNet(nn.Module):
    def __init__(self):
        super(BackgammonNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(198, 80),
            nn.Sigmoid(),
            nn.Linear(80, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class TrainedAgent:
    def __init__(self, color, model_path):
        self.color = color
        self.name = 'TrainedAgent({})'.format(self.color)
        self.model = BackgammonNet()
        # Load the weights
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def roll_dice(self):
        return (-random.randint(1, 6), -random.randint(1, 6)) if self.color == WHITE else (
        random.randint(1, 6), random.randint(1, 6))

    def get_features_after_action(self, env, action):
        """Simulates a move, extracts features, and reverts the board."""
        b_copy = env.unwrapped.game.board.copy()
        bar_copy = env.unwrapped.game.bar.copy()
        off_copy = env.unwrapped.game.off.copy()

        env.unwrapped.game.execute_play(self.color, action)
        features = np.array(env.unwrapped.game.get_board_features(self.color), dtype=np.float32)

        env.unwrapped.game.board = b_copy
        env.unwrapped.game.bar = bar_copy
        env.unwrapped.game.off = off_copy
        return features

    def choose_best_action(self, actions, env):
        if not actions:
            return None

        best_action = None
        # White maximizes (aims for 1.0), Black minimizes (aims for 0.0)
        best_val = -1.0 if self.color == WHITE else 2.0

        for action in actions:
            feat = self.get_features_after_action(env, action)
            with torch.no_grad():
                val = self.model(torch.tensor(feat).float()).item()

            if self.color == WHITE:
                if val > best_val:
                    best_val, best_action = val, action
            else:
                if val < best_val:
                    best_val, best_action = val, action

        return best_action

class HumanAgent:
    def __init__(self, color):
        self.color = color
        self.name = f"HumanAgent({self.color})"

    def roll_dice(self):
        return (-random.randint(1, 6), -random.randint(1, 6)) if self.color == WHITE else (
            random.randint(1, 6), random.randint(1, 6))

    def choose_action(self, actions):
        if not actions:
            print("No legal moves. Passing turn.")
            return None

        print("\nAvailable actions:")
        for i, action in enumerate(actions):
            print(f"[{i}] {action}")

        while True:
            try:
                idx = int(input("Choose action: "))
                if 0 <= idx < len(actions):
                    return list(actions)[idx]
            except ValueError:
                pass
            print("Invalid input. Try again.")



def make_plays():
    env = gym.make('gym_backgammon:backgammon-v0', render_mode='rgb_array') #render_mode='human' for the other one
    wins = {WHITE: 0, BLACK: 0}

    agents = {
        WHITE: HumanAgent(WHITE),
        BLACK: TrainedAgent(BLACK, "models/backgammon_TD_lmbda_ep_500000.pth")
    }

    observation, info = env.reset()
    agent_color = info['current_agent']
    first_roll = info['roll']
    agent = agents[agent_color]

    t = time.time()
    env.render()

    for i in count():
        if first_roll:
            roll = first_roll
            env.unwrapped.current_roll = roll
            first_roll = None
        else:
            roll = agent.roll_dice()
            env.unwrapped.current_roll = roll
        env.render()
        time.sleep(2)
        print(
            "Current player={} ({} - {}) | Roll={}".format(agent.color, TOKEN[agent.color], COLORS[agent.color], roll))

        actions = env.unwrapped.get_valid_actions(roll)

        if isinstance(agent, TrainedAgent):
            action = agent.choose_best_action(actions, env)
        else:
            action = agent.choose_action(actions)

        observation_next, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        winner = info.get('winner')

        env.render()

        if done:
            if winner is not None:
                wins[winner] += 1
            print(f"Game Finished! Winner: {winner}")
            break
        time.sleep(3)
        agent_color = env.unwrapped.get_opponent_agent()
        agent = agents[agent_color]
        observation = observation_next

    env.close()


if __name__ == '__main__':
    make_plays()