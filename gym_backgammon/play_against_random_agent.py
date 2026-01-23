import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
import random
from tqdm import tqdm
from gym_backgammon.envs.backgammon import WHITE, BLACK


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


def get_features_after_action(env, player, action):
    b_copy = env.unwrapped.game.board.copy()
    bar_copy = env.unwrapped.game.bar.copy()
    off_copy = env.unwrapped.game.off.copy()

    env.unwrapped.game.execute_play(player, action)
    features = np.array(env.unwrapped.game.get_board_features(player), dtype=np.float32)

    env.unwrapped.game.board = b_copy
    env.unwrapped.game.bar = bar_copy
    env.unwrapped.game.off = off_copy
    return features


def evaluate(num_games=100):
    env = gym.make('gym_backgammon:backgammon-v0')
    model = BackgammonNet()
    model2 = BackgammonNet()


    try:
        model.load_state_dict(torch.load("backgammon_model2.pth"))
        model.eval()
        print(f"Loaded model. Starting tournament: AI (White) vs Random (Black) for {num_games} games...")
    except:
        print("Model not found. Please train first.")
        return

    try:
        model2.load_state_dict(torch.load("backgammon_model3.pth"))
        model2.eval()
        print(f"Loaded model2. Starting tournament: AI (White) vs Random (Black) for {num_games} games...")
    except:
        print("Model2 not found. Please train first.")
        return

    ai_wins = 0
    random_wins = 0

    for game_num in tqdm(range(num_games)):
        obs, info = env.reset()
        current_agent = info['current_agent']
        roll = info['roll']
        done = False

        while not done:
            actions = env.unwrapped.get_valid_actions(roll)

            if not actions:
                action = None
            else:
                if current_agent == WHITE:
                    action = random.choice(list(actions))
                    # AI DECISION
                    # best_action = None
                    # best_val = -1.0
                    # for act in actions:
                    #     feat = get_features_after_action(env, current_agent, act)
                    #     with torch.no_grad():
                    #         val = model(torch.tensor(feat).float()).item()
                    #     if val > best_val:
                    #         best_val, best_action = val, act
                    # action = best_action
                else:
                    # RANDOM DECISION
                    # action = random.choice(list(actions))
                    # AI DECISION
                    best_action = None
                    best_val = 2.0
                    for act in actions:
                        feat = get_features_after_action(env, current_agent, act)
                        with torch.no_grad():
                            val = model2(torch.tensor(feat).float()).item()
                        if val < best_val:
                            best_val, best_action = val, act
                    action = best_action

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if not done:
                current_agent = env.unwrapped.get_opponent_agent()
                d1, d2 = random.randint(1, 6), random.randint(1, 6)
                roll = (-d1, -d2) if current_agent == WHITE else (d1, d2)

        if info['winner'] == WHITE:
            ai_wins += 1
        else:
            random_wins += 1

    print("\n" + "=" * 30)
    print(f"TOURNAMENT RESULTS")
    print(f"AI Wins: {ai_wins} ({ai_wins / num_games:.1%})")
    print(f"Random Wins: {random_wins} ({random_wins / num_games:.1%})")
    print("=" * 30)


if __name__ == "__main__":
    evaluate(1000)