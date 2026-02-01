import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
import random
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from gym_backgammon.envs.backgammon import WHITE, BLACK


# ---------------- NETWORK ----------------
class BackgammonNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(198, 80),
            nn.Sigmoid(),
            nn.Linear(80, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# ---------------- FEATURE SIMULATION ----------------
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


# ---------------- PLAY ONE GAME ----------------
def play_game(modelA, modelB, modelA_white):
    env = gym.make('gym_backgammon:backgammon-v0')
    obs, info = env.reset()
    current_agent = info['current_agent']
    roll = info['roll']
    done = False

    player_to_model = {
        WHITE: modelA if modelA_white else modelB,
        BLACK: modelB if modelA_white else modelA
    }

    while not done:
        actions = env.unwrapped.get_valid_actions(roll)

        if not actions:
            action = None
        else:
            model = player_to_model[current_agent]
            best_action = None
            best_val = -1.0 if current_agent == WHITE else 2.0

            for act in actions:
                feat = get_features_after_action(env, current_agent, act)
                with torch.no_grad():
                    val = model(torch.tensor(feat).float()).item()

                if (current_agent == WHITE and val > best_val) or \
                   (current_agent == BLACK and val < best_val):
                    best_val, best_action = val, act

            action = best_action

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if not done:
            current_agent = env.unwrapped.get_opponent_agent()
            d1, d2 = random.randint(1, 6), random.randint(1, 6)
            roll = (-d1, -d2) if current_agent == WHITE else (d1, d2)

    winner = info["winner"]
    return (winner == WHITE and modelA_white) or (winner == BLACK and not modelA_white)
    # Returns True if modelA won


# ---------------- TOURNAMENT ----------------
def evaluate_models(model_paths, games_per_pair=2000):
    models = []
    names = []

    for name, path in model_paths.items():
        m = BackgammonNet()
        m.load_state_dict(torch.load(path))
        m.eval()
        models.append(m)
        names.append(name)

    n = len(models)
    results = np.zeros((n, n))

    print("\nStarting round-robin tournament...\n")

    for i in range(n):
        for j in range(i + 1, n):

            modelA = models[i]
            modelB = models[j]

            wins_A = 0
            half = games_per_pair // 2

            print(f"{names[i]} vs {names[j]}")

            # A plays WHITE
            for _ in tqdm(range(half)):
                if play_game(modelA, modelB, modelA_white=True):
                    wins_A += 1

            # A plays BLACK
            for _ in tqdm(range(half)):
                if play_game(modelA, modelB, modelA_white=False):
                    wins_A += 1

            win_rate = wins_A / games_per_pair
            results[i, j] = win_rate
            results[j, i] = 1 - win_rate

    # ---------------- HEATMAP ----------------
    plt.figure(figsize=(8, 6))
    sns.heatmap(results, annot=True, xticklabels=names, yticklabels=names,
                cmap="coolwarm", vmin=0, vmax=1)
    plt.title("Pairwise Win Rate Matrix")
    plt.show()

    # ---------------- OVERALL STRENGTH ----------------
    strength = results.sum(axis=1) / (n - 1)

    plt.figure(figsize=(6, 4))
    plt.bar(names, strength)
    plt.ylabel("Average Win Rate")
    plt.title("Overall Model Strength")
    plt.show()

    # Print ranking
    print("\n=== MODEL RANKING ===")
    for name, s in sorted(zip(names, strength), key=lambda x: x[1], reverse=True):
        print(f"{name}: {s:.3f}")


# ---------------- RUN ----------------
if __name__ == "__main__":
    model_paths = {
        "TD(0)": "models/backgammon_model_1_step_ep_500000.pth",
        "3-step TD": "models/backgammon_model_3_step_ep_500000.pth",
        "5-step TD": "models/backgammon_model_5_step_ep_500000.pth",
        "MC": "models/backgammon_model_MC_ep_500000.pth",
        "TD(lambda)": "models/backgammon_TD_lmbda_ep_500000.pth"
    }

    evaluate_models(model_paths, games_per_pair=2000)
