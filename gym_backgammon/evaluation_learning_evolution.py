import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from gym_backgammon.envs.backgammon import WHITE, BLACK


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

def evaluate_progress(checkpoints, games=2000):
    print("\n=== LEARNING PROGRESS ===")

    baseline = BackgammonNet()
    baseline.load_state_dict(torch.load(checkpoints[0]))
    baseline.eval()

    win_rates = []

    for ckpt in checkpoints:
        model = BackgammonNet()
        model.load_state_dict(torch.load(ckpt))
        model.eval()

        wins = 0
        half = games // 2

        for _ in tqdm(range(half), desc=f"{ckpt} vs baseline"):
            if play_game(model, baseline, True): wins += 1
        for _ in tqdm(range(half)):
            if play_game(model, baseline, False): wins += 1

        win_rates.append(wins / games)

    return win_rates


def evaluate_speed(checkpoints, games=2000):
    print("\n=== LEARNING SPEED ===")

    win_rates = []

    for i in range(len(checkpoints) - 1):
        m1 = BackgammonNet()
        m2 = BackgammonNet()
        m1.load_state_dict(torch.load(checkpoints[i]))
        m2.load_state_dict(torch.load(checkpoints[i + 1]))
        m1.eval()
        m2.eval()

        wins = 0
        half = games // 2

        for _ in tqdm(range(half), desc=f"{checkpoints[i]} vs {checkpoints[i+1]}"):
            if play_game(m2, m1, True): wins += 1
        for _ in tqdm(range(half)):
            if play_game(m2, m1, False): wins += 1

        win_rates.append(wins / games)

    return win_rates

def plot_results(progress_rates, speed_rates, labels, model_name):
    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.plot(labels, progress_rates, marker='o')
    plt.title(f"Learning Progress (vs 100k), {model_name}")
    plt.xlabel("Checkpoint")
    plt.ylabel("Win Rate")

    plt.subplot(1,2,2)
    plt.plot(labels[1:], speed_rates, marker='o')
    plt.title(f"Learning Speed (Consecutive), {model_name}")
    plt.xlabel("Training Segment")
    plt.ylabel("Win Rate")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    checkpoints = [
        "models/backgammon_model_1_step_ep_100000.pth",
        "models/backgammon_model_1_step_ep_200000.pth",
        "models/backgammon_model_1_step_ep_300000.pth",
        "models/backgammon_model_1_step_ep_400000.pth",
        "models/backgammon_model_1_step_ep_500000.pth"
    ]

    labels = ["100k","200k","300k","400k","500k"]

    progress = evaluate_progress(checkpoints)
    speed = evaluate_speed(checkpoints)

    plot_results(progress, speed, labels, 'TD(0)')

    checkpoints = [
        "models/backgammon_model_3_step_ep_100000.pth",
        "models/backgammon_model_3_step_ep_200000.pth",
        "models/backgammon_model_3_step_ep_300000.pth",
        "models/backgammon_model_3_step_ep_400000.pth",
        "models/backgammon_model_3_step_ep_500000.pth"
    ]

    labels = ["100k", "200k", "300k", "400k", "500k"]

    progress = evaluate_progress(checkpoints)
    speed = evaluate_speed(checkpoints)

    plot_results(progress, speed, labels, '3-step TD')

    checkpoints = [
        "models/backgammon_model_5_step_ep_100000.pth",
        "models/backgammon_model_5_step_ep_200000.pth",
        "models/backgammon_model_5_step_ep_300000.pth",
        "models/backgammon_model_5_step_ep_400000.pth",
        "models/backgammon_model_5_step_ep_500000.pth"
    ]

    labels = ["100k", "200k", "300k", "400k", "500k"]

    progress = evaluate_progress(checkpoints)
    speed = evaluate_speed(checkpoints)

    plot_results(progress, speed, labels, '5-step TD')

    checkpoints = [
        "models/backgammon_model_MC_ep_100000.pth",
        "models/backgammon_model_MC_ep_200000.pth",
        "models/backgammon_model_MC_ep_300000.pth",
        "models/backgammon_model_MC_ep_400000.pth",
        "models/backgammon_model_MC_ep_500000.pth"
    ]

    labels = ["100k", "200k", "300k", "400k", "500k"]

    progress = evaluate_progress(checkpoints)
    speed = evaluate_speed(checkpoints)

    plot_results(progress, speed, labels, 'MC')

    checkpoints = [
        "models/backgammon_TD_lmbda_ep_100000.pth",
        "models/backgammon_TD_lmbda_ep_200000.pth",
        "models/backgammon_TD_lmbda_ep_300000.pth",
        "models/backgammon_TD_lmbda_ep_400000.pth",
        "models/backgammon_TD_lmbda_ep_500000.pth"
    ]

    labels = ["100k", "200k", "300k", "400k", "500k"]

    progress = evaluate_progress(checkpoints)
    speed = evaluate_speed(checkpoints)

    plot_results(progress, speed, labels, 'TD(lambda)')