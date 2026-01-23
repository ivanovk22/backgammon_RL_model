import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import random
from gym_backgammon.envs.backgammon import WHITE, BLACK


# 1. Neural Network Architecture
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


# 2. Simulation Helper: Peeks at the future without ruining the current game
def get_features_after_action(env, player, action):
    # Save current state
    b_copy = env.unwrapped.game.board.copy()
    bar_copy = env.unwrapped.game.bar.copy()
    off_copy = env.unwrapped.game.off.copy()

    # Apply action
    env.unwrapped.game.execute_play(player, action)

    # Get features (198-vector)
    features = np.array(env.unwrapped.game.get_board_features(player), dtype=np.float32)

    # Restore state
    env.unwrapped.game.board = b_copy
    env.unwrapped.game.bar = bar_copy
    env.unwrapped.game.off = off_copy

    return features


def train():
    env = gym.make('gym_backgammon:backgammon-v0')
    model = BackgammonNet()
    try:
        model.load_state_dict(torch.load("backgammon_model3.pth"))
        print("Resuming training from existing model...")
    except FileNotFoundError:
        print("No existing model found. Starting from scratch.")

    optimizer = optim.SGD(model.parameters(), lr=0.01)

    epsilon = 0.05  # 10% Exploration
    total_episodes = 400000
    wins = {WHITE: 0, BLACK: 0}

    print("Starting Training...")

    for episode in range(1, total_episodes + 1):
        obs, info = env.reset()
        current_agent = info['current_agent']
        roll = info['roll']
        done = False

        while not done:
            actions = env.unwrapped.get_valid_actions(roll)

            if not actions:
                # No moves possible
                obs_next, reward, terminated, truncated, info = env.step(None)
            else:
                # --- ACTION SELECTION (Epsilon-Greedy) ---
                if random.random() < epsilon:
                    best_action = random.choice(list(actions))
                else:
                    best_action = None
                    # White wants output -> 1.0 (Win), Black wants output -> 0.0 (Loss)
                    best_val = -1.0 if current_agent == WHITE else 2.0

                    for action in actions:
                        feat = get_features_after_action(env, current_agent, action)
                        with torch.no_grad():
                            val = model(torch.tensor(feat).float()).item()

                        if current_agent == WHITE:
                            if val > best_val:
                                best_val, best_action = val, action
                        else:
                            if val < best_val:
                                best_val, best_action = val, action

                # --- ENVIRONMENT STEP ---
                obs_next, reward, terminated, truncated, info = env.step(best_action)

                # --- TD-LEARNING UPDATE ---
                v_s = model(torch.tensor(obs).float())

                if terminated:
                    # Ground truth: 1 for White win, 0 for Black
                    target = torch.tensor([float(reward)])
                else:
                    with torch.no_grad():
                        target = model(torch.tensor(obs_next).float())

                loss = nn.MSELoss()(v_s, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                obs = obs_next

            done = terminated or truncated
            if not done:
                # Switch Player and Roll Dice
                current_agent = env.unwrapped.get_opponent_agent()

                # Manual Dice Roll logic based on your engine's expectations
                d1, d2 = random.randint(1, 6), random.randint(1, 6)
                roll = (-d1, -d2) if current_agent == WHITE else (d1, d2)

        # Log Progress
        winner = info.get('winner')
        if winner is not None:
            wins[winner] += 1

        if episode % 1000 == 0:
            white_rate = (wins[WHITE] / episode) * 100
            print(f"Ep {episode:4d} | White Win Rate: {white_rate:4.1f}% | Last Loss: {loss.item():.6f}")
            torch.save(model.state_dict(), "backgammon_model3.pth")

    print("Training Finished. Model saved as backgammon_model3.pth")


if __name__ == "__main__":
    train()