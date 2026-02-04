import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import random
from gym_backgammon.envs.backgammon import WHITE, BLACK
from collections import deque

"""
This file is used for training the models
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



# simulate action without changing the board
def get_features_after_action(env, player, action):
    # Save current state
    b_copy = env.unwrapped.game.board.copy()
    bar_copy = env.unwrapped.game.bar.copy()
    off_copy = env.unwrapped.game.off.copy()

    # Apply action
    env.unwrapped.game.execute_play(player, action)

    # Get features
    features = np.array(env.unwrapped.game.get_board_features(player), dtype=np.float32)

    # Restore state
    env.unwrapped.game.board = b_copy
    env.unwrapped.game.bar = bar_copy
    env.unwrapped.game.off = off_copy

    return features

# Monte Carlo
def train_mc():
    env = gym.make('gym_backgammon:backgammon-v0')
    model = BackgammonNet()

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay_steps = 150000

    total_episodes = 500000
    wins = {WHITE: 0, BLACK: 0}

    print("Starting MC Training...")

    for episode in range(1, total_episodes + 1):

        # Linear epsilon decay
        epsilon = max(epsilon_end, epsilon_start -
                      episode * (epsilon_start - epsilon_end) / epsilon_decay_steps)

        obs, info = env.reset()
        current_agent = info['current_agent']
        roll = info['roll']
        done = False

        episode_states = []  # store visited states

        while not done:
            actions = env.unwrapped.get_valid_actions(roll)

            if not actions:
                obs_next, reward, terminated, truncated, info = env.step(None)
            else:
                # ε-greedy
                if random.random() < epsilon:
                    best_action = random.choice(list(actions))
                else:
                    best_action = None
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

                obs_next, reward, terminated, truncated, info = env.step(best_action)

            # Store the state visited
            episode_states.append(obs)

            obs = obs_next
            done = terminated or truncated

            if not done:
                current_agent = env.unwrapped.get_opponent_agent()
                d1, d2 = random.randint(1, 6), random.randint(1, 6)
                roll = (-d1, -d2) if current_agent == WHITE else (d1, d2)


        # UPDATE
        final_reward = float(reward)  # 1 win, 0 loss

        for state in episode_states:
            v_s = model(torch.tensor(state).float())
            target = torch.tensor([final_reward])
            loss = nn.MSELoss()(v_s, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        winner = info.get('winner')
        if winner is not None:
            wins[winner] += 1

        if episode % 10000 == 0:
            white_rate = (wins[WHITE] / episode) * 100
            print(f"Ep {episode} | White Win Rate: {white_rate:.1f}% | Loss: {loss.item():.5f}")
            torch.save(model.state_dict(), "backgammon_model_mc.pth")
        if episode % 100000 == 0:
            torch.save(model.state_dict(), f"backgammon_model_MC_ep_{episode}.pth")

    print("MC Training Finished.")

# n-step TD Training
def train_n_step_TD(n=5):
    env = gym.make('gym_backgammon:backgammon-v0')
    model = BackgammonNet()
    try:
        model.load_state_dict(torch.load("backgammon_model_nstep.pth"))
        print("Resuming training from existing model...")
    except FileNotFoundError:
        print("No existing model found. Starting from scratch.")

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    epsilon = 0.05
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay_steps = 150000
    total_episodes = 500000
    gamma = 1.0
    wins = {WHITE: 0, BLACK: 0}

    print("Starting Training...")

    for episode in range(1, total_episodes + 1):
        obs, info = env.reset()
        current_agent = info['current_agent']
        roll = info['roll']
        done = False

        # n-step buffer: stores (state, reward)
        buffer = deque()
        epsilon = max(epsilon_end, epsilon_start - (episode / epsilon_decay_steps) * (epsilon_start - epsilon_end))

        while not done:
            actions = env.unwrapped.get_valid_actions(roll)

            if not actions:
                obs_next, reward, terminated, truncated, info = env.step(None)
            else:
                # --- ε-greedy action selection ---
                if random.random() < epsilon:
                    best_action = random.choice(list(actions))
                else:
                    best_action = None
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

                obs_next, reward, terminated, truncated, info = env.step(best_action)

            # ----------------------
            # Append current state and reward
            # ----------------------
            buffer.append((obs, reward))

            # ----------------------
            # n-step update if buffer has enough elements
            # ----------------------
            if len(buffer) >= n:
                # Compute n-step return for the oldest state
                G = sum([gamma ** i * buffer[i][1] for i in range(n)])

                # Add bootstrap from next state if not terminal
                if not terminated:
                    with torch.no_grad():
                        G += gamma ** n * model(torch.tensor(obs_next).float()).item()

                # Update oldest state in buffer
                state_to_update = buffer[0][0]
                v_s = model(torch.tensor(state_to_update).float())
                target = torch.tensor([G])
                loss = nn.MSELoss()(v_s, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Remove oldest state
                buffer.popleft()

            # Prepare for next step
            obs = obs_next
            done = terminated or truncated
            if not done:
                current_agent = env.unwrapped.get_opponent_agent()
                d1, d2 = random.randint(1, 6), random.randint(1, 6)
                roll = (-d1, -d2) if current_agent == WHITE else (d1, d2)

        # ----------------------
        # Flush remaining states in buffer at episode end
        # ----------------------
        while len(buffer) > 0:
            G = sum([gamma ** i * buffer[i][1] for i in range(len(buffer))])
            # Terminal state: no bootstrap
            state_to_update = buffer[0][0]
            v_s = model(torch.tensor(state_to_update).float())
            target = torch.tensor([G])
            loss = nn.MSELoss()(v_s, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            buffer.popleft()

        # Track wins
        winner = info.get('winner')
        if winner is not None:
            wins[winner] += 1

        # Logging
        if episode % 10000 == 0:
            white_rate = (wins[WHITE] / episode) * 100
            print(f"Ep {episode:4d} | White Win Rate: {white_rate:4.1f}% | Last Loss: {loss.item():.6f}")
        if episode % 100000 == 0:
            torch.save(model.state_dict(), f"backgammon_model_{n}_step_ep_{episode}.pth")

    print(f"{n}-step TD training finished")

# TD(lamda) Training

def train_td_lambda(lmbda=0.7):

    env = gym.make('gym_backgammon:backgammon-v0')
    model = BackgammonNet()

    gamma = 1.0
    alpha = 0.01

    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay_steps = 150000
    total_episodes = 500000

    wins = {WHITE: 0, BLACK: 0}

    print("Training TD(lambda)...")

    for episode in range(1, total_episodes + 1):

        epsilon = max(epsilon_end,
                      epsilon_start - episode * (epsilon_start - epsilon_end) / epsilon_decay_steps)

        obs, info = env.reset()
        current_agent = info['current_agent']
        roll = info['roll']
        done = False

        # Eligibility traces (one per parameter)
        traces = [torch.zeros_like(p) for p in model.parameters()]

        while not done:

            actions = env.unwrapped.get_valid_actions(roll)

            if not actions:
                obs_next, reward, terminated, truncated, info = env.step(None)
            else:
                if random.random() < epsilon:
                    action = random.choice(list(actions))
                else:
                    best_val = -1.0 if current_agent == WHITE else 2.0
                    action = None
                    for a in actions:
                        feat = get_features_after_action(env, current_agent, a)
                        with torch.no_grad():
                            val = model(torch.tensor(feat).float()).item()
                        if (current_agent == WHITE and val > best_val) or \
                                (current_agent == BLACK and val < best_val):
                            best_val, action = val, a

                obs_next, reward, terminated, truncated, info = env.step(action)

            # -------- TD(λ) UPDATE --------
            s = torch.tensor(obs, dtype=torch.float32)
            s_next = torch.tensor(obs_next, dtype=torch.float32)

            v_s = model(s)

            with torch.no_grad():
                v_next = torch.tensor([reward], dtype=torch.float32) if terminated else model(s_next)

            delta = (reward + gamma * v_next - v_s).detach()

            model.zero_grad()
            v_s.backward()  # ∇V(s_t)

            with torch.no_grad():
                for p, e in zip(model.parameters(), traces):
                    e.mul_(gamma * lmbda).add_(p.grad)
                    p += alpha * delta * e

            obs = obs_next
            done = terminated or truncated

            if not done:
                current_agent = env.unwrapped.get_opponent_agent()
                d1, d2 = random.randint(1, 6), random.randint(1, 6)
                roll = (-d1, -d2) if current_agent == WHITE else (d1, d2)

        # Track wins
        winner = info.get('winner')
        if winner is not None:
            wins[winner] += 1

        if episode % 10000 == 0:
            white_rate = 100 * wins[WHITE] / episode
            print(f"Ep {episode:6d} | White Win Rate: {white_rate:5.1f}%")

        if episode % 100000 == 0:
            torch.save(model.state_dict(), f"backgammon_TD_lmbda_ep_{episode}.pth")

    print(f"TD({lmbda}) Training finished.")


if __name__ == "__main__":
    # train_n_step_TD(n=1) # TD(0)
    # train_n_step_TD(n=3) # n-step TD
    # train_mc()
    train_td_lambda() # default lambda = 0.7
