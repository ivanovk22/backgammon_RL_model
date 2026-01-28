import gymnasium as gym
import numpy as np
import random
import copy
import time
from gym_backgammon.envs.backgammon import WHITE, BLACK


class MonteCarloAgent:
    def __init__(self, num_rollouts=15):
        self.num_rollouts = num_rollouts

    def select_action(self, env, player, valid_actions):
        if not valid_actions:
            return None

        action_win_counts = {action: 0 for action in valid_actions}

        for action in valid_actions:
            for _ in range(self.num_rollouts):
                if self.simulate_random_rollout(env, player, action):
                    action_win_counts[action] += 1

        best_action = max(action_win_counts, key=action_win_counts.get)
        return best_action

    def simulate_random_rollout(self, env, root_player, first_action):
        sim_game = copy.deepcopy(env.unwrapped.game)
        sim_game.execute_play(root_player, first_action)

        if self.check_win(sim_game):
            return sim_game.get_winner() == root_player

        current_sim_player = BLACK if root_player == WHITE else WHITE

        # Limit rollout length to prevent infinite loops in simulations
        for _ in range(200):
            d1, d2 = random.randint(1, 6), random.randint(1, 6)
            roll = (-d1, -d2) if current_sim_player == WHITE else (d1, d2)

            actions = list(sim_game.get_valid_plays(current_sim_player, roll))
            if actions:
                action = random.choice(actions)
                sim_game.execute_play(current_sim_player, action)

            if self.check_win(sim_game):
                return sim_game.get_winner() == root_player

            current_sim_player = BLACK if current_sim_player == WHITE else WHITE
        return False

    def check_win(self, game):
        return game.off[WHITE] == 15 or game.off[BLACK] == 15


def run_visual_game():
    # 1. SET RENDER_MODE TO 'human'
    env = gym.make('gym_backgammon:backgammon-v0', render_mode='rgb_array')

    # Increase rollouts for better play, decrease for faster turns
    mc_agent = MonteCarloAgent(num_rollouts=20)

    obs, info = env.reset()
    done = False
    current_agent = info['current_agent']
    roll = info['roll']

    print("Starting Visual Game: MC Agent (White) vs Random (Black)")

    while not done:
        # 2. RENDER THE BOARD
        env.render()

        actions = env.unwrapped.get_valid_actions(roll)

        if current_agent == WHITE:
            print(f"White (MC) is thinking... Roll: {roll}")
            action = mc_agent.select_action(env, current_agent, actions)
        else:
            print(f"Black (Random) turn. Roll: {roll}")
            action = random.choice(list(actions)) if actions else None

        # 3. STEP THE ENVIRONMENT
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # 4. SMALL DELAY SO WE CAN WATCH
        time.sleep(0.4)

        if not done:
            current_agent = env.unwrapped.get_opponent_agent()
            d1, d2 = random.randint(1, 6), random.randint(1, 6)
            roll = (-d1, -d2) if current_agent == WHITE else (d1, d2)

    # Show final board state
    env.render()
    winner_name = "White (Monte Carlo)" if info['winner'] == WHITE else "Black (Random)"
    print(f"\nGAME OVER! Winner: {winner_name}")

    time.sleep(3)  # Keep the window open for 3 seconds after the game ends
    env.close()


if __name__ == "__main__":
    run_visual_game()