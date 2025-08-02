# -*- coding: utf-8 -*-
from collections import deque
import sys
import os
import torch

# Add parent directory to path to import race car environment
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.environments.race_car_env import RealRaceCarEnv


class Env:
    def __init__(self, args):
        self.device = args.device
        self.race_env = RealRaceCarEnv(seed_value=args.seed, headless=False)
        self.window = args.history_length  # Number of states to concatenate
        self.state_buffer = deque([], maxlen=args.history_length)
        self.training = True  # Consistent with model training mode
        self.state_size = 20  # Race car environment observation size

    def _get_state(self):
        # Get current observation from race car environment
        obs, _ = (
            self.race_env.reset()
            if not hasattr(self, "_current_obs")
            else (self._current_obs, {})
        )
        return torch.tensor(obs, dtype=torch.float32, device=self.device)

    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(torch.zeros(self.state_size, device=self.device))

    def reset(self):
        # Reset the race car environment
        obs, info = self.race_env.reset()
        self._current_obs = obs

        # Reset state buffer
        self._reset_buffer()

        # Get initial state and add to buffer
        state = torch.tensor(obs, dtype=torch.float32, device=self.device)
        self.state_buffer.append(state)

        # Return stacked states
        return torch.stack(list(self.state_buffer), 0)

    def step(self, action):
        # Step in the race car environment
        obs, reward, terminated, truncated, info = self.race_env.step(action)
        done = terminated or truncated

        self._current_obs = obs
        state = torch.tensor(obs, dtype=torch.float32, device=self.device)
        self.state_buffer.append(state)

        # Return stacked states, reward, done
        return torch.stack(list(self.state_buffer), 0), reward, done

    # Uses crash as terminal signal in training
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False

    def action_space(self):
        return self.race_env.action_space.n

    def render(self):
        # Race car environment handles its own rendering
        pass

    def close(self):
        self.race_env.close()
