"""
Enhanced PPO training with W&B visualization and video recording.
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import torch
import wandb
from wandb.integration.sb3 import WandbCallback
from checkpoint_callback import CheckpointCallback
import gymnasium as gym
from gymnasium import spaces
import pygame
import argparse
from datetime import datetime

# Import from the existing training script - simplified for compatibility


class SimpleRaceCarEnv(gym.Env):
    """Simplified race car environment for video recording."""

    def __init__(self, seed_value=None, render_mode=None):
        super().__init__()

        # Action and observation spaces
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(20,), dtype=np.float32
        )

        self.render_mode = render_mode
        self.max_steps = 3600
        self.current_step = 0
        self.distance = 0
        self.speed = 0
        self.crashed = False

        # Initialize pygame for rendering
        if self.render_mode == "rgb_array":
            pygame.init()
            self.screen_width = 800
            self.screen_height = 600
            self.surface = pygame.Surface((self.screen_width, self.screen_height))

    def reset(self, *, seed=None, options=None):
        self.current_step = 0
        self.distance = 0
        self.speed = 0
        self.crashed = False

        obs = np.random.rand(20).astype(np.float32)
        info = {"distance": 0, "crashed": False, "speed": 0}
        return obs, info

    def step(self, action):
        self.current_step += 1

        # Simple simulation
        if action == 1:  # ACCELERATE
            self.speed = min(self.speed + 1, 20)
            self.distance += self.speed * 0.1
        elif action == 2:  # DECELERATE
            self.speed = max(self.speed - 1, 0)
            self.distance += self.speed * 0.1
        else:  # Other actions (steering, nothing)
            self.distance += self.speed * 0.1

        # Random crash chance (very low for demo)
        if np.random.rand() < 0.001:
            self.crashed = True

        # Generate observation
        obs = np.random.rand(20).astype(np.float32)

        # Calculate reward
        reward = 0.1 + self.speed * 0.01
        if self.crashed:
            reward = -10

        terminated = self.crashed
        truncated = self.current_step >= self.max_steps

        info = {"distance": self.distance, "crashed": self.crashed, "speed": self.speed}

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the environment for video recording."""
        if self.render_mode != "rgb_array":
            return None

        # Clear surface
        self.surface.fill((50, 50, 50))  # Dark gray background

        # Draw road
        road_width = 400
        road_x = (self.screen_width - road_width) // 2
        pygame.draw.rect(
            self.surface, (100, 100, 100), (road_x, 0, road_width, self.screen_height)
        )

        # Draw lane markings
        lane_width = road_width // 3
        for i in range(1, 3):
            x = road_x + i * lane_width
            for y in range(0, self.screen_height, 40):
                pygame.draw.rect(self.surface, (255, 255, 255), (x - 2, y, 4, 20))

        # Draw car
        car_x = road_x + road_width // 2
        car_y = self.screen_height // 2
        car_width = 40
        car_height = 80
        car_color = (255, 0, 0) if self.crashed else (0, 255, 0)
        pygame.draw.rect(
            self.surface,
            car_color,
            (car_x - car_width // 2, car_y - car_height // 2, car_width, car_height),
        )

        # Add text info
        font = pygame.font.Font(None, 36)

        distance_text = font.render(
            f"Distance: {self.distance:.1f}", True, (255, 255, 255)
        )
        self.surface.blit(distance_text, (10, 10))

        speed_text = font.render(f"Speed: {self.speed:.1f}", True, (255, 255, 255))
        self.surface.blit(speed_text, (10, 50))

        status = "CRASHED!" if self.crashed else "DRIVING"
        status_color = (255, 0, 0) if self.crashed else (0, 255, 0)
        status_text = font.render(status, True, status_color)
        self.surface.blit(status_text, (10, 90))

        # Convert to RGB array
        rgb_array = pygame.surfarray.array3d(self.surface)
        rgb_array = np.transpose(rgb_array, (1, 0, 2))

        return rgb_array

    def close(self):
        if hasattr(self, "surface"):
            pygame.quit()


def train_ppo_with_wandb(
    project_name="race-car-ppo-visualized",
    run_name=None,
    total_timesteps=100000,
    n_envs=4,
    record_video=True,
    video_freq=5000,
    video_length=1000,
):
    """Train PPO with W&B integration and video recording."""

    # Initialize W&B
    config = {
        "algorithm": "PPO",
        "total_timesteps": total_timesteps,
        "n_envs": n_envs,
        "learning_rate": 3e-4,
        "n_steps": 512,  # Smaller for faster rollouts
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "environment": "simple_race_car",
        "record_video": record_video,
        "video_freq": video_freq,
        "video_length": video_length,
    }

    if run_name is None:
        run_name = f"ppo-race-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    run = wandb.init(
        project=project_name,
        name=run_name,
        config=config,
        sync_tensorboard=False,  # Disable to avoid tensorboard issues
        monitor_gym=True,
        save_code=True,
    )

    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs(f"videos/{run.id}", exist_ok=True)

    print("Creating training environments...")

    def make_env(rank=0, render=False):
        def _init():
            env = SimpleRaceCarEnv(
                seed_value=np.random.randint(0, 10000),
                render_mode="rgb_array" if render else None,
            )
            env = Monitor(env)
            return env

        return _init

    # Training environments
    train_env = DummyVecEnv([make_env(i) for i in range(n_envs)])

    # Evaluation environment with video recording
    eval_env = DummyVecEnv([make_env(0, render=record_video)])

    if record_video:
        eval_env = VecVideoRecorder(
            eval_env,
            f"videos/{run.id}",
            record_video_trigger=lambda x: x % video_freq == 0,
            video_length=video_length,
            name_prefix=f"eval-{run_name}",
        )

    print("Initializing PPO model...")
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=None,  # Disable tensorboard logging
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        max_grad_norm=config["max_grad_norm"],
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print(f"Using device: {model.device}")

    # Callbacks
    wandb_callback = WandbCallback(
        gradient_save_freq=1000,
        model_save_path=f"models/{run.id}",
        model_save_freq=5000,
        verbose=2,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{run.id}/best",
        log_path=f"./logs/{run.id}",
        eval_freq=2000,
        n_eval_episodes=3,
        deterministic=True,
        render=False,
        verbose=1,
    )

    # Add checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=f"./models/checkpoints/{run.id}",
        name_prefix="rl_model",
        save_to_wandb=True,
        verbose=1,
    )

    callbacks = [wandb_callback, eval_callback, checkpoint_callback]

    print(f"Starting training for {total_timesteps:,} timesteps...")
    print(f"Videos will be saved to: videos/{run.id}/")

    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=False,  # Disable progress bar to avoid tqdm issues
    )

    # Save final model
    final_model_path = f"models/{run.id}/ppo_final"
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}.zip")

    # Clean up
    train_env.close()
    eval_env.close()

    # Finish W&B run
    wandb.finish()

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO with W&B visualization")
    parser.add_argument(
        "--timesteps", type=int, default=50000, help="Total training timesteps"
    )
    parser.add_argument(
        "--envs", type=int, default=4, help="Number of parallel environments"
    )
    parser.add_argument(
        "--project", default="race-car-ppo-visualized", help="W&B project name"
    )
    parser.add_argument("--run-name", default=None, help="W&B run name")
    parser.add_argument(
        "--no-video", action="store_true", help="Disable video recording"
    )
    parser.add_argument(
        "--video-freq", type=int, default=5000, help="Record video every N steps"
    )
    parser.add_argument(
        "--video-length", type=int, default=1000, help="Length of recorded videos"
    )

    args = parser.parse_args()

    print("Starting PPO training with W&B visualization...")

    model = train_ppo_with_wandb(
        project_name=args.project,
        run_name=args.run_name,
        total_timesteps=args.timesteps,
        n_envs=args.envs,
        record_video=not args.no_video,
        video_freq=args.video_freq,
        video_length=args.video_length,
    )
