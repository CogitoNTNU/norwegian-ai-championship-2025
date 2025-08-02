import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.monitor import Monitor
import torch
import wandb
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.environments.race_car_env import RealRaceCarEnv
from callbacks import WandbCallback


# RealRaceCarEnv class has been moved to src/environments/race_car_env.py
# WandbCallback class has been moved to callbacks.py

# Classes are imported at the top of this file
# All training logic remains in the train_real_ppo_model function below


def train_real_ppo_model(
    project_name="race-car-real-ppo-batch",
    run_name=None,
    timesteps=90000,  # Increased default to account for 3x longer episodes
    training_rounds=None,
    use_wandb=True,
    resume_from=None,
):
    """Train PPO model using the real race car game with 3-game batches."""

    n_envs = 4
    eval_freq = 50_000  # Adjusted for longer episodes

    # Convert training rounds to timesteps if specified
    if training_rounds is not None:
        # Each batch now contains 3 games, so episodes are ~3x longer
        n_steps = 3600  # 3 games * 3600 steps per game = 10800 steps per batch
        timesteps = training_rounds * n_steps
        print(f"Training for {training_rounds} rounds ({timesteps:,} timesteps)")
        print("Each batch contains 3 games of 1 minute each")
        print("PPO updates after every 3-game batch completion")
    else:
        print(f"Training for {timesteps:,} timesteps")
        print("Each episode is now a batch of 3 games (3 minutes total)")

    if use_wandb:
        config = {
            "algorithm": "PPO",
            "total_timesteps": timesteps,
            "n_envs": n_envs,
            "learning_rate": 3e-4,
            "reward_threshold": 600,  # Increased for 3-game batches
            "environment": "real_race_car_batch_game",
            "games_per_batch": 3,
            "game_duration_seconds": 60,
        }

        wandb.init(
            project=project_name,
            name=run_name,
            config=config,
            tags=["ppo", "race-car", "real-game", "rl", "batch-training"],
        )

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    print("Creating training environments using REAL race car game...")

    def make_env(rank=0):
        env = RealRaceCarEnv(seed_value=np.random.randint(0, 10000), headless=True)
        return Monitor(env)

    train_env = make_vec_env(make_env, n_envs=n_envs)
    eval_env = Monitor(RealRaceCarEnv(seed_value=42, headless=True))

    if resume_from and os.path.exists(resume_from + ".zip"):
        print(f"Loading existing model from {resume_from}...")
        model = PPO.load(resume_from, env=train_env)
        print(f"Resumed from checkpoint: {resume_from}")
        print(f"Model will train for {timesteps:,} additional timesteps")
    else:
        print("Initializing new PPO model...")
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,  # Collect more steps since episodes are longer (3 games)
            batch_size=64,
            n_epochs=10,
            gamma=0.99,  # Keep standard discount factor
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256])
            ),  # Larger network for complex batch learning
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    print(f"Using device: {model.device}")

    # Callbacks
    callback_on_best = StopTrainingOnRewardThreshold(
        reward_threshold=1200, verbose=1
    )  # ~Full race completion
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/",
        log_path="./logs/",
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        callback_on_new_best=callback_on_best,
        verbose=1,
    )

    callbacks = [eval_callback]

    if use_wandb:
        wandb_callback = WandbCallback(verbose=1)
        callbacks.append(wandb_callback)

    print(f"Starting REAL batch training for {timesteps:,} timesteps...")
    print("Each episode contains 3 consecutive 1-minute games")
    print("PPO updates after completing each 3-game batch")

    # Train
    model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=True)

    # Save final model
    final_model_path = "models/ppo_racecar_real_final"
    model.save(final_model_path)
    print("Real game training completed!")
    print(f"Final model saved to: {final_model_path}.zip")

    # Also save a checkpoint with timestamp for easy resuming
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = f"models/checkpoint_{timestamp}"
    model.save(checkpoint_path)
    print(f"Checkpoint saved to: {checkpoint_path}.zip")

    # Clean up
    train_env.close()
    eval_env.close()

    if use_wandb:
        wandb.finish()

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train PPO model using REAL Race Car game"
    )
    parser.add_argument(
        "--timesteps", type=int, default=100000, help="Training timesteps"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Training rounds (each round = 2048 timesteps)",
    )
    parser.add_argument(
        "--project", default="race-car-real-ppo", help="Wandb project name"
    )
    parser.add_argument("--run-name", default=None, help="Wandb run name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Path to model checkpoint to resume from (without .zip extension)",
    )

    args = parser.parse_args()

    print("Starting REAL PPO training for Race Car with 3-game batches...")
    print(
        "Using actual game elements, collision detection, and batch termination logic"
    )
    print("Each training episode = 3 consecutive games of 1 minute each")

    if args.resume_from:
        print(f"Resuming training from: {args.resume_from}")

    model = train_real_ppo_model(
        project_name=args.project,
        run_name=args.run_name,
        timesteps=args.timesteps,
        training_rounds=args.rounds,
        use_wandb=not args.no_wandb,
        resume_from=args.resume_from,
    )
