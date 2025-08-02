import os
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
import torch
import wandb

# If you use your own CheckpointCallback, import it here!
from checkpoint_callback import CheckpointCallback

# Add parent directory to path so we can import src modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.environments.race_car_env import RealRaceCarEnv


class CustomWandbCallback(BaseCallback):
    """
    Logs per-episode reward breakdown, PPO loss, and hyperparameters to wandb.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_counter = 0
        self.initialized = False

    def _on_training_start(self):
        if not self.initialized:
            config = {
                "n_steps": self.model.n_steps,
                "batch_size": self.model.batch_size,
                "learning_rate": float(self.model.learning_rate),
                "n_epochs": self.model.n_epochs,
                "gamma": self.model.gamma,
                "gae_lambda": self.model.gae_lambda,
                "ent_coef": self.model.ent_coef,
                "vf_coef": self.model.vf_coef,
                "clip_range": self.model.clip_range,
                "policy_arch": getattr(self.model.policy, "net_arch", None),
            }
            wandb.config.update(config)
            self.initialized = True

    def _on_step(self) -> bool:
        # Per-episode logging (NO `step=`, wandb will auto-increment)
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_counter += 1
                # Log all the keys you want per episode
                wandb.log(
                    {
                        "Episode/Reward": info["episode"]["r"],
                        "Episode/Length": info["episode"]["l"],
                        "Episode/CrashPenalty": info.get("reward_breakdown", {}).get(
                            "crash_penalty", 0
                        ),
                        "Episode/Distance": info.get("distance", 0),
                        "Episode/Speed": info.get("speed", 0),
                        "Episode/CompletionBonus": info.get("reward_breakdown", {}).get(
                            "completion_bonus", 0
                        ),
                    }
                )
        return True

    def _on_rollout_end(self):
        # Log PPO stats with correct global step
        metrics = {}
        if hasattr(self.model, "logger") and hasattr(
            self.model.logger, "name_to_value"
        ):
            for key, value in self.model.logger.name_to_value.items():
                if key.startswith(("train/", "rollout/")):
                    metrics[key] = value
        if metrics:
            wandb.log(metrics, step=self.model.num_timesteps)


def train_real_ppo_model(
    project_name="race-car-real-ppo-batch",
    run_name=None,
    timesteps=90000,
    training_rounds=None,
    use_wandb=True,
    resume_from=None,
):
    n_envs = 4
    eval_freq = 50_000

    if training_rounds is not None:
        n_steps = 3600
        timesteps = training_rounds * n_steps
        print(f"Training for {training_rounds} rounds ({timesteps:,} timesteps)")
        print("Each batch contains 3 games of 1 minute each")
        print("PPO updates after every 3-game batch completion")
    else:
        print(f"Training for {timesteps:,} timesteps")
        print("Each episode is now a batch of 3 games (3 minutes total)")

    # WANDB INIT
    run = None
    if use_wandb:
        config = {
            "algorithm": "PPO",
            "total_timesteps": timesteps,
            "n_envs": n_envs,
            "learning_rate": 3e-4,
            "reward_threshold": 600,
            "environment": "real_race_car_batch_game",
            "games_per_batch": 3,
            "game_duration_seconds": 60,
        }
        run = wandb.init(
            project=project_name,
            name=run_name,
            config=config,
            sync_tensorboard=True,
            save_code=True,
            tags=["ppo", "race-car", "real-game", "rl", "batch-training"],
        )

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    print("Creating training environments using REAL race car game...")

    def make_env(rank=0):
        env = RealRaceCarEnv(seed_value=np.random.randint(0, 10000), headless=True)
        env = Monitor(env)
        return env

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
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    print(f"Using device: {model.device}")

    # Callbacks: evaluation, checkpoint, and custom wandb logging
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=1200, verbose=1)
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

    # Add checkpoint callback
    checkpoint_freq = 5000  # Save every 5000 steps
    # Generate unique run identifier
    if use_wandb and run:
        run_identifier = run.id
    else:
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_identifier = f"local_{run_name or 'training'}_{timestamp}"
    checkpoint_path = f"./models/checkpoints/{run_identifier}"
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=checkpoint_path,
        name_prefix="rl_model",
        save_to_wandb=use_wandb,
        verbose=1,
    )
    callbacks.append(checkpoint_callback)
    print(
        f"Checkpoints will be saved every {checkpoint_freq} steps to: {checkpoint_path}"
    )

    # Add custom per-episode+loss logging callback
    if use_wandb:
        wandb_callback = CustomWandbCallback(verbose=1)
        callbacks.append(wandb_callback)

    print(f"Starting REAL batch training for {timesteps:,} timesteps...")
    print("Each episode contains 3 consecutive 1-minute games")
    print("PPO updates after completing each 3-game batch")

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
