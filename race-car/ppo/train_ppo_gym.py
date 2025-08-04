import os
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
import pygame

# If you use your own CheckpointCallback, import it here!
from checkpoint_callback import CheckpointCallback

# Import the gym environment
from race_car_gym_env import RaceCarEnv


class CustomWandbCallback(BaseCallback):
    """
    Logs per-episode reward breakdown, PPO loss, and hyperparameters to wandb.
    Also records and uploads game videos periodically.
    """

    def __init__(self, verbose=0, record_freq=50, upload_freq=100):
        super().__init__(verbose)
        self.episode_counter = 0
        self.initialized = False
        self.record_freq = record_freq  # Record every N episodes
        self.upload_freq = upload_freq  # Upload every N episodes
        self.current_recording = None
        self.video_writer = None
        self.is_recording = False
        self.recording_episode = 0

        # Pygame setup for recording
        self.pygame_initialized = False
        self.screen = None

    def _on_training_start(self):
        if not self.initialized:
            # Extract the actual values from schedule objects if needed
            clip_range_value = self.model.clip_range
            if hasattr(clip_range_value, "func"):
                # It's a schedule object, get the initial value
                clip_range_value = clip_range_value.func(1.0)
            elif isinstance(clip_range_value, (int, float)):
                # It's already a number
                pass
            else:
                # Try to call it as a function
                clip_range_value = clip_range_value(1.0)

            learning_rate_value = self.model.learning_rate
            if hasattr(learning_rate_value, "func"):
                # It's a schedule object, get the initial value
                learning_rate_value = learning_rate_value.func(1.0)
            elif isinstance(learning_rate_value, (int, float)):
                # It's already a number
                pass
            else:
                # Try to call it as a function
                learning_rate_value = learning_rate_value(1.0)

            config = {
                "n_steps": self.model.n_steps,
                "batch_size": self.model.batch_size,
                "learning_rate": float(learning_rate_value),
                "n_epochs": self.model.n_epochs,
                "gamma": self.model.gamma,
                "gae_lambda": self.model.gae_lambda,
                "ent_coef": self.model.ent_coef,
                "vf_coef": self.model.vf_coef,
                "clip_range": float(clip_range_value),
                "policy_arch": getattr(self.model.policy, "net_arch", None),
                "normalize_advantage": getattr(self.model, "normalize_advantage", True),
                "environment": "race_car_gym_env",
                "action_space": "with_steering",
            }
            wandb.config.update(config, allow_val_change=True)
            self.initialized = True

    def _on_step(self) -> bool:
        # Per-episode logging (NO `step=`, wandb will auto-increment)
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_counter += 1

                # Log all the keys you want per episode
                reward_breakdown = info.get("reward_breakdown", {})
                wandb.log(
                    {
                        "Episode/Reward": info["episode"]["r"],
                        "Episode/Length": info["episode"]["l"],
                        "Episode/Distance": info.get("distance", 0),
                        "Episode/VelocityX": info.get("velocity_x", 0),
                        "Episode/VelocityY": info.get("velocity_y", 0),
                        # Current step rewards (for debugging)
                        "Step/DistanceReward": reward_breakdown.get(
                            "distance_reward", 0
                        ),
                        "Step/SpeedReward": reward_breakdown.get("speed_reward", 0),
                        "Step/LaneReward": reward_breakdown.get("lane_reward", 0),
                        "Step/ProximityPenalty": reward_breakdown.get(
                            "proximity_penalty", 0
                        ),
                        "Step/CollisionPenalty": reward_breakdown.get(
                            "collision_penalty", 0
                        ),
                        # Status
                        "Episode/Crashed": info.get("crashed", False),
                        "Episode/YPosition": info.get("y_position", 0),
                        "Episode/StepsWithoutProgress": info.get(
                            "steps_without_progress", 0
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

    def _on_training_end(self):
        if self.pygame_initialized:
            pygame.quit()


def linear_schedule(initial_value: float):
    """
    Linear learning rate schedule.
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.
        """
        return progress_remaining * initial_value

    return func


def train_ppo_gym_model(
    project_name="race-car-gym-ppo-improved",
    run_name=None,
    timesteps=2000000,
    use_wandb=True,
    resume_from=None,
    render=False,
):
    n_envs = 8  # Reduced for better sample efficiency
    eval_freq = 25_000  # More frequent evaluation

    print(f"Training for {timesteps:,} timesteps")
    print("Using race_car_gym_env with STEERING (5 actions)")

    # WANDB INIT
    run = None
    if use_wandb:
        config = {
            "algorithm": "PPO",
            "total_timesteps": timesteps,
            "n_envs": n_envs,
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "clip_range_vf": None,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "reward_threshold": 800,
            "environment": "race_car_gym_env",
            "action_space": "with_steering",
            "num_actions": 5,
        }
        run = wandb.init(
            project=project_name,
            name=run_name,
            config=config,
            sync_tensorboard=True,
            save_code=True,
            entity="nm-i-ki",
            tags=["ppo", "race-car", "gym-env", "rl", "improved"],
        )

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    print("Creating training environments using gym environment...")

    def make_env(rank=0):
        # Use None for headless training, "human" for rendering
        env = RaceCarEnv(
            render_mode=None,
            seed=str(np.random.randint(0, 10000)),
        )
        env = Monitor(env)
        return env

    train_env = make_vec_env(make_env, n_envs=n_envs)
    eval_env = Monitor(RaceCarEnv(render_mode=None, seed="eval_seed"))

    if resume_from and os.path.exists(resume_from + ".zip"):
        print(f"Loading existing model from {resume_from}...")
        model = PPO.load(resume_from, env=train_env)
        print(f"Resumed from checkpoint: {resume_from}")
        print(f"Model will train for {timesteps:,} additional timesteps")
    else:
        print("Initializing new PPO model with improved hyperparameters...")

        # Build policy_kwargs with better architecture
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),  # Deeper networks
            activation_fn=torch.nn.ReLU,
            normalize_images=False,
        )

        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=linear_schedule(3e-4),  # Learning rate schedule
            n_steps=2048,  # More steps per update
            batch_size=64,  # Batch size
            n_epochs=10,  # More epochs per update
            gamma=0.99,  # Discount factor
            gae_lambda=0.95,  # GAE lambda
            clip_range=0.2,  # PPO clip range
            clip_range_vf=None,  # No value function clipping
            ent_coef=0.01,  # Entropy coefficient for exploration
            vf_coef=0.5,  # Value function coefficient
            max_grad_norm=0.5,  # Gradient clipping
            policy_kwargs=policy_kwargs,
            verbose=1,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    print(f"Using device: {model.device}")
    print(f"Action space: {train_env.action_space}")
    print("Actions: 0=NOTHING, 1=ACCELERATE, 2=DECELERATE, 3=STEER_LEFT, 4=STEER_RIGHT")

    # Callbacks: evaluation, checkpoint, and custom wandb logging
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=800, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/",
        log_path="./logs/",
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        callback_on_new_best=callback_on_best,
        verbose=1,
        n_eval_episodes=10,  # More evaluation episodes
    )
    callbacks = [eval_callback]

    checkpoint_freq = 50000

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

    # Add custom per-episode+loss logging callback with video recording
    if use_wandb:
        # Record every 50 episodes and upload every 100 episodes to balance storage and visibility
        wandb_callback = CustomWandbCallback(verbose=1, record_freq=50, upload_freq=100)
        callbacks.append(wandb_callback)

    print(f"Starting training for {timesteps:,} timesteps...")
    print(f"Running {n_envs} environments in parallel")

    model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=True)

    # Save final model
    final_model_path = "models/ppo_gym_improved_final"
    model.save(final_model_path)
    print("Training completed!")
    print(f"Final model saved to: {final_model_path}.zip")

    # Also save a checkpoint with timestamp for easy resuming
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = f"models/checkpoint_gym_{timestamp}"
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
        description="Train PPO model using improved race_car_gym_env"
    )
    parser.add_argument(
        "--timesteps", type=int, default=2000000, help="Training timesteps"
    )
    parser.add_argument(
        "--project", default="race-car-gym-ppo-improved", help="Wandb project name"
    )
    parser.add_argument("--run-name", default=None, help="Wandb run name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Path to model checkpoint to resume from (without .zip extension)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable rendering for the first environment",
    )

    args = parser.parse_args()

    print("Starting PPO training for Race Car (Improved)...")
    print("Using race_car_gym_env with steering")
    print("Actions: 0=NOTHING, 1=ACCELERATE, 2=DECELERATE, 3=STEER_LEFT, 4=STEER_RIGHT")

    if args.resume_from:
        print(f"Resuming training from: {args.resume_from}")

    model = train_ppo_gym_model(
        project_name=args.project,
        run_name=args.run_name,
        timesteps=args.timesteps,
        use_wandb=not args.no_wandb,
        resume_from=args.resume_from,
        render=args.render,
    )
