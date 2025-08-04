import os
import numpy as np
import torch
import wandb
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor

# Import your checkpoint callback wrapper
try:
    from checkpoint_callback_wrapper import CheckpointCallbackWithNormalize

    has_checkpoint_wrapper = True
except ImportError:
    has_checkpoint_wrapper = False
    print("Warning: CheckpointCallbackWithNormalize not found")

# Try to import the original CheckpointCallback as fallback
try:
    from checkpoint_callback import CheckpointCallback
except ImportError:
    print("Warning: CheckpointCallback not found, using basic checkpointing")
    CheckpointCallback = None

# Import the simplified gym environment
from race_car_gym_env import RaceCarEnv


class CustomWandbCallback(BaseCallback):
    """
    Logs per-episode metrics and training stats to wandb.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_counter = 0
        self.initialized = False

    def _on_training_start(self):
        if not self.initialized:
            # Extract hyperparameters
            clip_range_value = self.model.clip_range
            if hasattr(clip_range_value, "func"):
                clip_range_value = clip_range_value.func(1.0)
            elif not isinstance(clip_range_value, (int, float)):
                try:
                    clip_range_value = clip_range_value(1.0)
                except (TypeError, ValueError):
                    clip_range_value = 0.2

            learning_rate_value = self.model.learning_rate
            if hasattr(learning_rate_value, "func"):
                learning_rate_value = learning_rate_value.func(1.0)
            elif not isinstance(learning_rate_value, (int, float)):
                try:
                    learning_rate_value = learning_rate_value(1.0)
                except (TypeError, ValueError):
                    learning_rate_value = 3e-4

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
                "normalize_observations": True,
                "normalize_rewards": True,
                "environment": "race_car_gym_env_simplified",
            }
            wandb.config.update(config, allow_val_change=True)
            self.initialized = True

    def _on_step(self) -> bool:
        # Log episode metrics
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_counter += 1

                wandb.log(
                    {
                        "Episode/Reward": info["episode"]["r"],
                        "Episode/Length": info["episode"]["l"],
                        "Episode/Distance": info["episode"].get("distance", 0),
                        "Episode/Crashed": info.get("crashed", False),
                        "Episode/Number": self.episode_counter,
                    }
                )
        return True

    def _on_rollout_end(self):
        # Log training metrics
        metrics = {}
        if hasattr(self.model, "logger") and hasattr(
            self.model.logger, "name_to_value"
        ):
            for key, value in self.model.logger.name_to_value.items():
                if key.startswith(("train/", "rollout/")):
                    metrics[key] = value

        if metrics:
            wandb.log(metrics, step=self.model.num_timesteps)


def linear_schedule(initial_value: float):
    """
    Linear learning rate schedule.
    """

    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


def train_ppo_normalized(
    project_name="race-car-ppo-normalized",
    run_name=None,
    timesteps=2000000,
    use_wandb=True,
    resume_from=None,
    save_normalized_env=True,
):
    """
    Train PPO with normalized observations and rewards.
    """
    n_envs = 8
    # Adjust eval_freq to be divisible by n_envs for more predictable evaluation
    eval_freq = max(10000 // n_envs * n_envs, n_envs)  # This will be 10008 for n_envs=8
    checkpoint_freq = max(
        50000 // n_envs * n_envs, n_envs
    )  # This will be 50000 for n_envs=8

    print(f"Training for {timesteps:,} timesteps with normalized rewards")
    print("Using simplified race_car_gym_env with 5 actions")
    print(f"Number of parallel environments: {n_envs}")
    print(
        f"Actual evaluation frequency: {eval_freq} timesteps (every {eval_freq // n_envs} vec env steps)"
    )
    print(f"Actual checkpoint frequency: {checkpoint_freq} timesteps")

    # Initialize wandb if requested
    run = None
    if use_wandb:
        config = {
            "algorithm": "PPO",
            "total_timesteps": timesteps,
            "n_envs": n_envs,
            "learning_rate": 1e-4,  # Lower learning rate for stability
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.005,  # Lower entropy for more exploitation
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "normalize_advantage": True,
            "normalize_observations": True,
            "normalize_rewards": True,
        }
        run = wandb.init(
            project=project_name,
            name=run_name
            or f"ppo_normalized_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=config,
            sync_tensorboard=True,
            save_code=True,
            tags=["ppo", "race-car", "normalized", "simplified-reward"],
        )

    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)

    # Ensure evaluation callback has its log directory
    eval_log_path = "./logs/"
    os.makedirs(eval_log_path, exist_ok=True)

    print("Creating training environments...")

    def make_env(rank=0):
        """Create a single environment instance."""
        env = RaceCarEnv(
            render_mode=None,
            seed=f"train_{rank}_{np.random.randint(0, 10000)}",
        )
        env = Monitor(env)
        return env

    # Create vectorized environments
    train_env = make_vec_env(make_env, n_envs=n_envs)

    # Wrap with VecNormalize for observation and reward normalization
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
        training=True,
    )

    # Create evaluation environment (also normalized)
    eval_env = RaceCarEnv(render_mode=None, seed="eval_seed")
    eval_env = Monitor(eval_env)
    eval_env = VecNormalize(
        make_vec_env(lambda: eval_env, n_envs=1),
        norm_obs=True,
        norm_reward=False,  # Don't normalize rewards during evaluation
        clip_obs=10.0,
        training=False,  # Don't update stats during evaluation
    )

    # Load existing model or create new one
    if resume_from and os.path.exists(resume_from + ".zip"):
        print(f"Loading existing model from {resume_from}...")
        model = PPO.load(resume_from, env=train_env)

        # Also load the normalization statistics if available
        norm_path = resume_from + "_vecnormalize.pkl"
        if os.path.exists(norm_path):
            train_env = VecNormalize.load(norm_path, train_env)
            print(f"Loaded normalization statistics from {norm_path}")
    else:
        print("Initializing new PPO model...")

        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=torch.nn.ReLU,
        )

        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=linear_schedule(1e-4),  # Lower initial learning rate
            n_steps=2048,  # Steps per environment before update
            batch_size=64,  # Minibatch size
            n_epochs=10,  # Number of epochs when optimizing
            gamma=0.99,  # Discount factor
            gae_lambda=0.95,  # GAE lambda
            clip_range=0.2,  # PPO clip range
            clip_range_vf=None,  # No value function clipping
            ent_coef=0.005,  # Lower entropy for more exploitation
            vf_coef=0.5,  # Value function coefficient
            max_grad_norm=0.5,  # Gradient clipping
            normalize_advantage=True,  # Normalize advantages
            policy_kwargs=policy_kwargs,
            verbose=1,
            device="cuda" if torch.cuda.is_available() else "cpu",
            tensorboard_log="./logs/" if use_wandb else None,
        )

    print(f"Using device: {model.device}")
    print(f"Action space: {train_env.action_space}")
    print("\nPPO Configuration:")
    print(f"  n_steps: {model.n_steps} (steps per env before update)")
    print(f"  n_envs: {n_envs}")
    print(f"  rollout buffer size: {model.n_steps * n_envs} = {model.n_steps * n_envs}")
    print(f"  batch_size: {model.batch_size}")
    print(f"  n_epochs: {model.n_epochs}")
    print(f"  Total timesteps per update: {model.n_steps * n_envs}")

    # Setup callbacks
    callbacks = []

    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/",
        log_path=eval_log_path,
        eval_freq=eval_freq,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=1,
    )
    callbacks.append(eval_callback)

    # Checkpoint callback with VecNormalize support
    checkpoint_path = f"./models/checkpoints/{run.id if run else 'local'}"

    if has_checkpoint_wrapper:
        # Use the wrapper that supports VecNormalize
        checkpoint_callback = CheckpointCallbackWithNormalize(
            save_freq=checkpoint_freq,
            save_path=checkpoint_path,
            name_prefix="ppo_normalized",
            save_to_wandb=use_wandb,
            verbose=1,
        )
        print("Using CheckpointCallbackWithNormalize (with VecNormalize support)")
    elif CheckpointCallback:
        # Use the original CheckpointCallback without VecNormalize support
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=checkpoint_path,
            name_prefix="ppo_normalized",
            verbose=1,
        )
        print("Note: Using CheckpointCallback without VecNormalize support")

    callbacks.append(checkpoint_callback)
    print(f"Checkpoints will be saved to: {checkpoint_path}")

    # Wandb callback
    if use_wandb:
        wandb_callback = CustomWandbCallback(verbose=1)
        callbacks.append(wandb_callback)

    # Training
    print(f"\nStarting training for {timesteps:,} timesteps...")
    print(f"Running {n_envs} environments in parallel")
    print("Reward normalization: ENABLED")
    print("Observation normalization: ENABLED")

    try:
        model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            progress_bar=True,
            log_interval=10,
        )

        # Save final model
        final_model_path = "models/ppo_normalized_final"
        model.save(final_model_path)
        print(f"\nFinal model saved to: {final_model_path}.zip")

        # Save normalization statistics
        if save_normalized_env and isinstance(train_env, VecNormalize):
            train_env.save(f"{final_model_path}_vecnormalize.pkl")
            print(f"Normalization stats saved to: {final_model_path}_vecnormalize.pkl")

        # Save timestamped checkpoint
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = f"models/checkpoint_normalized_{timestamp}"
        model.save(checkpoint_path)
        if isinstance(train_env, VecNormalize):
            train_env.save(f"{checkpoint_path}_vecnormalize.pkl")
        print(f"Checkpoint saved to: {checkpoint_path}.zip")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save progress
        interrupt_path = (
            f"models/interrupted_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        model.save(interrupt_path)
        if isinstance(train_env, VecNormalize):
            train_env.save(f"{interrupt_path}_vecnormalize.pkl")
        print(f"Progress saved to: {interrupt_path}.zip")

    finally:
        # Cleanup
        train_env.close()
        eval_env.close()

        if use_wandb:
            wandb.finish()

    return model


def test_trained_model(model_path, num_episodes=5):
    """Test a trained model with rendering."""
    print(f"\nTesting model: {model_path}")

    # Load model
    model = PPO.load(model_path)

    # Create environment with rendering
    env = RaceCarEnv(render_mode="human", seed="test_seed")

    # Load normalization if available
    norm_path = model_path.replace(".zip", "_vecnormalize.pkl")
    if os.path.exists(norm_path):
        env = Monitor(env)
        env = VecNormalize.load(norm_path, make_vec_env(lambda: env, n_envs=1))
        env.training = False
        env.norm_reward = False
        print("Loaded normalization statistics")

    # Run episodes
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1

            if done:
                print(
                    f"Episode {episode + 1}: Distance={info[0].get('distance', 0):.1f}, "
                    f"Reward={total_reward:.2f}, Steps={steps}"
                )

    env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train PPO with reward normalization")
    parser.add_argument(
        "--timesteps", type=int, default=2000000, help="Training timesteps"
    )
    parser.add_argument(
        "--project", default="race-car-ppo-normalized", help="Wandb project"
    )
    parser.add_argument("--run-name", default=None, help="Wandb run name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb")
    parser.add_argument("--resume-from", default=None, help="Resume from checkpoint")
    parser.add_argument("--test", default=None, help="Test a trained model")

    args = parser.parse_args()

    if args.test:
        test_trained_model(args.test)
    else:
        print("Starting PPO training with reward normalization...")
        print("=" * 60)

        model = train_ppo_normalized(
            project_name=args.project,
            run_name=args.run_name,
            timesteps=args.timesteps,
            use_wandb=not args.no_wandb,
            resume_from=args.resume_from,
        )
