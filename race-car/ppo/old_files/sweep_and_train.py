"""
Complete hyperparameter search and training pipeline.
Each job runs extensive hyperparameter evaluation, then full training with best parameters.
"""

import os
import sys
import numpy as np
import random
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.monitor import Monitor
import torch
import wandb
from checkpoint_callback import CheckpointCallback
from wandb.integration.sb3 import WandbCallback

# Add parent directory to path so we can import src modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.environments.race_car_env import RealRaceCarEnv


def sample_hyperparameters():
    """Sample hyperparameters from the defined search space."""

    search_space = {
        "batch_size": [32, 64, 128, 256, 512],
        "clip_range": [0.1, 0.2, 0.3],
        "ent_coef": [
            0.001,
            0.01,
            0.02,
            0.03,
            0.05,
        ],  # Higher entropy for more exploration
        "gae_lambda": (0.9, 1.0),  # uniform distribution
        "gamma": (0.95, 0.99),  # Adjusted for better long-term planning
        "learning_rate": [
            0.00005,
            0.0001,
            0.0002,
            0.0003,
        ],  # More stable learning rates
        "log_std_init": [-1, 0, 1, 2],  # Removed extreme value
        "n_epochs": [3, 6, 9, 12, 15, 18, 21, 24, 27, 30],
        "n_steps": [512, 1024, 2048, 4096],
        "normalize_advantage": [True, False],
        "target_kl": [0.01, 0.02, 0.03],  # Adjusted target KL
        "vf_coef": [0.5, 1],
    }

    # Sample parameters
    params = {}
    for key, value_range in search_space.items():
        if isinstance(value_range, tuple):
            # Uniform distribution
            params[key] = random.uniform(value_range[0], value_range[1])
        else:
            # Discrete choices
            params[key] = random.choice(value_range)

    return params


def evaluate_hyperparameters(params, job_id, trial_id, eval_steps=100000):
    """Thoroughly evaluate hyperparameters with 100k steps."""

    print(f"🧪 Job {job_id}, Trial {trial_id}: Evaluating hyperparameters...")
    print(f"   Parameters: {params}")
    print(f"   Evaluation steps: {eval_steps:,}")

    # Initialize wandb for evaluation
    wandb.init(
        project="race-car-hyperparam-search",
        name=f"eval_job{job_id}_trial{trial_id}",
        config={
            **params,
            "eval_steps": eval_steps,
            "job_id": job_id,
            "trial_id": trial_id,
        },
        tags=["hyperparameter-evaluation", f"job-{job_id}", f"trial-{trial_id}"],
    )

    # Create environment
    def make_env(rank=0):
        env = RealRaceCarEnv(seed_value=np.random.randint(0, 10000), headless=True)
        env = Monitor(env)
        return env

    train_env = make_vec_env(make_env, n_envs=4)
    eval_env = Monitor(RealRaceCarEnv(seed_value=42, headless=True))

    # Build policy_kwargs
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    if "log_std_init" in params:
        policy_kwargs["log_std_init"] = params["log_std_init"]

    # Create model with sampled hyperparameters
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=0,
        learning_rate=params["learning_rate"],
        n_steps=params["n_steps"],
        batch_size=params["batch_size"],
        n_epochs=params["n_epochs"],
        gamma=params["gamma"],
        gae_lambda=params["gae_lambda"],
        clip_range=params["clip_range"],
        ent_coef=params["ent_coef"],
        vf_coef=params["vf_coef"],
        max_grad_norm=0.5,
        target_kl=params.get("target_kl"),
        normalize_advantage=params.get("normalize_advantage", True),
        policy_kwargs=policy_kwargs,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print(f"   Using device: {model.device}")

    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/eval_job{job_id}_trial{trial_id}/",
        log_path=f"./logs/eval_job{job_id}_trial{trial_id}/",
        eval_freq=10000,
        deterministic=True,
        render=False,
        verbose=1,
    )

    # Wandb callback
    wandb_callback = WandbCallback(
        gradient_save_freq=0,
        model_save_freq=0,
        verbose=1,
    )

    callbacks = [eval_callback, wandb_callback]

    print(f"   Starting evaluation training for {eval_steps:,} steps...")

    try:
        # Train for evaluation
        model.learn(total_timesteps=eval_steps, callback=callbacks, progress_bar=True)

        # Calculate performance metrics
        final_reward = 0
        final_distance = 0

        if len(model.ep_info_buffer) > 0:
            # Get recent episodes for scoring
            recent_episodes = list(model.ep_info_buffer)[-10:]  # Last 10 episodes
            rewards = [ep["r"] for ep in recent_episodes]

            # Try to extract distance from episode info
            distances = []
            for ep in recent_episodes:
                distance = 0
                if "distance" in ep:
                    distance = ep["distance"]
                elif hasattr(ep, "distance"):
                    distance = ep.distance
                distances.append(distance)

            final_reward = np.mean(rewards)
            final_distance = np.mean(distances) if distances else 0

        # Calculate composite score (same as sweep)
        composite_score = final_reward

        print("   ✅ Evaluation complete!")
        print(f"      Average reward: {final_reward:.2f}")
        print(f"      Average distance: {final_distance:.1f}")
        print(f"      Composite score: {composite_score:.2f}")

        # Log final metrics
        wandb.log(
            {
                "eval/final_reward": final_reward,
                "eval/final_distance": final_distance,
                "eval/composite_score": composite_score,
                "eval/completed": True,
            }
        )

    except Exception as e:
        print(f"   ❌ Evaluation failed: {e}")
        composite_score = -float("inf")
        wandb.log({"eval/failed": True, "eval/error": str(e)})

    finally:
        wandb.finish()
        train_env.close()
        eval_env.close()

    return composite_score, params


def full_training(best_params, job_id, training_rounds=10000):
    """Run full training with the best hyperparameters using train_ppo_real logic."""

    print(f"🚀 Job {job_id}: Starting full training with best parameters...")
    print(f"   Best parameters: {best_params}")

    # Use same logic as train_ppo_real.py --rounds 10000
    n_envs = 12
    eval_freq = 100_000  # Same as train_ppo_real.py

    # Calculate timesteps like train_ppo_real.py does for rounds
    n_steps = 36000  # 3 games * 3600 steps per game = 10800 steps per batch
    timesteps = training_rounds * n_steps
    print(f"   Training for {training_rounds} rounds ({timesteps:,} timesteps)")
    print("   Each batch contains 3 games of 1 minute each")
    print("   PPO updates after every 3-game batch completion")

    config = {
        "algorithm": "PPO",
        "total_timesteps": timesteps,
        "n_envs": n_envs,
        "reward_threshold": 1200,  # Increased for 3-game batches
        "environment": "real_race_car_batch_game",
        "games_per_batch": 1,
        "game_duration_seconds": 60,
        "job_id": job_id,
        "best_hyperparams": best_params,
    }

    wandb.init(
        project="race-car-final-training",
        name=f"final_job{job_id}",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        save_code=True,  # optional
        tags=[
            "ppo",
            "race-car",
            "real-game",
            "rl",
            "batch-training",
            "final-training",
            f"job-{job_id}",
        ],
    )

    # Create directories
    os.makedirs(f"models/final_job_{job_id}", exist_ok=True)
    os.makedirs(f"logs/final_job_{job_id}", exist_ok=True)

    print("   Creating training environments using REAL race car game...")

    def make_env(rank=0):
        env = RealRaceCarEnv(seed_value=np.random.randint(0, 10000), headless=True)
        env = Monitor(env)
        return env

    train_env = make_vec_env(make_env, n_envs=n_envs)
    eval_env = Monitor(RealRaceCarEnv(seed_value=42, headless=True))

    print("   Initializing PPO model with best hyperparameters...")

    # Build policy_kwargs like train_ppo_real.py
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    if "log_std_init" in best_params:
        policy_kwargs["log_std_init"] = best_params["log_std_init"]

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=0,
        learning_rate=best_params["learning_rate"],
        n_steps=best_params["n_steps"],  # Use optimized n_steps from search
        batch_size=best_params["batch_size"],
        n_epochs=best_params["n_epochs"],
        gamma=best_params["gamma"],
        gae_lambda=best_params["gae_lambda"],
        clip_range=best_params["clip_range"],
        ent_coef=best_params["ent_coef"],
        vf_coef=best_params["vf_coef"],
        max_grad_norm=0.5,
        target_kl=best_params.get("target_kl"),
        normalize_advantage=best_params.get("normalize_advantage", True),
        policy_kwargs=policy_kwargs,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print(f"   Using device: {model.device}")

    # Callbacks - same as train_ppo_real.py
    callback_on_best = StopTrainingOnRewardThreshold(
        reward_threshold=2000, verbose=1
    )  # ~Full race completion

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/final_job_{job_id}/",
        log_path=f"./logs/final_job_{job_id}/",
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        callback_on_new_best=callback_on_best,
        verbose=1,
    )

    # Use the same custom wandb callback from train_ppo_real.py
    from train_ppo_real import CustomWandbCallback

    custom_wandb_callback = CustomWandbCallback(verbose=1)

    # Add checkpoint callback to save models periodically
    checkpoint_freq = 100000  # Save every 100000 steps
    run_identifier = job_id
    checkpoint_path = f"./models/final_job_{job_id}/checkpoints/{run_identifier}"

    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=checkpoint_path,
        name_prefix="rl_model",
        save_to_wandb=False,  # Disable to avoid symlink issues
        verbose=1,
    )

    callbacks = [eval_callback, custom_wandb_callback, checkpoint_callback]

    print(
        f"   Checkpoints will be saved every {checkpoint_freq} steps to: {checkpoint_path}"
    )
    print(f"   Starting REAL batch training for {timesteps:,} timesteps...")
    print("   Each episode contains 3 consecutive 1-minute games")
    print("   PPO updates after completing each 3-game batch")

    try:
        # Train
        model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=True)

        # Save final model like train_ppo_real.py
        final_model_path = f"models/final_job_{job_id}/ppo_racecar_real_final"
        model.save(final_model_path)
        print("   Real game training completed!")
        print(f"   Final model saved to: {final_model_path}.zip")

        # Also save a checkpoint with timestamp for easy resuming
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = f"models/final_job_{job_id}/checkpoint_{timestamp}"
        model.save(checkpoint_path)
        print(f"   Checkpoint saved to: {checkpoint_path}.zip")

    except Exception as e:
        print(f"   ❌ Training failed: {e}")
        raise

    finally:
        # Clean up
        train_env.close()
        eval_env.close()
        wandb.finish()

    return model


def main():
    """Main function: extensive hyperparameter search + full training."""

    # Get job ID from environment or generate random one
    job_id = os.environ.get("SLURM_JOB_ID", f"local_{random.randint(1000, 9999)}")

    print("🎯 Extensive Hyperparameter Search and Training")
    print(f"   Job ID: {job_id}")
    print("=" * 70)

    # Phase 1: Extensive hyperparameter search
    print("📊 Phase 1: Extensive Hyperparameter Search")
    print("   Each trial: 100,000 training steps")
    print("-" * 50)

    num_trials = 75  # Test 10 different hyperparameter combinations thoroughly
    eval_steps = 100000  # 100k steps per evaluation

    best_score = -float("inf")
    best_params = None

    for trial in range(1, num_trials + 1):
        print(f"\n🔍 Trial {trial}/{num_trials} - Extensive evaluation...")

        # Sample hyperparameters
        params = sample_hyperparameters()

        # Thorough evaluation with 100k steps
        try:
            score, _ = evaluate_hyperparameters(params, job_id, trial, eval_steps)

            if score > best_score:
                best_score = score
                best_params = params
                print(f"   ⭐ NEW BEST! Score: {score:.2f}")
            else:
                print(f"   Score: {score:.2f} (best: {best_score:.2f})")

        except Exception as e:
            print(f"   ❌ Trial {trial} failed: {e}")
            continue

    if best_params is None:
        print("❌ No successful hyperparameter evaluation. Exiting.")
        return

    print("\n✅ Hyperparameter search complete!")
    print(f"   Trials completed: {num_trials}")
    print(f"   Total evaluation steps: {num_trials * eval_steps:,}")
    print(f"   Best composite score: {best_score:.2f}")
    print(f"   Best parameters: {best_params}")

    # Phase 2: Full training with best hyperparameters
    print("\n🏋️ Phase 2: Full Training with Best Parameters")
    print("-" * 50)

    try:
        full_training(best_params, job_id, training_rounds=10000)
        print(f"🎉 Job {job_id} completed successfully!")

    except Exception as e:
        print(f"❌ Full training failed: {e}")
        return

    print(f"\n🏁 JOB {job_id} COMPLETE!")
    print(f"   Hyperparameter search: {num_trials} trials × {eval_steps:,} steps")
    print("   Final training: 10,000 rounds")
    print(f"   Best model: models/final_job_{job_id}/")
    print("   Wandb projects:")
    print("     - race-car-hyperparam-search (search results)")
    print("     - race-car-final-training (final model)")


if __name__ == "__main__":
    main()
