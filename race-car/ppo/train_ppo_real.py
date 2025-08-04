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
import cv2
import pygame
from datetime import datetime

# If you use your own CheckpointCallback, import it here!
from checkpoint_callback import CheckpointCallback

# Add parent directory to path so we can import src modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.environments.race_car_env import RealRaceCarEnv


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
            }
            wandb.config.update(config, allow_val_change=True)
            self.initialized = True

    def _init_pygame_for_recording(self):
        """Initialize pygame for video recording if not already done."""
        if not self.pygame_initialized:
            try:
                pygame.init()
                self.screen = pygame.Surface((1600, 1200))
                self.pygame_initialized = True
                if self.verbose:
                    print("Pygame initialized for video recording")
            except Exception as e:
                if self.verbose:
                    print(f"Could not initialize pygame for recording: {e}")
                return False
        return True

    def _start_recording(self, episode_num):
        """Start recording a video for the current episode."""
        if not self._init_pygame_for_recording():
            return

        try:
            os.makedirs("videos/training", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = f"videos/training/episode_{episode_num}_{timestamp}.mp4"

            fourcc = cv2.VideoWriter_fourcc(*"H264")
            self.video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (1600, 1200))
            self.current_recording = video_path
            self.is_recording = True
            self.recording_episode = episode_num

            if self.verbose:
                print(f"Started recording episode {episode_num} to {video_path}")
        except Exception as e:
            if self.verbose:
                print(f"Failed to start recording: {e}")
            self.is_recording = False

    def _stop_recording(self):
        """Stop recording and clean up."""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        recording_path = self.current_recording
        self.current_recording = None
        self.is_recording = False

        if self.verbose and recording_path:
            print(f"Stopped recording, saved to {recording_path}")

        return recording_path

    def _record_frame(self):
        """Record a single frame during training."""
        if not self.is_recording or self.video_writer is None or self.screen is None:
            return

        try:
            # Import STATE from the actual game
            import src.game.core as core

            STATE = core.STATE

            # Clear screen
            self.screen.fill((0, 0, 0))

            # Draw road
            if hasattr(STATE, "road") and STATE.road:
                if hasattr(STATE.road, "surface") and STATE.road.surface:
                    self.screen.blit(STATE.road.surface, (0, 0))

                # Draw walls
                if hasattr(STATE.road, "walls"):
                    for wall in STATE.road.walls:
                        if hasattr(wall, "draw"):
                            wall.draw(self.screen)

            # Draw cars
            if hasattr(STATE, "cars") and STATE.cars:
                for car in STATE.cars:
                    if hasattr(car, "sprite") and car.sprite:
                        self.screen.blit(car.sprite, (car.x, car.y))
                        # Draw bounding box
                        bounds = car.get_bounds()
                        color = (255, 255, 0) if car == STATE.ego else (255, 0, 0)
                        pygame.draw.rect(self.screen, color, bounds, width=2)

            # Draw sensors
            if hasattr(STATE, "sensors") and STATE.sensors:
                for sensor in STATE.sensors:
                    if hasattr(sensor, "draw"):
                        sensor.draw(self.screen)

            # Add episode info overlay
            font = pygame.font.Font(None, 36)
            info_text = f"Training Episode {self.recording_episode}"
            text_surface = font.render(info_text, True, (255, 255, 255))
            self.screen.blit(text_surface, (10, 10))

            # Convert pygame surface to OpenCV format
            frame = pygame.surfarray.array3d(self.screen)
            frame = np.transpose(frame, (1, 0, 2))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Write frame to video
            self.video_writer.write(frame)

        except Exception as e:
            if self.verbose:
                print(f"Error recording frame: {e}")

    def _upload_video_to_wandb(self, video_path, episode_num):
        """Upload recorded video to wandb."""
        try:
            if os.path.exists(video_path):
                wandb.log(
                    {
                        f"training_video": wandb.Video(
                            video_path, format="mp4", caption=f"Episode {episode_num}"
                        )
                    },
                    step=self.model.num_timesteps,
                )

                if self.verbose:
                    print(f"Uploaded video for episode {episode_num} to wandb")

                # Clean up local file to save space
                try:
                    os.remove(video_path)
                    if self.verbose:
                        print(f"Cleaned up local video file: {video_path}")
                except:
                    pass

        except Exception as e:
            if self.verbose:
                print(f"Failed to upload video to wandb: {e}")

    def _on_step(self) -> bool:
        # Record frame if we're currently recording
        if self.is_recording:
            self._record_frame()

        # Per-episode logging (NO `step=`, wandb will auto-increment)
        for info in self.locals.get("infos", []):
            if "episode" in info:
                # Stop recording if we were recording this episode
                if self.is_recording:
                    video_path = self._stop_recording()
                    if video_path and self.episode_counter % self.upload_freq == 0:
                        self._upload_video_to_wandb(video_path, self.episode_counter)

                self.episode_counter += 1

                # Start recording for specific episodes
                if self.episode_counter % self.record_freq == 0:
                    self._start_recording(self.episode_counter)

                # Log all the keys you want per episode
                reward_breakdown = info.get("reward_breakdown", {})
                accumulated_rewards = info.get("accumulated_rewards", {})
                wandb.log(
                    {
                        "Episode/Reward": info["episode"]["r"],
                        "Episode/Length": info["episode"]["l"],
                        "Episode/Distance": info.get("distance", 0),
                        "Episode/TotalDistance": info.get(
                            "distance", 0
                        ),  # Total distance traveled
                        "Episode/Speed": info.get("speed", 0),
                        # Current step rewards (for debugging)
                        "Step/SpeedReward": reward_breakdown.get("speed_reward", 0),
                        "Step/OvertakingReward": reward_breakdown.get(
                            "overtaking_reward", 0
                        ),
                        "Step/DistanceReward": reward_breakdown.get(
                            "distance_reward", 0
                        ),
                        "Step/ProximityPenalty": reward_breakdown.get(
                            "proximity_penalty", 0
                        ),
                        "Step/CrashPenalty": reward_breakdown.get("crash_penalty", 0),
                        "Step/CompletionBonus": reward_breakdown.get(
                            "completion_bonus", 0
                        ),
                        # Accumulated episode totals
                        "Episode/AccumSpeedReward": accumulated_rewards.get(
                            "speed_reward", 0
                        ),
                        "Episode/AccumOvertakingReward": accumulated_rewards.get(
                            "overtaking_reward", 0
                        ),
                        "Episode/AccumDistanceReward": accumulated_rewards.get(
                            "distance_reward", 0
                        ),
                        "Episode/AccumProximityPenalty": accumulated_rewards.get(
                            "proximity_penalty", 0
                        ),
                        "Episode/AccumCrashPenalty": accumulated_rewards.get(
                            "crash_penalty", 0
                        ),
                        "Episode/AccumCompletionBonus": accumulated_rewards.get(
                            "completion_bonus", 0
                        ),
                        # Status
                        "Episode/Crashed": info.get("crashed", False),
                        "Episode/RaceCompleted": info.get("race_completed", False),
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
        """Clean up video recording resources when training ends."""
        if self.is_recording:
            video_path = self._stop_recording()
            if video_path:
                self._upload_video_to_wandb(video_path, self.episode_counter)

        if self.pygame_initialized:
            try:
                pygame.quit()
            except:
                pass


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
        n_steps = 3600  # One full 60-second game
        timesteps = training_rounds * n_steps
        print(f"Training for {training_rounds} rounds ({timesteps:,} timesteps)")
        print("Each round is 1 game of 60 seconds (3600 timesteps)")
    else:
        print(f"Training for {timesteps:,} timesteps")
        print("Each episode is 1 game of 60 seconds (3600 timesteps)")

    # WANDB INIT
    run = None
    if use_wandb:
        config = {
            "algorithm": "PPO",
            "total_timesteps": timesteps,
            "n_envs": n_envs,
            "ent_coef": 0.01,  # Only custom setting
            "reward_threshold": 600,
            "environment": "real_race_car_batch_game",
            "games_per_batch": 1,
            "game_duration_seconds": 60,
            "video_recording_enabled": True,
            "video_record_freq": 50,
            "video_upload_freq": 100,
        }
        run = wandb.init(
            project=project_name,
            name=run_name,
            config=config,
            sync_tensorboard=True,
            save_code=True,
            entity="nm-i-ki",
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
        print("Initializing new PPO model with specified hyperparameters...")

        # Build policy_kwargs with log_std_init
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            log_std_init=1,  # From your specified parameters
        )

        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=0,
            ent_coef=0.01,  # Only custom setting
            policy_kwargs=policy_kwargs,
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
    checkpoint_freq = 50000  # Save every 50000 steps
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

    # Add custom per-episode+loss logging callback with video recording
    if use_wandb:
        # Record every 50 episodes and upload every 100 episodes to balance storage and visibility
        wandb_callback = CustomWandbCallback(verbose=1, record_freq=50, upload_freq=100)
        callbacks.append(wandb_callback)

    print(f"Starting REAL training for {timesteps:,} timesteps...")
    print("Each episode is 1 game lasting 60 seconds (3600 timesteps)")
    print(f"Running {n_envs} environments in parallel")

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
        help="Training rounds (each round = 3600 timesteps = 1 full 60-second game)",
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

    print("Starting REAL PPO training for Race Car...")
    print("Using actual game elements, collision detection, and termination logic")
    print("Each training episode = 1 game of 60 seconds")

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
