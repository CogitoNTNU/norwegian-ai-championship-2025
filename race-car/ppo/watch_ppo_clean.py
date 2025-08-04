import pygame
import os
import sys
import cv2
import numpy as np
import torch
from datetime import datetime

# Add parent directory to path so we can import src modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from race_car_gym_env import RaceCarEnv
from train_ppo_clean import PPONetwork


def load_ppo_model(model_path, obs_dim, action_dim, hidden_dim=256):
    """Load our custom PPO model from .pth file."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found!")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))

    # Create network
    network = PPONetwork(obs_dim, action_dim, hidden_dim)
    network.load_state_dict(checkpoint["policy_state_dict"])
    network.eval()  # Set to evaluation mode

    print(f"Model loaded from {model_path}")
    print(f"  Episode count: {checkpoint.get('episode_count', 'unknown')}")
    print(f"  Total steps: {checkpoint.get('total_steps', 'unknown')}")

    return network


def watch_clean_ppo_model(
    model_path="models/ppo_racecar_final.pth",
    episodes=3,
    record_video=False,
    deterministic=True,
    debug_interval=100,
):
    """
    Watch the trained PPO model play the race car game using our clean environment.

    Args:
        model_path: Path to the .pth model file
        episodes: Number of episodes to watch
        record_video: Whether to record gameplay video
        deterministic: Use deterministic (most probable) actions vs stochastic sampling
        debug_interval: Print detailed debug info every N steps
    """

    print(f"Loading PPO model from {model_path}...")

    # Create environment to get dimensions
    env = RaceCarEnv(render_mode=None)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Load model
    try:
        model = load_ppo_model(model_path, obs_dim, action_dim)
    except FileNotFoundError:
        print(f"Model file {model_path} not found!")
        print("Available models:")
        models_dir = "models"
        if os.path.exists(models_dir):
            for f in os.listdir(models_dir):
                if f.endswith(".pth"):
                    print(f"  {os.path.join(models_dir, f)}")
        else:
            print("  No models directory found!")
        return

    env.close()

    # Create environment with visualization enabled
    env = RaceCarEnv(render_mode="human")

    # Setup video recording if requested
    video_writer = None
    if record_video:
        os.makedirs("videos", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"videos/ppo_clean_demo_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_filename, fourcc, 60.0, (1600, 1200))
        print(f"Recording video to: {video_filename}")

    print(f"\nWatching PPO model for {episodes} episodes...")
    print("Press ESC to quit, SPACE to pause")
    print("Using our clean environment and PPO implementation")

    for episode in range(episodes):
        print(f"\n=== Episode {episode + 1}/{episodes} ===")

        obs, info = env.reset()
        episode_reward = 0
        step_count = 0
        paused = False

        while True:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        print("PAUSED" if paused else "RESUMED")

            if not paused:
                # Get action from PPO model
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    action_probs, value = model.forward(obs_tensor)

                    if deterministic:
                        # Take the most probable action
                        action = torch.argmax(action_probs, dim=-1).item()
                    else:
                        # Sample from the distribution
                        from torch.distributions import Categorical

                        dist = Categorical(action_probs)
                        action = dist.sample().item()

                # Take step in environment
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                step_count += 1

                # Print progress every debug_interval steps
                if step_count % debug_interval == 0:
                    print(f"\n  === Step {step_count} Debug Info ===")
                    print("  Environment State:")
                    print(f"    Distance: {info['distance']:.1f}")
                    print(f"    Speed (X): {info['velocity_x']:.1f}")
                    print(f"    Speed (Y): {info['velocity_y']:.1f}")
                    print(f"    Y Position: {info['y_position']:.1f}")
                    print(f"    Crashed: {info['crashed']}")
                    print(f"    Episode Reward: {episode_reward:.2f}")
                    print(f"    Last Step Reward: {reward:.3f}")

                    # Print reward breakdown if available
                    if "reward_breakdown" in info and info["reward_breakdown"]:
                        print("  Reward Breakdown:")
                        for reward_type, value in info["reward_breakdown"].items():
                            print(f"    {reward_type}: {value:.3f}")

                    # Print observation values (sensor readings + normalized values)
                    print("  Model Observations (19 values):")
                    obs_labels = []

                    # Import game core to get sensor info
                    import src.game.core as game_core

                    if (
                        hasattr(game_core, "STATE")
                        and game_core.STATE
                        and hasattr(game_core.STATE, "sensors")
                    ):
                        for i, sensor in enumerate(game_core.STATE.sensors):
                            if hasattr(sensor, "name"):
                                obs_labels.append(f"Sensor_{sensor.name}")
                            else:
                                obs_labels.append(f"Sensor_{i}")
                    else:
                        # Fallback labels
                        for i in range(16):
                            obs_labels.append(f"Sensor_{i}")

                    obs_labels.extend(
                        ["Velocity_X_norm", "Velocity_Y_norm", "Lane_Position_norm"]
                    )

                    print("    Raw sensor readings:")
                    if (
                        hasattr(game_core, "STATE")
                        and game_core.STATE
                        and hasattr(game_core.STATE, "sensors")
                    ):
                        for i, sensor in enumerate(game_core.STATE.sensors):
                            raw_reading = (
                                sensor.reading
                                if sensor.reading is not None
                                else "No reading"
                            )
                            normalized = obs[i]
                            print(
                                f"      {sensor.name}: raw={raw_reading}, normalized={normalized:.3f}"
                            )

                    print("    Other normalized values:")
                    print(
                        f"      Velocity X (norm): {obs[16]:.3f} (raw: {info['velocity_x']:.1f})"
                    )
                    print(
                        f"      Velocity Y (norm): {obs[17]:.3f} (raw: {info['velocity_y']:.1f})"
                    )
                    print(
                        f"      Lane Position (norm): {obs[18]:.3f} (raw y: {info['y_position']:.1f})"
                    )

                    # Print action probabilities
                    print("  Model Decision:")
                    with torch.no_grad():
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                        action_probs, value = model.forward(obs_tensor)
                        action_probs_np = action_probs.squeeze().numpy()
                        value_estimate = value.item()

                    action_names = [
                        "NOTHING",
                        "ACCELERATE",
                        "DECELERATE",
                        "STEER_LEFT",
                        "STEER_RIGHT",
                    ]
                    print("    Action Probabilities:")
                    for i, (action_name, prob) in enumerate(
                        zip(action_names, action_probs_np)
                    ):
                        marker = " <-- CHOSEN" if i == action else ""
                        print(f"      {action_name}: {prob:.3f}{marker}")
                    print(f"    Value Estimate: {value_estimate:.3f}")
                    print(f"    Chosen Action: {action_names[action]}")
                    print()

                # Check if episode is done
                if terminated or truncated:
                    print(
                        f"  Episode finished: Distance = {info['distance']:.1f}, "
                        f"Reward = {episode_reward:.2f}"
                    )
                    if info["crashed"]:
                        print(f"  Result: CRASHED at step {step_count}")
                    else:
                        print(f"  Result: Completed {step_count} steps")
                    break

            # Render the game
            env.render()

            # Record frame if video recording is enabled
            if video_writer is not None and env.screen is not None:
                # Capture the screen surface
                frame = pygame.surfarray.array3d(env.screen)
                # Convert from (width, height, channels) to (height, width, channels)
                frame = np.transpose(frame, (1, 0, 2))
                # Convert RGB to BGR for OpenCV
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame)

            # Show pause indicator
            if paused and env.screen is not None:
                pause_font = pygame.font.Font(None, 72)
                pause_text = pause_font.render("PAUSED", True, (255, 0, 0))
                env.screen.blit(
                    pause_text,
                    (env.screen.get_width() // 2 - 100, env.screen.get_height() // 2),
                )
                pygame.display.flip()

            if env.clock:
                env.clock.tick(60)  # 60 FPS

    print("\nDemo complete!")

    # Clean up video recording
    if video_writer is not None:
        video_writer.release()
        print(f"Video saved to: {video_filename}")

    env.close()


def list_available_models():
    """List all available .pth model files."""
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("No models directory found!")
        return []

    models = [f for f in os.listdir(models_dir) if f.endswith(".pth")]
    if models:
        print("Available models:")
        for i, model in enumerate(models, 1):
            model_path = os.path.join(models_dir, model)
            print(f"  {i}. {model}")

            # Try to load info about the model
            checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
            steps = checkpoint.get("total_steps", "unknown")
            episodes = checkpoint.get("episode_count", "unknown")
            print(f"     Steps: {steps}, Episodes: {episodes}")
    else:
        print("No .pth model files found in models directory!")

    return models


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Watch trained PPO model play Race Car using our clean environment"
    )
    parser.add_argument(
        "--model",
        default="models/ppo_racecar_final.pth",
        help="Path to trained .pth model",
    )
    parser.add_argument(
        "--episodes", type=int, default=3, help="Number of episodes to watch"
    )
    parser.add_argument(
        "--record", action="store_true", help="Record video of the gameplay"
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions instead of deterministic",
    )
    parser.add_argument(
        "--list", action="store_true", help="List available models and exit"
    )
    parser.add_argument(
        "--debug-interval",
        type=int,
        default=100,
        help="Print debug info every N steps (default: 100)",
    )

    args = parser.parse_args()

    print("PPO Race Car Visualization - Clean Environment")
    print("=" * 50)

    if args.list:
        list_available_models()
    else:
        watch_clean_ppo_model(
            args.model,
            args.episodes,
            args.record,
            deterministic=not args.stochastic,
            debug_interval=args.debug_interval,
        )
