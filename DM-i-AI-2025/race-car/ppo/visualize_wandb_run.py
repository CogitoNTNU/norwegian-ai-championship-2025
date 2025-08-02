"""
Download and visualize models from W&B runs.
"""

import os
import argparse
import wandb
import tempfile
from stable_baselines3 import PPO
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def download_model_from_wandb(
    run_id, model_name="rl_model_final.zip", project="race-car-real-ppo"
):
    """Download a model from W&B."""

    print(f"Connecting to W&B project: {project}")
    print(f"Downloading from run: {run_id}")

    # Initialize W&B API
    api = wandb.Api()

    # Get the run
    try:
        run = api.run(f"{project}/{run_id}")
        print(f"SUCCESS: Found run: {run.name}")
        print(f"   Status: {run.state}")
        print(f"   Duration: {run.summary.get('_runtime', 'unknown')}s")
        print(f"   Final reward: {run.summary.get('rollout/ep_rew_mean', 'unknown')}")
    except Exception as e:
        print(f"ERROR: Error accessing run: {e}")
        return None

    # List available files
    print("\nAvailable files in run:")
    model_files = []
    for file in run.files():
        print(f"  - {file.name}")
        if file.name.endswith(".zip") and (
            "model" in file.name or "checkpoint" in file.name
        ):
            model_files.append(file.name)

    if not model_files:
        print("ERROR: No model files found in this run!")
        return None

    # Choose model file
    if model_name not in [f for f in model_files]:
        print(f"\n'{model_name}' not found. Available models:")
        for i, mf in enumerate(model_files):
            print(f"  {i}: {mf}")

        choice = input(
            f"Enter model number (0-{len(model_files) - 1}) or filename: "
        ).strip()

        try:
            model_name = model_files[int(choice)]
        except (ValueError, IndexError):
            if choice in model_files:
                model_name = choice
            else:
                print("ERROR: Invalid choice")
                return None

    # Download the model
    print(f"\nDownloading {model_name}...")

    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, model_name)

        # Download file
        for file in run.files():
            if file.name == model_name:
                file.download(temp_dir, replace=True)
                break

        if not os.path.exists(file_path):
            print(f"ERROR: Failed to download {model_name}")
            return None

        # Extract if it's a zip
        extract_path = file_path.replace(".zip", "")

        print("SUCCESS: Downloaded successfully!")
        print(f"   File size: {os.path.getsize(file_path) / (1024 * 1024):.2f} MB")

        return file_path, extract_path


def visualize_model(model_path, episodes=3, max_steps=3600):
    """Visualize a downloaded model."""

    print(f"\nLoading model from {model_path}")

    try:
        # Load the model
        model = PPO.load(model_path.replace(".zip", ""))
        print("SUCCESS: Model loaded successfully!")

        # Print model info
        print(f"   Device: {model.device}")
        print(f"   Policy: {type(model.policy).__name__}")

    except Exception as e:
        print(f"ERROR: Error loading model: {e}")
        return

    # Import environment and pygame only when visualization is needed
    try:
        from src.environments.race_car_env import RealRaceCarEnv
        import pygame
    except Exception as e:
        print(f"ERROR: Error importing visualization dependencies: {e}")
        print("This might be due to pygame compatibility issues with Python 3.13")
        return

    # Create environment for visualization
    print("\nCreating race environment...")
    env = RealRaceCarEnv(seed_value=42, headless=False)

    # Initialize pygame for event handling
    pygame.init()
    pygame.display.set_mode((1600, 1200))
    pygame.display.set_caption("W&B Model Visualization")
    clock = pygame.time.Clock()

    print(f"\nRunning {episodes} episodes...")
    print("Controls: ESC to quit, SPACE to pause")

    total_rewards = []
    total_distances = []

    for episode in range(episodes):
        print(f"\n--- Episode {episode + 1}/{episodes} ---")

        obs, info = env.reset()
        episode_reward = 0
        step_count = 0
        paused = False

        while step_count < max_steps:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    env.close()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        env.close()
                        return
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        print("Paused" if paused else "Resumed")

            if not paused:
                # Get action from model
                action, _ = model.predict(obs, deterministic=True)

                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                step_count += 1

                # Print progress
                if step_count % 600 == 0:  # Every 10 seconds
                    print(
                        f"  Step {step_count}: Distance = {info['distance']:.1f}, Speed = {info['speed']:.1f}"
                    )

                if terminated or truncated:
                    break

            clock.tick(60)  # 60 FPS

        # Episode summary
        final_distance = info["distance"]
        total_rewards.append(episode_reward)
        total_distances.append(final_distance)

        status = (
            "CRASHED"
            if info["crashed"]
            else ("COMPLETED" if info["race_completed"] else "TIMEOUT")
        )
        print(f"  Result: {status}")
        print(f"  Distance: {final_distance:.1f}")
        print(f"  Reward: {episode_reward:.2f}")

    # Final summary
    print("\nPERFORMANCE SUMMARY:")
    print(f"  Average distance: {sum(total_distances) / len(total_distances):.1f}")
    print(f"  Best distance: {max(total_distances):.1f}")
    print(f"  Average reward: {sum(total_rewards) / len(total_rewards):.2f}")
    print(f"  Crashes: {sum(1 for info in [{}] if info.get('crashed', False))}")

    pygame.quit()
    env.close()


def list_wandb_runs(project="race-car-real-ppo", limit=10):
    """List recent W&B runs."""

    print(f"Recent runs in {project}:")

    api = wandb.Api()
    runs = api.runs(project)

    for i, run in enumerate(runs[:limit]):
        status = (
            "FINISHED"
            if run.state == "finished"
            else "RUNNING"
            if run.state == "running"
            else "FAILED"
        )
        print(f"  {i + 1:2d}. [{status}] {run.id} - {run.name}")
        print(f"      Created: {run.created_at}")
        print(f"      Reward: {run.summary.get('rollout/ep_rew_mean', 'N/A')}")
        print(
            f"      Distance: {run.summary.get('Racing/Average_Distance_Traveled', 'N/A')}"
        )
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize models from W&B runs")
    parser.add_argument("--run-id", type=str, help="W&B run ID to download from")
    parser.add_argument(
        "--model",
        type=str,
        default="rl_model_final.zip",
        help="Model filename to download",
    )
    parser.add_argument(
        "--project", type=str, default="race-car-real-ppo", help="W&B project name"
    )
    parser.add_argument(
        "--episodes", type=int, default=3, help="Number of episodes to run"
    )
    parser.add_argument("--list-runs", action="store_true", help="List recent W&B runs")

    args = parser.parse_args()

    if args.list_runs:
        list_wandb_runs(args.project)

    elif args.run_id:
        # Download and visualize
        result = download_model_from_wandb(args.run_id, args.model, args.project)
        if result:
            model_path, extract_path = result
            visualize_model(model_path, args.episodes)
    else:
        print("Usage:")
        print("  python ppo/visualize_wandb_run.py --list-runs")
        print("  python ppo/visualize_wandb_run.py --run-id <run_id>")
        print("\nExample:")
        print("  python ppo/visualize_wandb_run.py --run-id abc123 --episodes 5")
