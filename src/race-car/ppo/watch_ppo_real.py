import random
import pygame
import os
import sys
import cv2
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO

# Add parent directory to path so we can import src modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.environments.race_car_env import RealRaceCarEnv


def watch_real_ppo_model(
    model_path="models/ppo_racecar_real_final", episodes=3, record_video=False
):
    """
    Watch the trained PPO model play the race car game using REAL game logic.
    """

    # Check if model exists
    if not os.path.exists(model_path + ".zip"):
        print(f"Model file {model_path}.zip not found!")
        print("Please train a model first using: python train_ppo_real.py")
        return

    print(f"Loading PPO model from {model_path}...")
    model = PPO.load(model_path)
    print("Model loaded successfully!")

    # Create environment with visualization enabled
    env = RealRaceCarEnv(seed_value=random.randint(1, 5000), headless=False)

    # Initialize pygame display
    pygame.init()
    screen = pygame.display.set_mode((1600, 1200))
    pygame.display.set_caption("PPO Race Car Agent - REAL Game")
    clock = pygame.time.Clock()

    # Setup video recording if requested
    video_writer = None
    if record_video:
        os.makedirs("videos", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"videos/ppo_real_demo_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_filename, fourcc, 60.0, (1600, 1200))
        print(f"Recording video to: {video_filename}")

    print(f"\nWatching PPO model for {episodes} episodes...")
    print("Press ESC to quit, SPACE to pause")
    print("Using REAL game collision detection and termination logic")

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
                action, _ = model.predict(obs, deterministic=True)

                # Take step in environment
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                step_count += 1

                # Print progress every 100 steps (every ~1.67 seconds)
                if step_count % 100 == 0:
                    time_seconds = step_count / 60.0
                    print(
                        f"  Step {step_count} ({time_seconds:.1f}s/{60}s): Distance = {info['distance']:.1f}, Speed = {info['speed']:.1f}, Reward = {episode_reward:.2f}"
                    )

                # Check if episode is done
                if terminated or truncated:
                    time_seconds = step_count / 60.0
                    print(
                        f"  Episode finished: Distance = {info['distance']:.1f}, Reward = {episode_reward:.2f}"
                    )
                    if info["crashed"]:
                        print(
                            f"  Result: CRASHED at step {step_count} ({time_seconds:.1f}s / 60s)"
                        )
                    elif info.get("race_completed", False):
                        print(
                            f"  Result: RACE COMPLETED! Full 60 seconds ({step_count} steps)"
                        )
                    else:
                        print(f"  Result: TIME LIMIT ({time_seconds:.1f}s / 60s)")
                    break

            # Render the game using REAL game state
            screen.fill((0, 0, 0))

            # Import STATE from the actual game
            import src.game.core as core

            STATE = core.STATE

            # Draw road
            if hasattr(STATE, "road") and STATE.road:
                if hasattr(STATE.road, "surface") and STATE.road.surface:
                    screen.blit(STATE.road.surface, (0, 0))

                # Draw walls
                if hasattr(STATE.road, "walls"):
                    for wall in STATE.road.walls:
                        if hasattr(wall, "draw"):
                            wall.draw(screen)

            # Draw cars
            if hasattr(STATE, "cars") and STATE.cars:
                for car in STATE.cars:
                    if hasattr(car, "sprite") and car.sprite:
                        screen.blit(car.sprite, (car.x, car.y))
                        # Draw bounding box
                        bounds = car.get_bounds()
                        color = (255, 255, 0) if car == STATE.ego else (255, 0, 0)
                        pygame.draw.rect(screen, color, bounds, width=2)

            # Draw sensors
            if hasattr(STATE, "sensors") and STATE.sensors:
                for sensor in STATE.sensors:
                    if hasattr(sensor, "draw"):
                        sensor.draw(screen)

            # Draw HUD info
            font = pygame.font.Font(None, 36)
            time_seconds = step_count / 60.0
            texts = [
                f"Episode: {episode + 1}/{episodes}",
                f"Time: {time_seconds:.1f}s / 60s",
                f"Step: {step_count} / 3600",
                f"Distance: {info.get('distance', 0):.0f}",
                f"Speed: {info.get('speed', 0):.1f}",
                f"Reward: {episode_reward:.1f}",
                f"Cars nearby: {info.get('cars_nearby', 0)}",
                f"Crashed: {info.get('crashed', False)}",
                f"Race Complete: {info.get('race_completed', False)}",
                "",
                "REAL Game Mode (60s races)",
                "Controls:",
                "ESC - Quit",
                "SPACE - Pause/Resume",
            ]

            for i, text in enumerate(texts):
                if text:  # Skip empty lines
                    if text == "REAL Game Mode":
                        color = (0, 255, 0)  # Green for real mode
                    elif text.startswith("Controls"):
                        color = (255, 255, 0)  # Yellow for controls
                    elif text.startswith("Crashed") and info.get("crashed", False):
                        color = (255, 0, 0)  # Red for crash
                    else:
                        color = (255, 255, 255)  # White for normal text

                    text_surface = font.render(text, True, color)
                    screen.blit(text_surface, (10, 10 + i * 30))

            # Show pause indicator
            if paused:
                pause_font = pygame.font.Font(None, 72)
                pause_text = pause_font.render("PAUSED", True, (255, 0, 0))
                screen.blit(
                    pause_text,
                    (screen.get_width() // 2 - 100, screen.get_height() // 2),
                )

            # Show crash indicator
            if info.get("crashed", False):
                crash_font = pygame.font.Font(None, 72)
                crash_text = crash_font.render("CRASHED!", True, (255, 0, 0))
                screen.blit(
                    crash_text,
                    (screen.get_width() // 2 - 120, screen.get_height() // 2 + 100),
                )

            pygame.display.flip()

            # Record frame if video recording is enabled
            if video_writer is not None:
                # Capture the screen surface
                frame = pygame.surfarray.array3d(screen)
                # Convert from (width, height, channels) to (height, width, channels)
                frame = np.transpose(frame, (1, 0, 2))
                # Convert RGB to BGR for OpenCV
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame)

            clock.tick(60)  # 60 FPS

    print("\nDemo complete!")

    # Clean up video recording
    if video_writer is not None:
        video_writer.release()
        print(f"Video saved to: {video_filename}")

    pygame.quit()
    env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Watch trained PPO model play Race Car using REAL game logic"
    )
    parser.add_argument(
        "--model", default="models/ppo_racecar_real_final", help="Path to trained model"
    )
    parser.add_argument(
        "--episodes", type=int, default=3, help="Number of episodes to watch"
    )
    parser.add_argument(
        "--record", action="store_true", help="Record video of the gameplay"
    )

    args = parser.parse_args()

    print("PPO Race Car Visualization - REAL Game Mode")
    print("=" * 50)

    watch_real_ppo_model(args.model, args.episodes, args.record)
