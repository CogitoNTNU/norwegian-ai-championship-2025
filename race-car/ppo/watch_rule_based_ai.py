import pygame
import numpy as np
import time
from typing import Dict, List
from race_car_gym_env import RaceCarEnv
import sys
import os

# Add parent directory to path to import game modules if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RuleBasedRaceCarAI:
    """
    Rule-based AI for the race car game with proper state management.
    Exact implementation of the API's predict_actions_from_game_bot function.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.action_map = {
            "NOTHING": 0,
            "ACCELERATE": 1,
            "DECELERATE": 2,
            "STEER_LEFT": 3,
            "STEER_RIGHT": 4,
        }
        self.action_names = {v: k for k, v in self.action_map.items()}
        self.reset()

    def reset(self):
        """Reset the AI state for a new episode."""
        self.action_queue = []
        self.current_maneuver = None
        self.maneuver_start_step = 0
        self.steps_since_last_decision = 0
        if self.verbose:
            print("🔄 AI state reset")

    def predict_actions_from_game_bot(self, request_data: dict) -> List[str]:
        """
        Exact copy of the API's predict_actions_from_game_bot function.
        """
        # Extract sensor data
        sensors = request_data.get("sensors", {})
        velocity = request_data.get("velocity", {"x": 10, "y": 0})
        did_crash = request_data.get("did_crash", False)

        # If crashed, return nothing
        if did_crash:
            return ["NOTHING"]

        # Get front sensor reading (assuming sensor_0 or 'front' is the front sensor)
        front_sensor = (
            sensors.get("sensor_0") or sensors.get("front") or sensors.get("0")
        )
        back_sensor = sensors.get("sensor_4") or sensors.get("back") or sensors.get("4")

        # Simulate the core.py bot logic
        actions = []

        # Check if there's an obstacle in front
        if (
            front_sensor is not None and front_sensor < 999
        ):  # Obstacle detected in front
            # Determine if we should change lanes based on left/right sensors
            left_sensors = []
            right_sensors = []

            # Collect left and right sensor readings
            for sensor_key, reading in sensors.items():
                if reading is not None:
                    if "left" in str(sensor_key).lower():
                        left_sensors.append(reading)
                    elif "right" in str(sensor_key).lower():
                        right_sensors.append(reading)

            # Calculate sums for lane change decision
            left_sum = sum(left_sensors) if left_sensors else 0
            right_sum = sum(right_sensors) if right_sensors else 0
            min_safe_distance = 340
            higher_threshold_sensors = [
                "front_right_front",
                "back_right_back",
                "back_left_back",
                "front_left_front",
            ]

            # Check if lanes are blocked with different thresholds
            left_blocked = False
            right_blocked = False

            for sensor_key, reading in sensors.items():
                if reading is not None:
                    # Use higher threshold for specific sensors
                    threshold = (
                        500
                        if str(sensor_key) in higher_threshold_sensors
                        else min_safe_distance
                    )

                    if "left" in str(sensor_key).lower() and reading < threshold:
                        left_blocked = True
                    elif "right" in str(sensor_key).lower() and reading < threshold:
                        right_blocked = True

            if left_blocked and right_blocked:
                # Both directions blocked - slow down
                actions.extend(["DECELERATE"])
            elif left_blocked:
                # Left blocked, go right if possible
                actions.extend(["STEER_RIGHT"] * 48)  # First phase
                actions.extend(["STEER_LEFT"] * 48)  # Second phase to straighten
            elif right_blocked:
                # Right blocked, go left if possible
                actions.extend(["STEER_LEFT"] * 48)  # First phase
                actions.extend(["STEER_RIGHT"] * 48)  # Second phase to straighten
            elif left_sum > right_sum:
                # Left is clearer, go left
                actions.extend(["STEER_LEFT"] * 48)
                actions.extend(["STEER_RIGHT"] * 48)
            else:
                # Right is clearer or equal, go right
                actions.extend(["STEER_RIGHT"] * 48)
                actions.extend(["STEER_LEFT"] * 48)

        # Check back sensor for approaching cars
        elif (
            back_sensor is not None and back_sensor < 800
        ):  # Car approaching from behind
            # Similar lane change logic but more urgent
            left_sensors = []
            right_sensors = []

            for sensor_key, reading in sensors.items():
                if reading is not None:
                    if "left" in str(sensor_key).lower():
                        left_sensors.append(reading)
                    elif "right" in str(sensor_key).lower():
                        right_sensors.append(reading)

            left_sum = sum(left_sensors) if left_sensors else 0
            right_sum = sum(right_sensors) if right_sensors else 0

            if left_sum > right_sum:
                actions.extend(["STEER_LEFT"] * 48)
                actions.extend(["STEER_RIGHT"] * 48)  # Changed from 47 to 48
            else:
                actions.extend(["STEER_RIGHT"] * 48)
                actions.extend(["STEER_LEFT"] * 48)  # Changed from 47 to 48

        # Default behavior - accelerate if no immediate threats
        if not actions:
            current_speed = abs(velocity.get("x", 10))
            if current_speed < 40:  # Target speed
                actions.append("ACCELERATE")
            else:
                actions.append("NOTHING")

        # Ensure we return at least one action and it's always a list
        if not actions:
            actions = ["ACCELERATE"]

        return actions

    def get_sensor_data_from_observation(self, observation: np.ndarray) -> Dict:
        """Extract sensor data from observation in the format expected by the API."""
        # Observation format: [sensors..., vel_x, vel_y, lane_pos, heading]
        num_sensors = len(observation) - 4

        # Convert normalized sensors back to distance readings
        sensors = {}

        # Sensor configuration based on your sensor_options
        sensor_angles = [
            (90, "front"),
            (135, "right_front"),
            (180, "right_side"),
            (225, "right_back"),
            (270, "back"),
            (315, "left_back"),
            (0, "left_side"),
            (45, "left_front"),
            (22.5, "left_side_front"),
            (67.5, "front_left_front"),
            (112.5, "front_right_front"),
            (157.5, "right_side_front"),
            (202.5, "right_side_back"),
            (247.5, "back_right_back"),
            (292.5, "back_left_back"),
            (337.5, "left_side_back"),
        ]

        # Map sensor indices to all possible naming conventions
        for i in range(num_sensors):
            # Convert from normalized (-1 to 1) to distance (0 to 1000)
            distance = (observation[i] + 1.0) * 500.0

            # Add multiple naming conventions for robustness
            sensors[f"sensor_{i}"] = distance
            sensors[str(i)] = distance

            # Add descriptive names based on sensor configuration
            if i < len(sensor_angles):
                angle, name = sensor_angles[i]
                sensors[name] = distance

                # Also add index-based names for compatibility
                if i == 0:
                    sensors["sensor_0"] = distance
                elif i == 4:
                    sensors["sensor_4"] = distance

        # Extract velocity
        vel_x = observation[num_sensors] * 30.0  # Denormalize
        vel_y = observation[num_sensors + 1] * 10.0

        return {
            "sensors": sensors,
            "velocity": {"x": vel_x, "y": vel_y},
            "did_crash": False,
        }

    def get_action(
        self, observation: np.ndarray, step: int, crashed: bool = False
    ) -> int:
        """Get the next action based on current observation."""
        self.steps_since_last_decision += 1

        # If we have queued actions, execute them
        if self.action_queue:
            action_name = self.action_queue.pop(0)

            # Log maneuver progress
            if self.verbose and len(self.action_queue) > 0:
                if self.current_maneuver and len(self.action_queue) % 10 == 0:
                    total_actions = len(self.action_queue)
                    if total_actions > 40:  # Lane change maneuver
                        phase = "Steering" if total_actions > 47 else "Correcting"
                        print(
                            f"  {phase}: {action_name} ({len(self.action_queue)} steps remaining)"
                        )

            return self.action_map[action_name]

        # Get data in API format
        data = self.get_sensor_data_from_observation(observation)
        request_data = {
            "sensors": data["sensors"],
            "velocity": data["velocity"],
            "did_crash": crashed,
        }

        # Debug logging every 100 steps if verbose
        if self.verbose and step % 100 == 0:
            print(f"\n📊 Step {step} sensor check:")
            front_val = data["sensors"].get("front", None)
            back_val = data["sensors"].get("back", None)
            left_val = data["sensors"].get("left_side", None)
            right_val = data["sensors"].get("right_side", None)

            if front_val is not None:
                print(f"  Front: {front_val:.1f}")
            else:
                print("  Front: N/A")

            if back_val is not None:
                print(f"  Back: {back_val:.1f}")
            else:
                print("  Back: N/A")

            if left_val is not None:
                print(f"  Left: {left_val:.1f}")
            else:
                print("  Left: N/A")

            if right_val is not None:
                print(f"  Right: {right_val:.1f}")
            else:
                print("  Right: N/A")

            print(f"  Speed: {data['velocity']['x']:.1f}")

        # Call the API logic
        actions = self.predict_actions_from_game_bot(request_data)

        # Queue up all actions
        self.action_queue = actions.copy()
        self.steps_since_last_decision = 0

        # Determine and log what maneuver we're starting
        if self.verbose and len(actions) > 1:
            # Check what type of maneuver
            if len(actions) == 95:  # Lane change maneuver
                if "STEER_LEFT" in actions[0]:
                    self.current_maneuver = "LANE_CHANGE_LEFT"
                    print(f"\n⬅️  Lane change LEFT initiated at step {step}")
                else:
                    self.current_maneuver = "LANE_CHANGE_RIGHT"
                    print(f"\n➡️  Lane change RIGHT initiated at step {step}")
                self.maneuver_start_step = step

                # Show why we're changing lanes
                front = data["sensors"].get("front", 999)
                back = data["sensors"].get("back", 999)
                if front < 999:
                    print(f"  Reason: Obstacle ahead at {front:.1f}")
                elif back < 800:
                    print(f"  Reason: Car approaching from behind at {back:.1f}")
            elif "DECELERATE" in actions:
                print("\n🚫 Both lanes blocked - Braking!")

        # Return first action
        if self.action_queue:
            return self.action_map[self.action_queue.pop(0)]
        else:
            return self.action_map["NOTHING"]


def run_rule_based_ai_demo(
    episodes: int = 5, render_speed: float = 1.0, verbose: bool = True
):
    """
    Run the rule-based AI in the race car environment with proper state management.
    """
    print("🏁 Starting Rule-Based AI Demo (Fixed Version)")
    print("=" * 50)

    # Stats tracking
    episode_stats = []

    # Create environment with proper error handling
    try:
        env = RaceCarEnv(render_mode="human", seed="demo_seed")
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        print("Make sure the RaceCarEnv is properly initialized.")
        return

    # Create AI instance
    ai = RuleBasedRaceCarAI(verbose=verbose)

    try:
        for episode in range(episodes):
            print(f"\n🏎️  Episode {episode + 1}/{episodes}")
            print("-" * 30)

            # Reset environment and AI
            try:
                obs, info = env.reset()
                ai.reset()  # Reset AI state
            except Exception as e:
                print(f"❌ Error resetting environment: {e}")
                continue

            done = False
            episode_distance = 0
            episode_reward = 0
            episode_steps = 0
            action_counts = {name: 0 for name in ai.action_map.keys()}
            maneuvers_completed = 0

            # Performance tracking
            start_time = time.time()

            while not done:
                try:
                    # Get AI decision with step number
                    action = ai.get_action(
                        obs, step=episode_steps, crashed=info.get("crashed", False)
                    )
                    action_name = ai.action_names[action]
                    action_counts[action_name] += 1

                    # Track completed maneuvers
                    if ai.current_maneuver and len(ai.action_queue) == 0:
                        maneuvers_completed += 1
                        if verbose:
                            print(
                                f"  ✅ {ai.current_maneuver} completed after {episode_steps - ai.maneuver_start_step} steps"
                            )
                        ai.current_maneuver = None

                    # Take action
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                    episode_reward += reward
                    episode_steps += 1

                    # Control rendering speed
                    if render_speed < 1.0:
                        time.sleep((1.0 - render_speed) * 0.016)

                    # Handle pygame events
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            done = True
                            episodes = episode + 1
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_SPACE:
                                # Pause
                                print("\n⏸️  PAUSED - Press SPACE to continue")
                                paused = True
                                while paused:
                                    for pause_event in pygame.event.get():
                                        if (
                                            pause_event.type == pygame.KEYDOWN
                                            and pause_event.key == pygame.K_SPACE
                                        ):
                                            paused = False
                                        elif pause_event.type == pygame.QUIT:
                                            paused = False
                                            done = True
                                    time.sleep(0.1)
                            elif event.key == pygame.K_r:
                                # Reset episode
                                print("\n🔄 Resetting episode...")
                                done = True
                            elif event.key == pygame.K_ESCAPE:
                                # Exit
                                print("\n🛑 Exiting...")
                                episodes = episode + 1
                                done = True

                except Exception as e:
                    print(f"❌ Error during episode: {e}")
                    done = True
                    break

            # Episode complete
            episode_time = time.time() - start_time
            episode_distance = info["distance"]

            print(f"\n✅ Episode {episode + 1} Complete!")
            print(f"  Final Distance: {episode_distance:.1f}")
            print(f"  Total Reward: {episode_reward:.2f}")
            print(f"  Steps: {episode_steps}")
            print(f"  Time: {episode_time:.1f}s")
            print(f"  Crashed: {'Yes' if info['crashed'] else 'No'}")
            print(f"  Lane Changes: {maneuvers_completed}")

            # Action distribution
            print("\n📊 Action Distribution:")
            total_actions = sum(action_counts.values())
            for action_name, count in sorted(
                action_counts.items(), key=lambda x: x[1], reverse=True
            ):
                percentage = (count / total_actions) * 100 if total_actions > 0 else 0
                if count > 0:  # Only show actions that were used
                    print(f"  {action_name}: {count} ({percentage:.1f}%)")

            episode_stats.append(
                {
                    "episode": episode + 1,
                    "distance": episode_distance,
                    "reward": episode_reward,
                    "steps": episode_steps,
                    "crashed": info["crashed"],
                    "time": episode_time,
                    "maneuvers": maneuvers_completed,
                }
            )

    except KeyboardInterrupt:
        print("\n\n🛑 Demo interrupted by user")

    finally:
        # Summary statistics
        if episode_stats:
            print("\n" + "=" * 50)
            print("📈 Summary Statistics")
            print("=" * 50)

            distances = [s["distance"] for s in episode_stats]
            rewards = [s["reward"] for s in episode_stats]
            steps = [s["steps"] for s in episode_stats]
            maneuvers = [s["maneuvers"] for s in episode_stats]
            crashes = [s["crashed"] for s in episode_stats]

            print(f"Episodes Run: {len(episode_stats)}")
            print(
                f"Average Distance: {np.mean(distances):.1f} ± {np.std(distances):.1f}"
            )
            print(f"Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
            print(f"Average Steps: {np.mean(steps):.0f}")
            print(f"Average Lane Changes: {np.mean(maneuvers):.1f}")
            print(f"Crash Rate: {sum(crashes) / len(crashes) * 100:.1f}%")

            # Show each episode's performance
            print("\n📊 Episode Details:")
            for stat in episode_stats:
                status = "❌ Crashed" if stat["crashed"] else "✅ Completed"
                print(
                    f"  Episode {stat['episode']}: {stat['distance']:.1f}m, "
                    f"{stat['maneuvers']} lane changes, {status}"
                )

            # Best episode
            best_episode = max(episode_stats, key=lambda x: x["distance"])
            print(f"\n🏆 Best Episode: #{best_episode['episode']}")
            print(f"  Distance: {best_episode['distance']:.1f}")
            print(f"  Lane Changes: {best_episode['maneuvers']}")

        env.close()
        print("\n👋 Demo complete!")


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Watch rule-based AI play Race Car")
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to watch (default: 5)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Rendering speed multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Disable verbose decision logging"
    )

    args = parser.parse_args()

    # Run the demo
    print("🎮 Race Car Rule-Based AI Demo")
    print("\nThis demo shows the exact algorithm from your API.")
    print("\nControls:")
    print("  SPACE - Pause/Resume")
    print("  R - Reset current episode")
    print("  ESC - Exit")
    print("  Close Window - Quit")
    print("\nPress CTRL+C to stop early\n")

    # Run with parsed arguments
    run_rule_based_ai_demo(
        episodes=args.episodes, render_speed=args.speed, verbose=not args.quiet
    )
