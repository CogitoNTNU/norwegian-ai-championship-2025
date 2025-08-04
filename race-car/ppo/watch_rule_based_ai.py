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
    Modified to check front diagonal sensors before lane changes and
    handle dynamic obstacles during maneuvers.
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
        self.lane_change_direction = None
        self.lane_change_steps_taken = 0
        self.lane_change_phase = None  # 'steering' or 'correcting'
        self.last_front_sensor = 999
        self.emergency_brake_steps = 0
        if self.verbose:
            print("🔄 AI state reset")

    def predict_actions_from_game_bot(self, request_data: dict) -> List[str]:
        """
        Modified version that checks front diagonal sensors before lane changes
        and returns dynamic lane change commands.
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

        # Get front diagonal sensors
        front_left_front = sensors.get("front_left_front", 999)
        front_right_front = sensors.get("front_right_front", 999)

        # Simulate the core.py bot logic
        actions = []

        # Check if there's an obstacle in front
        if (
            front_sensor is not None and front_sensor < 999
        ):  # Obstacle detected in front
            # First check if we can change lanes using front diagonal sensors
            can_go_left = (
                front_left_front > 870
            )  # Minimum safe distance for lane change
            can_go_right = front_right_front > 870

            if self.verbose:
                print(f"\n🚦 Obstacle ahead at {front_sensor:.1f}")
                print(
                    f"  Front-left-front sensor: {front_left_front:.1f} (Can go left: {can_go_left})"
                )
                print(
                    f"  Front-right-front sensor: {front_right_front:.1f} (Can go right: {can_go_right})"
                )

            # If both diagonal paths are blocked, brake
            if not can_go_left and not can_go_right:
                if self.verbose:
                    print("  ⚠️ Both diagonal paths blocked - BRAKING!")
                actions.extend(["DECELERATE"] * 100)  # Brake for multiple steps
                return actions

            # If only one diagonal is clear, go that way
            if can_go_left and not can_go_right:
                if self.verbose:
                    print("  ↖️ Only left diagonal clear - going LEFT")
                actions.append("DYNAMIC_LANE_CHANGE_LEFT")
                return actions
            elif can_go_right and not can_go_left:
                if self.verbose:
                    print("  ↗️ Only right diagonal clear - going RIGHT")
                actions.append("DYNAMIC_LANE_CHANGE_RIGHT")
                return actions

            # Both diagonals are clear - choose based on middle lane preference
            if can_go_left and can_go_right:
                # Get left_side and right_side sensor readings
                left_side_sensor = (
                    sensors.get("left_side")
                    or sensors.get("sensor_6")
                    or sensors.get("6")
                    or 0
                )
                right_side_sensor = (
                    sensors.get("right_side")
                    or sensors.get("sensor_2")
                    or sensors.get("2")
                    or 0
                )

                if self.verbose:
                    print("  🎯 Both paths clear - checking middle lane preference")
                    print(f"    Left side sensor: {left_side_sensor:.1f}")
                    print(f"    Right side sensor: {right_side_sensor:.1f}")

                # Choose direction that leads to middle lane (side with more space)
                if left_side_sensor > right_side_sensor:
                    if self.verbose:
                        print(
                            f"  ↖️ Left side has more space ({left_side_sensor:.1f} > {right_side_sensor:.1f}) - going LEFT toward middle lane"
                        )
                    actions.append("DYNAMIC_LANE_CHANGE_LEFT")
                else:
                    if self.verbose:
                        print(
                            f"  ↗️ Right side has more space ({right_side_sensor:.1f} >= {left_side_sensor:.1f}) - going RIGHT toward middle lane"
                        )
                    actions.append("DYNAMIC_LANE_CHANGE_RIGHT")
                return actions

            # Fallback to original logic if something went wrong
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
                actions.append("DYNAMIC_LANE_CHANGE_RIGHT")
            elif right_blocked:
                # Right blocked, go left if possible
                actions.append("DYNAMIC_LANE_CHANGE_LEFT")
            elif left_sum > right_sum:
                # Left is clearer, go left
                actions.append("DYNAMIC_LANE_CHANGE_LEFT")
            else:
                # Right is clearer or equal, go right
                actions.append("DYNAMIC_LANE_CHANGE_RIGHT")

        # Check back sensor for approaching cars
        elif (
            back_sensor is not None and back_sensor < 800
        ):  # Car approaching from behind
            # Check front diagonal sensors for lane change safety
            can_go_left = (
                front_left_front > 300
            )  # Slightly lower threshold for back approaching
            can_go_right = front_right_front > 300

            if self.verbose:
                print(f"\n🚗 Car approaching from behind at {back_sensor:.1f}")
                print(
                    f"  Front-left-front: {front_left_front:.1f} (Can go left: {can_go_left})"
                )
                print(
                    f"  Front-right-front: {front_right_front:.1f} (Can go right: {can_go_right})"
                )

            # If both paths blocked, just accelerate to get away
            if not can_go_left and not can_go_right:
                if self.verbose:
                    print("  ⚠️ Both diagonal paths blocked - ACCELERATING!")
                actions.extend(["ACCELERATE"] * 10)
                return actions

            # Choose the clearer path with dynamic lane change
            if can_go_left and not can_go_right:
                actions.append("DYNAMIC_LANE_CHANGE_LEFT")
            elif can_go_right and not can_go_left:
                actions.append("DYNAMIC_LANE_CHANGE_RIGHT")
            elif can_go_left and can_go_right:
                # Both clear - choose based on middle lane preference
                left_side_sensor = (
                    sensors.get("left_side")
                    or sensors.get("sensor_6")
                    or sensors.get("6")
                    or 0
                )
                right_side_sensor = (
                    sensors.get("right_side")
                    or sensors.get("sensor_2")
                    or sensors.get("2")
                    or 0
                )

                if left_side_sensor > right_side_sensor:
                    actions.append("DYNAMIC_LANE_CHANGE_LEFT")
                else:
                    actions.append("DYNAMIC_LANE_CHANGE_RIGHT")
            else:
                # Fallback to original logic
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
                    actions.append("DYNAMIC_LANE_CHANGE_LEFT")
                else:
                    actions.append("DYNAMIC_LANE_CHANGE_RIGHT")

        # Default behavior - move to center lane if safe, otherwise accelerate
        if not actions:
            # Get the front sensor (90 degrees) and adjacent sensors
            front_sensor = (
                sensors.get("front")
                or sensors.get("sensor_0")
                or sensors.get("0")
                or 999
            )
            front_left = sensors.get("front_left_front") or 999
            front_right = sensors.get("front_right_front") or 999

            # Get the side sensors (0 and 180 degrees)
            left_side_sensor = (
                sensors.get("left_side")
                or sensors.get("sensor_6")
                or sensors.get("6")
                or 999
            )
            right_side_sensor = (
                sensors.get("right_side")
                or sensors.get("sensor_2")
                or sensors.get("2")
                or 999
            )

            # Check if front sensors show clear path (nothing detected)
            front_clear_threshold = 990  # Near max sensor range means nothing detected
            if (
                front_sensor > front_clear_threshold
                and front_left > front_clear_threshold
                and front_right > front_clear_threshold
            ):
                # Front is clear, now check side sensors to determine if we're not in center
                side_close_threshold = (
                    500  # If a side sensor is under this, we're too close to that edge
                )

                if self.verbose:
                    print(
                        f"\n🔍 Front clear - checking lane position (L:{left_side_sensor:.1f} R:{right_side_sensor:.1f})"
                    )

                # If left side sensor is close, we're on the LEFT side of the road, move RIGHT to center
                if (
                    left_side_sensor < side_close_threshold
                    and right_side_sensor > side_close_threshold
                ):
                    if self.verbose:
                        print(
                            f"  ↗️ In left lane ({left_side_sensor:.1f} < {side_close_threshold}) - moving RIGHT to center"
                        )
                    actions.append("DYNAMIC_LANE_CHANGE_RIGHT")
                # If right side sensor is close, we're on the RIGHT side of the road, move LEFT to center
                elif (
                    right_side_sensor < side_close_threshold
                    and left_side_sensor > side_close_threshold
                ):
                    if self.verbose:
                        print(
                            f"  ↖️ In right lane ({right_side_sensor:.1f} < {side_close_threshold}) - moving LEFT to center"
                        )
                    actions.append("DYNAMIC_LANE_CHANGE_LEFT")
                else:
                    # Either centered or both sides are equally far/close, just maintain speed
                    current_speed = abs(velocity.get("x", 10))
                    if current_speed < 40:  # Target speed
                        actions.append("ACCELERATE")
                    else:
                        actions.append("NOTHING")
            else:
                # Front not clear, just maintain speed
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
        """Get the next action based on current observation with dynamic lane change handling."""
        self.steps_since_last_decision += 1

        # Get current sensor data
        data = self.get_sensor_data_from_observation(observation)
        current_front_sensor = data["sensors"].get("front", 999)

        # Handle emergency braking during lane change
        if self.emergency_brake_steps > 0:
            self.emergency_brake_steps -= 1
            if self.verbose and self.emergency_brake_steps % 10 == 0:
                print(
                    f"  🛑 Emergency braking ({self.emergency_brake_steps} steps remaining)"
                )

            # Check if obstacle is now moving away
            if (
                current_front_sensor > self.last_front_sensor + 5
            ):  # Obstacle moving away
                if self.verbose:
                    print(
                        f"  ✅ Obstacle moving away ({self.last_front_sensor:.1f} -> {current_front_sensor:.1f})"
                    )
                self.emergency_brake_steps = 0
                # Resume lane change
                if self.lane_change_direction:
                    remaining_steps = 48 - self.lane_change_steps_taken
                    if self.lane_change_phase == "steering":
                        # Continue steering phase
                        for _ in range(remaining_steps):
                            self.action_queue.append(
                                f"STEER_{self.lane_change_direction}"
                            )
                        # Add correction phase
                        opposite_dir = (
                            "RIGHT" if self.lane_change_direction == "LEFT" else "LEFT"
                        )
                        for _ in range(48):
                            self.action_queue.append(f"STEER_{opposite_dir}")
                    else:  # correcting phase
                        # Continue correction
                        opposite_dir = (
                            "RIGHT" if self.lane_change_direction == "LEFT" else "LEFT"
                        )
                        for _ in range(remaining_steps):
                            self.action_queue.append(f"STEER_{opposite_dir}")

            self.last_front_sensor = current_front_sensor
            return self.action_map["DECELERATE"]

        # Handle active lane change
        if self.lane_change_direction and self.action_queue:
            # Check for incoming obstacle during lane change
            if current_front_sensor < 300:  # Danger threshold during maneuver
                if self.verbose:
                    print(
                        f"  ⚠️ Obstacle detected during lane change at {current_front_sensor:.1f}!"
                    )
                # Clear action queue and start emergency braking
                self.action_queue.clear()
                self.emergency_brake_steps = 50  # Brake for 50 steps
                self.last_front_sensor = current_front_sensor
                return self.action_map["DECELERATE"]

            # Continue with lane change
            action_name = self.action_queue.pop(0)

            # Track lane change progress
            if "STEER" in action_name:
                self.lane_change_steps_taken += 1
                if self.lane_change_steps_taken <= 48:
                    self.lane_change_phase = "steering"
                else:
                    self.lane_change_phase = "correcting"

            # Check if lane change is complete
            if not self.action_queue:
                if self.verbose:
                    print(f"  ✅ Lane change {self.lane_change_direction} completed")
                self.lane_change_direction = None
                self.lane_change_steps_taken = 0
                self.lane_change_phase = None

            return self.action_map[action_name]

        # If we have other queued actions, execute them
        if self.action_queue:
            action_name = self.action_queue.pop(0)
            return self.action_map[action_name]

        # Get new decision
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
            front_left_front = data["sensors"].get("front_left_front", None)
            front_right_front = data["sensors"].get("front_right_front", None)

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

            if front_left_front is not None:
                print(f"  Front-Left-Front: {front_left_front:.1f}")
            else:
                print("  Front-Left-Front: N/A")

            if front_right_front is not None:
                print(f"  Front-Right-Front: {front_right_front:.1f}")
            else:
                print("  Front-Right-Front: N/A")

            print(f"  Speed: {data['velocity']['x']:.1f}")

        # Call the API logic
        actions = self.predict_actions_from_game_bot(request_data)

        # Handle dynamic lane change commands
        if actions and actions[0].startswith("DYNAMIC_LANE_CHANGE_"):
            direction = actions[0].split("_")[-1]  # "LEFT" or "RIGHT"
            self.lane_change_direction = direction
            self.lane_change_steps_taken = 0
            self.lane_change_phase = "steering"

            if self.verbose:
                print(
                    f"\n{'⬅️' if direction == 'LEFT' else '➡️'}  Dynamic lane change {direction} initiated at step {step}"
                )

            # Queue up the lane change actions
            for _ in range(48):
                self.action_queue.append(f"STEER_{direction}")

            # Queue up the correction
            opposite_dir = "RIGHT" if direction == "LEFT" else "LEFT"
            for _ in range(48):
                self.action_queue.append(f"STEER_{opposite_dir}")

            # Return first action
            return self.action_map[self.action_queue.pop(0)]

        # Queue up all other actions normally
        self.action_queue = actions.copy()
        self.steps_since_last_decision = 0

        # Determine and log what maneuver we're starting
        if self.verbose and len(actions) > 1:
            # Check what type of maneuver
            if "DECELERATE" in actions and len(actions) > 5:
                print("\n🚫 Both lanes blocked - Braking!")
            elif "ACCELERATE" in actions and len(actions) > 5:
                print("\n💨 Path blocked behind - Accelerating!")

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
    print(
        "🏁 Starting Rule-Based AI Demo (Dynamic Lane Changes with Emergency Braking)"
    )
    print("=" * 50)

    # Stats tracking
    episode_stats = []

    # Create environment with proper error handling
    try:
        env = RaceCarEnv(render_mode="human", seed=None)
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

                    # Print when decelerate action is taken
                    if action_name == "DECELERATE":
                        print(f"🛑 DECELERATE action at step {episode_steps}")

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
    print(
        "\nThis demo shows the algorithm with dynamic lane changes and emergency braking."
    )
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
