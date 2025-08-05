import sqlite3
from typing import List, Dict, Tuple
import json
from datetime import datetime
import threading


class LaneChangeController:
    def __init__(self, verbose: bool = True):
        """
        Initialize the lane change controller with in-memory SQLite database.

        Args:
            verbose: Whether to print debug information
        """
        self.verbose = verbose
        # Use :memory: with check_same_thread=False for thread safety
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        # Create a lock for thread-safe database operations
        self.db_lock = threading.Lock()

        # Safe distances mapping to sensor positions
        # [front_right_front, right_front, right_side_front, right_side, right_side_back, right_back, back_right_back]
        self.safe_distances = [999.0, 462.0, 354.0, 327.0, 354.0, 462.0, 999.9]

        # Sensor names for right side (left side mirrors these)
        self.right_sensors = [
            "front_right_front",
            "right_front",
            "right_side_front",
            "right_side",
            "right_side_back",
            "right_back",
            "back_right_back",
        ]

        self.left_sensors = [
            "front_left_front",
            "left_front",
            "left_side_front",
            "left_side",
            "left_side_back",
            "left_back",
            "back_left_back",
        ]

        # Lane boundaries (0 = leftmost, 4 = rightmost)
        self.min_lane = 0
        self.max_lane = 4
        self.center_lane = 2

        # Steering duration for lane changes
        self.steer_duration = 48
        
        # Progressive steering state for centering
        self.centering_active = False
        self.centering_direction = None  # "left" or "right"
        self.centering_steps_sent = 0
        self.centering_target_lane = None
        self.centering_original_lane = None

        # Initialize database
        self._init_database()

    def get_steer_duration(self, from_lane: int, to_lane: int) -> int:
        """Get steering duration based on lane change."""
        # 44 for lane changes between 1 and 2, 48 otherwise
        if (from_lane == 1 and to_lane == 2) or (from_lane == 2 and to_lane == 1):
            return 44
        else:
            return self.steer_duration

    def start_centering(self, current_lane: int, target_lane: int) -> str:
        """Start progressive centering process."""
        self.centering_active = True
        self.centering_original_lane = current_lane
        self.centering_target_lane = target_lane
        self.centering_steps_sent = 0
        
        if target_lane < current_lane:
            self.centering_direction = "left"
            return "STEER_LEFT"
        else:
            self.centering_direction = "right"
            return "STEER_RIGHT"

    def continue_centering(self, sensors: Dict) -> str:
        """Continue centering process or abort if unsafe."""
        if not self.centering_active:
            return "NOTHING"
            
        # Check if we should abort (only if 24 or fewer steps sent)
        if self.centering_steps_sent <= 24:
            # Check safety in the direction we're steering
            is_safe, _ = self.check_lane_safety(sensors, self.centering_direction)
            if not is_safe:
                return self.abort_centering()
        
        # Continue steering
        self.centering_steps_sent += 1
        total_duration = self.get_steer_duration(self.centering_original_lane, self.centering_target_lane)
        
        if self.centering_steps_sent < total_duration:
            # Still in first phase (steering toward target)
            return f"STEER_{self.centering_direction.upper()}"
        elif self.centering_steps_sent < total_duration * 2:
            # Second phase (steering back to center)
            opposite_direction = "LEFT" if self.centering_direction == "right" else "RIGHT"
            return f"STEER_{opposite_direction}"
        else:
            # Centering complete
            self.complete_centering()
            return "NOTHING"

    def abort_centering(self) -> str:
        """Abort centering and return to original lane."""
        if not self.centering_active:
            return "NOTHING"
            
        steps_sent = self.centering_steps_sent
        self.centering_active = False
        
        if self.verbose:
            print(f"  🚨 Aborting centering after {steps_sent} steps")
        
        # Return sequence: 2*steps in opposite direction, then steps in original direction
        opposite_direction = "RIGHT" if self.centering_direction == "left" else "LEFT"
        original_direction = self.centering_direction.upper()
        
        return_sequence = ([f"STEER_{opposite_direction}"] * (2 * steps_sent)) + \
                         ([f"STEER_{original_direction}"] * steps_sent)
        
        # Reset state
        self.centering_steps_sent = 0
        self.centering_direction = None
        self.centering_target_lane = None
        self.centering_original_lane = None
        
        return return_sequence

    def complete_centering(self):
        """Complete centering process and update lane."""
        if self.centering_active:
            self.update_lane(self.centering_target_lane, f"LANE_CHANGE_{self.centering_direction.upper()}")
            if self.verbose:
                print(f"  ✅ Centering complete, now in lane {self.centering_target_lane}")
        
        # Reset state
        self.centering_active = False
        self.centering_steps_sent = 0
        self.centering_direction = None
        self.centering_target_lane = None
        self.centering_original_lane = None

    def _init_database(self):
        """Create necessary tables if they don't exist."""
        cursor = self.conn.cursor()

        # Table for tracking current lane and state
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vehicle_state (
                id INTEGER PRIMARY KEY,
                current_lane INTEGER NOT NULL DEFAULT 2,
                last_action TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Table for sensor history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sensor_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sensors_json TEXT NOT NULL,
                velocity_json TEXT NOT NULL,
                action_taken TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Initialize with default state if empty
        cursor.execute("SELECT COUNT(*) FROM vehicle_state")
        if cursor.fetchone()[0] == 0:
            cursor.execute(
                "INSERT INTO vehicle_state (current_lane) VALUES (?)",
                (self.center_lane,),
            )

        self.conn.commit()

    def get_current_lane(self) -> int:
        """Get the current lane from database."""
        with self.db_lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT current_lane FROM vehicle_state ORDER BY id DESC LIMIT 1"
            )
            result = cursor.fetchone()
            return result[0] if result else self.center_lane

    def update_lane(self, new_lane: int, action: str):
        """Update the current lane in database."""
        with self.db_lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE vehicle_state SET current_lane = ?, last_action = ?, timestamp = ? WHERE id = 1",
                (new_lane, action, datetime.now()),
            )
            self.conn.commit()

    def save_sensor_reading(self, sensors: Dict, velocity: Dict, action: str):
        """Save sensor reading to history."""
        with self.db_lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO sensor_history (sensors_json, velocity_json, action_taken) VALUES (?, ?, ?)",
                (json.dumps(sensors), json.dumps(velocity), action),
            )
            self.conn.commit()

    def get_sensor_value(self, sensors: Dict, sensor_name: str) -> float:
        """Get sensor value, treating None as 1000.0 (far away)."""
        value = sensors.get(sensor_name)
        return 1000.0 if value is None else value

    def check_lane_safety(self, sensors: Dict, target_lane: str) -> Tuple[bool, str]:
        """
        Check if it's safe to change to the target lane.

        Args:
            sensors: Dictionary of sensor readings
            target_lane: 'left' or 'right'

        Returns:
            Tuple of (is_safe, reason)
        """
        current_lane = self.get_current_lane()

        # Check if we're at the edge lanes
        if target_lane == "left" and current_lane == self.min_lane:
            return False, "Already in leftmost lane"
        if target_lane == "right" and current_lane == self.max_lane:
            return False, "Already in rightmost lane"

        # Select appropriate sensor list
        sensor_list = self.left_sensors if target_lane == "left" else self.right_sensors

        # Check each sensor against its safe distance
        for idx, sensor_name in enumerate(sensor_list):
            sensor_value = self.get_sensor_value(sensors, sensor_name)
            safe_distance = self.safe_distances[idx]

            # Adjust safe distance for wall-adjacent lanes
            if current_lane == 1 and sensor_name in [
                "back_left_back",
                "front_left_front",
            ]:
                safe_distance = 856
            elif current_lane == 3 and sensor_name in [
                "back_right_back",
                "front_right_front",
            ]:
                safe_distance = 856

            if sensor_value < safe_distance:
                if self.verbose:
                    print(
                        f"  ⚠️ {sensor_name}: {sensor_value:.1f} < {safe_distance:.1f} (unsafe)"
                    )
                return False, f"{sensor_name} obstacle at {sensor_value:.1f}"

        if self.verbose:
            print(f"  ✅ All {target_lane} sensors clear")

        return True, "Lane change safe"

    def get_front_sensor(self, sensors: Dict) -> float:
        """Get front sensor reading."""
        value = sensors.get("front") or sensors.get("sensor_0") or sensors.get("0")
        return 1000.0 if value is None else value

    def get_back_sensor(self, sensors: Dict) -> float:
        """Get back sensor reading."""
        value = sensors.get("back") or sensors.get("sensor_4") or sensors.get("4")
        return 1000.0 if value is None else value

    def get_previous_front_distance(self) -> float:
        """Get the previous front sensor distance from database."""
        with self.db_lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT sensors_json FROM sensor_history 
                ORDER BY timestamp DESC LIMIT 1
            """)
            result = cursor.fetchone()
            if result:
                sensors = json.loads(result[0])
                return self.get_front_sensor(sensors)
            return 1000.0

    def calculate_speed_adjustment(
        self, current_front_distance: float, previous_front_distance: float
    ) -> str:
        """
        Calculate speed adjustment based on distance changes to car in front.

        Args:
            current_front_distance: Current distance to car in front
            previous_front_distance: Previous distance to car in front

        Returns:
            Action to take: "ACCELERATE", "DECELERATE", or "NOTHING"
        """
        distance_change = current_front_distance - previous_front_distance

        # If distance is increasing, car in front is moving away or faster
        if distance_change > 2:
            return "ACCELERATE"
        # If distance is decreasing rapidly, car in front is slower
        elif distance_change < -5:
            return "DECELERATE"
        # If distance is stable or slowly changing, maintain speed
        else:
            return "NOTHING"

    def predict_actions(self, request_data: dict) -> List[str]:
        """
        Predict actions based on sensor data with safety checks.

        Args:
            request_data: Dictionary containing sensors, velocity, and crash status

        Returns:
            List of actions to take
        """
        # Extract data
        sensors = request_data.get("sensors", {})
        velocity = request_data.get("velocity", {"x": 10, "y": 0})
        did_crash = request_data.get("did_crash", False)

        # Save sensor reading
        action_to_save = "PENDING"
        self.save_sensor_reading(sensors, velocity, action_to_save)

        if did_crash:
            # Reset centering if crashed
            self.centering_active = False
            return ["NOTHING"]

        # Handle ongoing centering process
        if self.centering_active:
            action = self.continue_centering(sensors)
            if isinstance(action, list):
                # Abort sequence returned
                return action
            elif action != "NOTHING":
                return [action]
            # If action is "NOTHING", centering is complete, continue with normal logic

        current_lane = self.get_current_lane()

        # Get key sensors
        front_sensor = self.get_front_sensor(sensors)
        back_sensor = self.get_back_sensor(sensors)

        if self.verbose:
            print(f"\n🚗 Current lane: {current_lane}")
            print(f"📡 Front: {front_sensor:.1f}, Back: {back_sensor:.1f}")
            # Print all sensor values for debugging
            print("📊 All sensors:")
            for sensor_name, value in sensors.items():
                print(f"   {sensor_name}: {value}")

        # Check for front obstacle
        if front_sensor < 950:  # Obstacle ahead
            if self.verbose:
                print(f"🚦 Obstacle ahead at {front_sensor:.1f}")

            # For close obstacles (under 570), check velocity matching first
            if front_sensor < 570:
                previous_front_distance = self.get_previous_front_distance()
                distance_change = front_sensor - previous_front_distance

                # If velocities are roughly matching (distance change is small), allow lane changes
                if abs(distance_change) <= 5:  # Velocities are close enough
                    if self.verbose:
                        print(
                            f"  🚗 Velocities matching (change: {distance_change:.1f}), checking lanes"
                        )

                    # Check both directions for safety
                    left_safe, left_reason = self.check_lane_safety(sensors, "left")
                    right_safe, right_reason = self.check_lane_safety(sensors, "right")
                else:
                    # Velocities don't match, just adjust speed
                    speed_action = self.calculate_speed_adjustment(
                        front_sensor, previous_front_distance
                    )
                    if self.verbose:
                        print(
                            f"  🚗 Velocities not matching (change: {distance_change:.1f}), adjusting speed: {speed_action}"
                        )
                    return [speed_action]
            else:
                # For farther obstacles, check lanes normally
                left_safe, left_reason = self.check_lane_safety(sensors, "left")
                right_safe, right_reason = self.check_lane_safety(sensors, "right")

            if self.verbose:
                print(f"  Left lane: {'✅' if left_safe else '❌'} {left_reason}")
                print(f"  Right lane: {'✅' if right_safe else '❌'} {right_reason}")

            # Decision logic
            if not left_safe and not right_safe:
                # Both blocked - match speed of car in front instead of hard braking
                previous_front_distance = self.get_previous_front_distance()
                speed_action = self.calculate_speed_adjustment(
                    front_sensor, previous_front_distance
                )

                if self.verbose:
                    print(f"  🚗 Both lanes blocked - matching front car speed")
                    print(
                        f"  📊 Distance change: {front_sensor - previous_front_distance:.1f}"
                    )
                    print(f"  🎮 Speed action: {speed_action}")

                return [speed_action]

            elif left_safe and right_safe:
                # Both safe - prefer moving toward center
                if current_lane < self.center_lane:
                    # We're left of center, go right
                    if self.verbose:
                        print("  ➡️ Both safe, moving RIGHT toward center")
                    new_lane = current_lane + 1
                    steer_duration = self.get_steer_duration(current_lane, new_lane)
                    self.update_lane(new_lane, "LANE_CHANGE_RIGHT")
                    # Return steering sequence: right, then left to straighten
                    return (["STEER_RIGHT"] * steer_duration) + (
                        ["STEER_LEFT"] * steer_duration
                    )
                elif current_lane > self.center_lane:
                    # We're right of center, go left
                    if self.verbose:
                        print("  ⬅️ Both safe, moving LEFT toward center")
                    new_lane = current_lane - 1
                    steer_duration = self.get_steer_duration(current_lane, new_lane)
                    self.update_lane(new_lane, "LANE_CHANGE_LEFT")
                    # Return steering sequence: left, then right to straighten
                    return (["STEER_LEFT"] * steer_duration) + (
                        ["STEER_RIGHT"] * steer_duration
                    )
                else:
                    # At center, prefer left
                    if self.verbose:
                        print("  ⬅️ Both safe at center, preferring LEFT")
                    new_lane = current_lane - 1
                    steer_duration = self.get_steer_duration(current_lane, new_lane)
                    self.update_lane(new_lane, "LANE_CHANGE_LEFT")
                    return (["STEER_LEFT"] * steer_duration) + (
                        ["STEER_RIGHT"] * steer_duration
                    )

            elif left_safe:
                # Only left is safe
                if self.verbose:
                    print("  ⬅️ Only left safe - changing LEFT")
                new_lane = current_lane - 1
                steer_duration = self.get_steer_duration(current_lane, new_lane)
                self.update_lane(new_lane, "LANE_CHANGE_LEFT")
                return (["STEER_LEFT"] * steer_duration) + (
                    ["STEER_RIGHT"] * steer_duration
                )

            else:  # only right_safe
                # Only right is safe
                if self.verbose:
                    print("  ➡️ Only right safe - changing RIGHT")
                new_lane = current_lane + 1
                steer_duration = self.get_steer_duration(current_lane, new_lane)
                self.update_lane(new_lane, "LANE_CHANGE_RIGHT")
                return (["STEER_RIGHT"] * steer_duration) + (
                    ["STEER_LEFT"] * steer_duration
                )

        # Check for car approaching from behind
        elif back_sensor < 800:
            if self.verbose:
                print(f"🚗 Car approaching from behind at {back_sensor:.1f}")

            # Similar logic but more urgent
            left_safe, left_reason = self.check_lane_safety(sensors, "left")
            right_safe, right_reason = self.check_lane_safety(sensors, "right")

            if self.verbose:
                print(f"  Left lane: {'✅' if left_safe else '❌'} {left_reason}")
                print(f"  Right lane: {'✅' if right_safe else '❌'} {right_reason}")

            if left_safe and right_safe:
                # Prefer left only when in middle lane (lane 2)
                if current_lane == self.center_lane:
                    if self.verbose:
                        print("  ⬅️ Both safe at center, preferring LEFT")
                    new_lane = current_lane - 1
                    steer_duration = self.get_steer_duration(current_lane, new_lane)
                    self.update_lane(new_lane, "LANE_CHANGE_LEFT")
                    return (["STEER_LEFT"] * steer_duration) + (
                        ["STEER_RIGHT"] * steer_duration
                    )
                # Otherwise choose direction toward center
                elif current_lane < self.center_lane:
                    if self.verbose:
                        print("  ➡️ Both safe, moving RIGHT toward center")
                    new_lane = current_lane + 1
                    steer_duration = self.get_steer_duration(current_lane, new_lane)
                    self.update_lane(new_lane, "LANE_CHANGE_RIGHT")
                    return (["STEER_RIGHT"] * steer_duration) + (
                        ["STEER_LEFT"] * steer_duration
                    )
                else:
                    if self.verbose:
                        print("  ⬅️ Both safe, moving LEFT toward center")
                    new_lane = current_lane - 1
                    steer_duration = self.get_steer_duration(current_lane, new_lane)
                    self.update_lane(new_lane, "LANE_CHANGE_LEFT")
                    return (["STEER_LEFT"] * steer_duration) + (
                        ["STEER_RIGHT"] * steer_duration
                    )
            elif left_safe:
                new_lane = current_lane - 1
                steer_duration = self.get_steer_duration(current_lane, new_lane)
                self.update_lane(new_lane, "LANE_CHANGE_LEFT")
                return (["STEER_LEFT"] * steer_duration) + (
                    ["STEER_RIGHT"] * steer_duration
                )
            elif right_safe:
                new_lane = current_lane + 1
                steer_duration = self.get_steer_duration(current_lane, new_lane)
                self.update_lane(new_lane, "LANE_CHANGE_RIGHT")
                return (["STEER_RIGHT"] * steer_duration) + (
                    ["STEER_LEFT"] * steer_duration
                )
            else:
                # Can't change lanes, accelerate
                return ["ACCELERATE"]

        # No immediate threats - maintain speed or move to center if safe
        else:
            # Check if we should move toward center lane
            if current_lane != self.center_lane:
                target_direction = (
                    "right" if current_lane < self.center_lane else "left"
                )
                is_safe, reason = self.check_lane_safety(sensors, target_direction)

                if is_safe:
                    if self.verbose:
                        print(
                            f"  🎯 Starting progressive centering {'RIGHT' if target_direction == 'right' else 'LEFT'} toward center lane"
                        )

                    # Start progressive centering instead of immediate full steering
                    if target_direction == "right":
                        target_lane = current_lane + 1
                        action = self.start_centering(current_lane, target_lane)
                        return [action]
                    else:
                        target_lane = current_lane - 1
                        action = self.start_centering(current_lane, target_lane)
                        return [action]

            # Maintain speed
            current_speed = abs(velocity.get("x", 10))
            # if current_speed < 70:
            return ["ACCELERATE"]
            # else:
            # return ["NOTHING"]

    def close(self):
        """Close database connection."""
        self.conn.close()


# Example usage
if __name__ == "__main__":
    # Initialize controller with in-memory database
    controller = LaneChangeController(verbose=True)

    # Example sensor data with proper sensor names
    test_data = {
        "sensors": {
            "front": 150,  # Front obstacle
            "front_right_front": 500,
            "right_front": 600,
            "right_side_front": 400,
            "right_side": 350,  # Too close! (< 345)
            "right_side_back": 400,
            "right_back": 500,
            "back_right_back": 1000,
            "front_left_front": 1000,
            "left_front": 500,
            "left_side_front": 400,
            "left_side": 400,  # Safe (> 345)
            "left_side_back": 400,
            "left_back": 500,
            "back_left_back": 1000,
            "back": 999,
        },
        "velocity": {"x": 30, "y": 0},
        "did_crash": False,
    }

    # Get actions
    actions = controller.predict_actions(test_data)
    print(f"\n🎮 Actions: {actions}")
    print(f"📍 Final lane: {controller.get_current_lane()}")

    # Don't forget to close when done
    controller.close()
