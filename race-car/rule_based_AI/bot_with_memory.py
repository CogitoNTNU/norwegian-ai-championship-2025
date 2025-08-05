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

        # Speed matching parameters
        self.speed_match_threshold = 8.0  # Consider speeds matched if distance change is within this
        self.min_safe_distance = 250  # Minimum safe following distance
        self.speed_match_attempts = 3  # Number of attempts to match speed before lane change

        # Initialize database
        self._init_database()

    def get_steer_duration(self, from_lane: int, to_lane: int) -> int:
        """Get steering duration based on lane change."""
        # 44 for lane changes between 1 and 2, 48 otherwise
        if (from_lane == 1 and to_lane == 2) or (from_lane == 2 and to_lane == 1):
            return 44
        else:
            return self.steer_duration

    def _init_database(self):
        """Create necessary tables if they don't exist."""
        cursor = self.conn.cursor()

        # Table for tracking current lane and state
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vehicle_state (
                id INTEGER PRIMARY KEY,
                current_lane INTEGER NOT NULL DEFAULT 2,
                last_action TEXT,
                speed_match_attempts INTEGER DEFAULT 0,
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
                front_distance REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Initialize with default state if empty
        cursor.execute("SELECT COUNT(*) FROM vehicle_state")
        if cursor.fetchone()[0] == 0:
            cursor.execute(
                "INSERT INTO vehicle_state (current_lane, speed_match_attempts) VALUES (?, ?)",
                (self.center_lane, 0),
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

    def get_speed_match_attempts(self) -> int:
        """Get the current speed match attempts count."""
        with self.db_lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT speed_match_attempts FROM vehicle_state ORDER BY id DESC LIMIT 1"
            )
            result = cursor.fetchone()
            return result[0] if result else 0

    def update_lane(self, new_lane: int, action: str):
        """Update the current lane in database and reset speed match attempts."""
        with self.db_lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE vehicle_state SET current_lane = ?, last_action = ?, speed_match_attempts = 0, timestamp = ? WHERE id = 1",
                (new_lane, action, datetime.now()),
            )
            self.conn.commit()

    def increment_speed_match_attempts(self):
        """Increment the speed match attempts counter."""
        with self.db_lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE vehicle_state SET speed_match_attempts = speed_match_attempts + 1, timestamp = ? WHERE id = 1",
                (datetime.now(),)
            )
            self.conn.commit()

    def reset_speed_match_attempts(self):
        """Reset the speed match attempts counter."""
        with self.db_lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE vehicle_state SET speed_match_attempts = 0, timestamp = ? WHERE id = 1",
                (datetime.now(),)
            )
            self.conn.commit()

    def save_sensor_reading(self, sensors: Dict, velocity: Dict, action: str):
        """Save sensor reading to history."""
        front_distance = self.get_front_sensor(sensors)
        with self.db_lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO sensor_history (sensors_json, velocity_json, action_taken, front_distance) VALUES (?, ?, ?, ?)",
                (json.dumps(sensors), json.dumps(velocity), action, front_distance),
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

    def get_recent_front_distances(self, count: int = 3) -> List[float]:
        """Get the last N front sensor distances from database."""
        with self.db_lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT front_distance FROM sensor_history 
                ORDER BY timestamp DESC LIMIT ?
            """, (count,))
            results = cursor.fetchall()
            return [row[0] for row in results] if results else []

    def is_speed_matched(self, current_front_distance: float) -> Tuple[bool, float, str]:
        """
        Check if our speed is matched with the car in front.
        
        Returns:
            Tuple of (is_matched, distance_change, recommended_action)
        """
        recent_distances = self.get_recent_front_distances(2)
        
        if len(recent_distances) < 2:
            return False, 0.0, "NOTHING"  # Not enough data
        
        # Calculate distance change (positive = gap increasing, negative = gap decreasing)
        distance_change = current_front_distance - recent_distances[1]
        
        if self.verbose:
            print(f"  📊 Distance change: {distance_change:.1f} (current: {current_front_distance:.1f}, previous: {recent_distances[1]:.1f})")
        
        # Check if speeds are matched
        if abs(distance_change) <= self.speed_match_threshold:
            return True, distance_change, "NOTHING"
        elif distance_change > self.speed_match_threshold:
            # Gap is increasing - car in front is faster or moving away
            return False, distance_change, "ACCELERATE"
        else:
            # Gap is decreasing - car in front is slower
            return False, distance_change, "DECELERATE"

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

    def should_attempt_lane_change(self, front_distance: float) -> bool:
        """
        Determine if we should attempt a lane change based on current conditions.
        
        Args:
            front_distance: Distance to car in front
            
        Returns:
            True if lane change should be attempted
        """
        attempts = self.get_speed_match_attempts()
        
        # Always attempt lane change if:
        # 1. Distance is critically low (emergency)
        # 2. We've tried speed matching enough times
        # 3. Distance is very far (no need to match speed)
        
        if front_distance < self.min_safe_distance:
            if self.verbose:
                print(f"  🚨 Emergency lane change needed - distance too low: {front_distance:.1f}")
            return True
        
        if front_distance > 950:
            if self.verbose:
                print(f"  🚗 No car ahead - no need for speed matching")
            return True
            
        if attempts >= self.speed_match_attempts:
            if self.verbose:
                print(f"  ⏰ Speed matching attempts exhausted ({attempts}/{self.speed_match_attempts}) - attempting lane change")
            return True
            
        return False

    def predict_actions(self, request_data: dict) -> List[str]:
        """
        Predict actions based on sensor data with speed matching priority.

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
            return ["NOTHING"]

        current_lane = self.get_current_lane()

        # Get key sensors
        front_sensor = self.get_front_sensor(sensors)
        back_sensor = self.get_back_sensor(sensors)

        if self.verbose:
            print(f"\n🚗 Current lane: {current_lane}")
            print(f"📡 Front: {front_sensor:.1f}, Back: {back_sensor:.1f}")
            print(f"🔄 Speed match attempts: {self.get_speed_match_attempts()}/{self.speed_match_attempts}")

        # Check for front obstacle - PRIORITY: Speed matching first
        if front_sensor < 950:  # Obstacle ahead
            if self.verbose:
                print(f"🚦 Obstacle ahead at {front_sensor:.1f}")

            # Check if we should attempt lane change or try speed matching first
            should_change_lanes = self.should_attempt_lane_change(front_sensor)
            
            if not should_change_lanes:
                # Try to match speed with car in front
                is_matched, distance_change, speed_action = self.is_speed_matched(front_sensor)
                
                if is_matched:
                    if self.verbose:
                        print("  ✅ Speed matched with front car")
                    self.reset_speed_match_attempts()
                    return ["NOTHING"]
                else:
                    if self.verbose:
                        print(f"  🎯 Attempting to match speed: {speed_action}")
                    self.increment_speed_match_attempts()
                    return [speed_action]
            
            # If we reach here, we should attempt lane changes
            if self.verbose:
                print("  🔄 Attempting lane change...")
            
            # Reset speed match attempts when starting lane change attempt
            self.reset_speed_match_attempts()

            # Check both directions for safety
            left_safe, left_reason = self.check_lane_safety(sensors, "left")
            right_safe, right_reason = self.check_lane_safety(sensors, "right")

            if self.verbose:
                print(f"  Left lane: {'✅' if left_safe else '❌'} {left_reason}")
                print(f"  Right lane: {'✅' if right_safe else '❌'} {right_reason}")

            # Decision logic
            if not left_safe and not right_safe:
                # Both blocked - continue trying to match speed of car in front
                is_matched, distance_change, speed_action = self.is_speed_matched(front_sensor)
                
                if self.verbose:
                    print(f"  🚗 Both lanes blocked - continuing speed matching")
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
                    self.update_lane(current_lane - 1, "LANE_CHANGE_LEFT")
                    return (["STEER_LEFT"] * self.steer_duration) + (
                        ["STEER_RIGHT"] * self.steer_duration
                    )

            elif left_safe:
                # Only left is safe
                if self.verbose:
                    print("  ⬅️ Only left safe - changing LEFT")
                self.update_lane(current_lane - 1, "LANE_CHANGE_LEFT")
                return (["STEER_LEFT"] * self.steer_duration) + (
                    ["STEER_RIGHT"] * self.steer_duration
                )

            else:  # only right_safe
                # Only right is safe
                if self.verbose:
                    print("  ➡️ Only right safe - changing RIGHT")
                self.update_lane(current_lane + 1, "LANE_CHANGE_RIGHT")
                return (["STEER_RIGHT"] * self.steer_duration) + (
                    ["STEER_LEFT"] * self.steer_duration
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
                    self.update_lane(current_lane - 1, "LANE_CHANGE_LEFT")
                    return (["STEER_LEFT"] * self.steer_duration) + (
                        ["STEER_RIGHT"] * self.steer_duration
                    )
                # Otherwise choose direction toward center
                elif current_lane < self.center_lane:
                    if self.verbose:
                        print("  ➡️ Both safe, moving RIGHT toward center")
                    self.update_lane(current_lane + 1, "LANE_CHANGE_RIGHT")
                    return (["STEER_RIGHT"] * self.steer_duration) + (
                        ["STEER_LEFT"] * self.steer_duration
                    )
                else:
                    if self.verbose:
                        print("  ⬅️ Both safe, moving LEFT toward center")
                    self.update_lane(current_lane - 1, "LANE_CHANGE_LEFT")
                    return (["STEER_LEFT"] * self.steer_duration) + (
                        ["STEER_RIGHT"] * self.steer_duration
                    )
            elif left_safe:
                self.update_lane(current_lane - 1, "LANE_CHANGE_LEFT")
                return (["STEER_LEFT"] * self.steer_duration) + (
                    ["STEER_RIGHT"] * self.steer_duration
                )
            elif right_safe:
                self.update_lane(current_lane + 1, "LANE_CHANGE_RIGHT")
                return (["STEER_RIGHT"] * self.steer_duration) + (
                    ["STEER_LEFT"] * self.steer_duration
                )
            else:
                # Can't change lanes, accelerate
                return ["ACCELERATE"]

        # No immediate threats - maintain speed or move to center if safe
        else:
            # Reset speed match attempts when no obstacles
            self.reset_speed_match_attempts()
            
            # Check if we should move toward center lane
            if current_lane != self.center_lane:
                target_direction = (
                    "right" if current_lane < self.center_lane else "left"
                )
                is_safe, reason = self.check_lane_safety(sensors, target_direction)

                if is_safe:
                    if self.verbose:
                        print(
                            f"  🎯 Moving {'RIGHT' if target_direction == 'right' else 'LEFT'} toward center lane"
                        )

                    if target_direction == "right":
                        self.update_lane(current_lane + 1, "LANE_CHANGE_RIGHT")
                        return (["STEER_RIGHT"] * self.steer_duration) + (
                            ["STEER_LEFT"] * self.steer_duration
                        )
                    else:
                        self.update_lane(current_lane - 1, "LANE_CHANGE_LEFT")
                        return (["STEER_LEFT"] * self.steer_duration) + (
                            ["STEER_RIGHT"] * self.steer_duration
                        )

            # Maintain speed
            current_speed = abs(velocity.get("x", 10))
            return ["ACCELERATE"]

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
