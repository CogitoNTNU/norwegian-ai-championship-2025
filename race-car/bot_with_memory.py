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
        self.safe_distances = [999.0, 474.0, 360.0, 345.0, 360.0, 474.0, 999.0]
        
        # Sensor names for right side (left side mirrors these)
        self.right_sensors = [
            "front_right_front",
            "right_front", 
            "right_side_front",
            "right_side",
            "right_side_back", 
            "right_back",
            "back_right_back"
        ]
        
        self.left_sensors = [
            "front_left_front",
            "left_front",
            "left_side_front", 
            "left_side",
            "left_side_back",
            "left_back",
            "back_left_back"
        ]
        
        # Lane boundaries (0 = leftmost, 4 = rightmost)
        self.min_lane = 0
        self.max_lane = 4
        self.center_lane = 2
        
        # Initialize database
        self._init_database()
        
    def _init_database(self):
        """Create necessary tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Table for tracking current lane and state
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vehicle_state (
                id INTEGER PRIMARY KEY,
                current_lane INTEGER NOT NULL DEFAULT 2,
                last_action TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Table for sensor history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sensor_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sensors_json TEXT NOT NULL,
                velocity_json TEXT NOT NULL,
                action_taken TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Initialize with default state if empty
        cursor.execute("SELECT COUNT(*) FROM vehicle_state")
        if cursor.fetchone()[0] == 0:
            cursor.execute(
                "INSERT INTO vehicle_state (current_lane) VALUES (?)", 
                (self.center_lane,)
            )
        
        self.conn.commit()
    
    def get_current_lane(self) -> int:
        """Get the current lane from database."""
        with self.db_lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT current_lane FROM vehicle_state ORDER BY id DESC LIMIT 1")
            result = cursor.fetchone()
            return result[0] if result else self.center_lane
    
    def update_lane(self, new_lane: int, action: str):
        """Update the current lane in database."""
        with self.db_lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE vehicle_state SET current_lane = ?, last_action = ?, timestamp = ? WHERE id = 1",
                (new_lane, action, datetime.now())
            )
            self.conn.commit()
    
    def save_sensor_reading(self, sensors: Dict, velocity: Dict, action: str):
        """Save sensor reading to history."""
        with self.db_lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO sensor_history (sensors_json, velocity_json, action_taken) VALUES (?, ?, ?)",
                (json.dumps(sensors), json.dumps(velocity), action)
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
            
            if sensor_value < safe_distance:
                if self.verbose:
                    print(f"  ⚠️ {sensor_name}: {sensor_value:.1f} < {safe_distance:.1f} (unsafe)")
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
            return ["NOTHING"]
        
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
        if front_sensor < 999:  # Obstacle ahead
            if self.verbose:
                print(f"🚦 Obstacle ahead at {front_sensor:.1f}")
            
            # Check both directions for safety
            left_safe, left_reason = self.check_lane_safety(sensors, "left")
            right_safe, right_reason = self.check_lane_safety(sensors, "right")
            
            if self.verbose:
                print(f"  Left lane: {'✅' if left_safe else '❌'} {left_reason}")
                print(f"  Right lane: {'✅' if right_safe else '❌'} {right_reason}")
            
            # Decision logic
            if not left_safe and not right_safe:
                # Both blocked - brake (single DECELERATE per tick)
                if self.verbose:
                    print("  🛑 Both lanes blocked - BRAKING!")
                return ["DECELERATE"]
            
            elif left_safe and right_safe:
                # Both safe - prefer moving toward center
                if current_lane < self.center_lane:
                    # We're left of center, go right
                    if self.verbose:
                        print("  ➡️ Both safe, moving RIGHT toward center")
                    self.update_lane(current_lane + 1, "LANE_CHANGE_RIGHT")
                    # Return steering sequence: 48 right, then 48 left to straighten
                    return (["STEER_RIGHT"] * 48) + (["STEER_LEFT"] * 48)
                elif current_lane > self.center_lane:
                    # We're right of center, go left
                    if self.verbose:
                        print("  ⬅️ Both safe, moving LEFT toward center")
                    self.update_lane(current_lane - 1, "LANE_CHANGE_LEFT")
                    # Return steering sequence: 48 left, then 48 right to straighten
                    return (["STEER_LEFT"] * 48) + (["STEER_RIGHT"] * 48)
                else:
                    # At center, choose based on which side has more clearance
                    left_clearance = sum((sensors.get(s) if sensors.get(s) is not None else 1000.0) for s in self.left_sensors)
                    right_clearance = sum((sensors.get(s) if sensors.get(s) is not None else 1000.0) for s in self.right_sensors)
                    
                    if left_clearance > right_clearance:
                        if self.verbose:
                            print(f"  ⬅️ Left side clearer ({left_clearance:.1f} > {right_clearance:.1f})")
                        self.update_lane(current_lane - 1, "LANE_CHANGE_LEFT")
                        return (["STEER_LEFT"] * 48) + (["STEER_RIGHT"] * 48)
                    else:
                        if self.verbose:
                            print(f"  ➡️ Right side clearer ({right_clearance:.1f} >= {left_clearance:.1f})")
                        self.update_lane(current_lane + 1, "LANE_CHANGE_RIGHT")
                        return (["STEER_RIGHT"] * 48) + (["STEER_LEFT"] * 48)
            
            elif left_safe:
                # Only left is safe
                if self.verbose:
                    print("  ⬅️ Only left safe - changing LEFT")
                self.update_lane(current_lane - 1, "LANE_CHANGE_LEFT")
                return (["STEER_LEFT"] * 48) + (["STEER_RIGHT"] * 48)
            
            else:  # only right_safe
                # Only right is safe
                if self.verbose:
                    print("  ➡️ Only right safe - changing RIGHT")
                self.update_lane(current_lane + 1, "LANE_CHANGE_RIGHT")
                return (["STEER_RIGHT"] * 48) + (["STEER_LEFT"] * 48)
        
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
                # Choose direction toward center
                if current_lane <= self.center_lane and right_safe:
                    self.update_lane(current_lane + 1, "LANE_CHANGE_RIGHT")
                    return (["STEER_RIGHT"] * 48) + (["STEER_LEFT"] * 48)
                elif left_safe:
                    self.update_lane(current_lane - 1, "LANE_CHANGE_LEFT")
                    return (["STEER_LEFT"] * 48) + (["STEER_RIGHT"] * 48)
            elif left_safe:
                self.update_lane(current_lane - 1, "LANE_CHANGE_LEFT")
                return (["STEER_LEFT"] * 48) + (["STEER_RIGHT"] * 48)
            elif right_safe:
                self.update_lane(current_lane + 1, "LANE_CHANGE_RIGHT")
                return (["STEER_RIGHT"] * 48) + (["STEER_LEFT"] * 48)
            else:
                # Can't change lanes, accelerate
                return ["ACCELERATE"] 
        
        # No immediate threats - maintain speed or move to center if safe
        else:
            # Check if we should move toward center lane
            if current_lane != self.center_lane:
                target_direction = "right" if current_lane < self.center_lane else "left"
                is_safe, reason = self.check_lane_safety(sensors, target_direction)
                
                if is_safe:
                    if self.verbose:
                        print(f"  🎯 Moving {'RIGHT' if target_direction == 'right' else 'LEFT'} toward center lane")
                    
                    if target_direction == "right":
                        self.update_lane(current_lane + 1, "LANE_CHANGE_RIGHT")
                        return (["STEER_RIGHT"] * 48) + (["STEER_LEFT"] * 48)
                    else:
                        self.update_lane(current_lane - 1, "LANE_CHANGE_LEFT")
                        return (["STEER_LEFT"] * 48) + (["STEER_RIGHT"] * 48)
            
            # Maintain speed
            current_speed = abs(velocity.get("x", 10))
            if current_speed < 70:
                return ["ACCELERATE"]
            else:
                return ["NOTHING"]
    
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
            "back": 999
        },
        "velocity": {"x": 30, "y": 0},
        "did_crash": False
    }
    
    # Get actions
    actions = controller.predict_actions(test_data)
    print(f"\n🎮 Actions: {actions}")
    print(f"📍 Final lane: {controller.get_current_lane()}")
    
    # Don't forget to close when done
    controller.close()
