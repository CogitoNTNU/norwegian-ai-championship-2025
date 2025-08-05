# front_right_front, right_front, right_side_front
safe_distances = [999.0, 474., 360., 345., 360., 474., 999.0]


def predict_actions_from_game_bot(request_data: dict) -> list[str]:
    """
    Use the game bot logic from core.py to predict actions based on sensor data.
    Returns a list of actions based on the current game state.
    """
    # Extract sensor data
    sensors = request_data.get("sensors", {})
    velocity = request_data.get("velocity", {"x": 10, "y": 0})

    # Get front sensor reading (assuming sensor_0 or 'front' is the front sensor)
    front_sensor = sensors.get("sensor_0") or sensors.get("front") or sensors.get("0")
    back_sensor = sensors.get("sensor_4") or sensors.get("back") or sensors.get("4")

    # Simulate the core.py bot logic
    actions = []

    # Check if there's an obstacle in front
    if front_sensor is not None and front_sensor < 999:  # Obstacle detected in front
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
            actions.extend(["STEER_LEFT"] * 47)  # Second phase to straighten
        elif right_blocked:
            # Right blocked, go left if possible
            actions.extend(["STEER_LEFT"] * 48)  # First phase
            actions.extend(["STEER_RIGHT"] * 47)  # Second phase to straighten
        elif left_sum > right_sum:
            # Left is clearer, go left
            actions.extend(["STEER_LEFT"] * 48)
            actions.extend(["STEER_RIGHT"] * 47)
        else:
            # Right is clearer or equal, go right
            actions.extend(["STEER_RIGHT"] * 48)
            actions.extend(["STEER_LEFT"] * 47)

    # Check back sensor for approaching cars
    elif back_sensor is not None and back_sensor < 800:  # Car approaching from behind
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
            actions.extend(["STEER_RIGHT"] * 47)
        else:
            actions.extend(["STEER_RIGHT"] * 48)
            actions.extend(["STEER_LEFT"] * 47)

    # Default behavior - accelerate if no immediate threats
    if not actions:
        current_speed = abs(velocity.get("x", 10))
        if current_speed < 23:  # Target speed
            actions.append("ACCELERATE")
        else:
            actions.append("NOTHING")

    return actions
