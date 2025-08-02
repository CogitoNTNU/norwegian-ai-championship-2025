"""
Shared utilities for leaderboard validation scripts.
"""

import os
import time
import requests
from pathlib import Path
from typing import Dict, Optional, Tuple
from dotenv import load_dotenv


def load_env() -> Dict[str, str]:
    """Load environment variables from the root .env file."""
    # Find the root directory (where .env is located)
    current_dir = Path(__file__).parent
    root_dir = current_dir

    # Traverse up to find .env file
    while root_dir.parent != root_dir:
        env_file = root_dir / ".env"
        if env_file.exists():
            break
        root_dir = root_dir.parent

    # Load environment variables
    env_file = root_dir / ".env"
    if env_file.exists():
        load_dotenv(env_file)

    return {
        "EVAL_API_TOKEN": os.getenv("EVAL_API_TOKEN", ""),
        "EMERGENCY_HEALTHCARE_SERVICE_URL": os.getenv(
            "EMERGENCY_HEALTHCARE_SERVICE_URL", "http://0.0.0.0:8000"
        ),
        "TUMOR_SEGMENTATION_SERVICE_URL": os.getenv(
            "TUMOR_SEGMENTATION_SERVICE_URL", "http://0.0.0.0:9051"
        ),
        "RACE_CAR_SERVICE_URL": os.getenv(
            "RACE_CAR_SERVICE_URL", "http://0.0.0.0:9052"
        ),
    }


def make_request(
    method: str, url: str, headers: Dict[str, str], data: Optional[Dict] = None
) -> Tuple[bool, Dict]:
    """
    Make HTTP request with error handling.

    Args:
        method: HTTP method (GET, POST)
        url: Request URL
        headers: Request headers
        data: Request data (for POST)

    Returns:
        Tuple of (success: bool, response_data: Dict)
    """
    try:
        if method.upper() == "POST":
            response = requests.post(url, headers=headers, json=data, timeout=30)
        else:
            response = requests.get(url, headers=headers, timeout=30)

        response.raise_for_status()
        return True, response.json()

    except requests.exceptions.RequestException as e:
        return False, {"error": str(e)}


def print_status(message: str, status: str = "info"):
    """Print colored status messages."""
    colors = {
        "success": "\033[92m",  # Green
        "error": "\033[91m",  # Red
        "warning": "\033[93m",  # Yellow
        "info": "\033[94m",  # Blue
        "reset": "\033[0m",  # Reset
    }

    color = colors.get(status, colors["info"])
    print(f"{color}{message}{colors['reset']}")


def wait_for_completion(
    task: str, uuid: str, token: str, max_wait: int = 300, poll_interval: int = 5
) -> Dict:
    """
    Wait for validation to complete by polling status.

    Args:
        task: Task name (e.g., 'emergency-healthcare-rag')
        uuid: Validation UUID
        token: API token
        max_wait: Maximum wait time in seconds
        poll_interval: Polling interval in seconds

    Returns:
        Final status response
    """
    url = f"https://cases.ainm.no/api/v1/usecases/{task}/validate/queue/{uuid}"
    headers = {"x-token": token}

    start_time = time.time()

    print_status(f"⏳ Waiting for validation to complete (UUID: {uuid})", "info")

    while time.time() - start_time < max_wait:
        success, response = make_request("GET", url, headers)

        if not success:
            print_status(
                f"❌ Error checking status: {response.get('error', 'Unknown error')}",
                "error",
            )
            return response

        status = response.get("status", "unknown")

        if status == "done":
            print_status("✅ Validation completed successfully!", "success")
            return response
        elif status == "failed":
            print_status("❌ Validation failed!", "error")
            return response
        elif status == "queued" or status == "running":
            position = response.get("position_in_queue", 0)
            if position > 0:
                print_status(f"⏳ Position in queue: {position}", "info")
            else:
                print_status("🔄 Validation in progress...", "info")

        time.sleep(poll_interval)

    print_status(f"⏰ Timeout after {max_wait} seconds", "warning")
    return {
        "status": "timeout",
        "message": f"Validation did not complete within {max_wait} seconds",
    }


def validate_env_vars(env_vars: Dict[str, str]) -> bool:
    """Validate that required environment variables are set."""
    required_vars = ["EVAL_API_TOKEN"]
    missing_vars = [var for var in required_vars if not env_vars.get(var)]

    if missing_vars:
        print_status(
            f"❌ Missing required environment variables: {', '.join(missing_vars)}",
            "error",
        )
        print_status("💡 Please check your .env file in the project root", "info")
        return False

    return True
