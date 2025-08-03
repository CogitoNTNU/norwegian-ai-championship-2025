#!/usr/bin/env python3
"""
Validation script for Race Car Control.
Submits the API endpoint to the Norwegian AI Championship leaderboard.
"""

import os
import sys
import requests
from typing import Optional
import argparse
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_token() -> str:
    """Get the evaluation API token from environment."""
    token = os.getenv("EVAL_API_TOKEN")
    if not token:
        print("❌ Error: EVAL_API_TOKEN not found in .env file")
        print("Please set your competition token in the .env file")
        sys.exit(1)
    return token


def get_service_url() -> str:
    """Get the service URL from environment or use default."""
    return os.getenv("RACE_CAR_SERVICE_URL", "http://0.0.0.0:9052")


def submit_validation(token: str, service_url: str) -> Optional[str]:
    """Submit validation request to competition system."""
    api_url = "https://cases.ainm.no/api/v1/usecases/race-car/validate/queue"

    headers = {"x-token": token, "Content-Type": "application/json"}

    payload = {"url": f"{service_url}/predict"}

    print("🚀 Submitting validation request...")
    print(f"   Service URL: {payload['url']}")
    print(f"   Competition API: {api_url}")

    try:
        response = requests.post(api_url, headers=headers, json=payload)

        if response.status_code == 200:
            try:
                result = response.json()
                print("✅ Validation submitted successfully!")
                print(f"   Raw response: {result}")

                # Try different possible field names for the validation ID
                validation_id = (
                    result.get("queued_attempt_uuid")
                    or result.get("id")
                    or result.get("validation_id")
                    or result.get("uuid")
                    or result.get("task_id")
                )

                if validation_id:
                    print(f"   Validation ID: {validation_id}")
                else:
                    print("   ⚠️  No validation ID found in response")

                return validation_id
            except ValueError:
                # Response is not JSON
                print("✅ Validation submitted successfully!")
                print(f"   Raw response (non-JSON): {response.text}")
                return None
        else:
            print(f"❌ Validation failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        return None


def check_status(token: str, validation_id: str) -> None:
    """Check the status of a validation request."""
    api_url = f"https://cases.ainm.no/api/v1/validation/{validation_id}"

    headers = {"x-token": token}

    try:
        response = requests.get(api_url, headers=headers)

        if response.status_code == 200:
            result = response.json()
            status = result.get("status", "unknown")
            print(f"📊 Validation status: {status}")

            if "score" in result:
                print(f"🎯 Score: {result['score']}")

            if "message" in result:
                print(f"💬 Message: {result['message']}")

        else:
            print(f"❌ Could not check status: {response.status_code}")
            print(f"   Response: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")


def wait_for_completion(token: str, validation_id: str, timeout: int = 300) -> None:
    """Wait for validation to complete."""
    print(f"⏳ Waiting for validation to complete (timeout: {timeout}s)...")

    start_time = time.time()
    while time.time() - start_time < timeout:
        api_url = f"https://cases.ainm.no/api/v1/validation/{validation_id}"
        headers = {"x-token": token}

        try:
            response = requests.get(api_url, headers=headers)
            if response.status_code == 200:
                result = response.json()
                status = result.get("status", "unknown")

                if status in ["completed", "failed", "error"]:
                    print(f"\n🏁 Validation completed with status: {status}")
                    if "score" in result:
                        print(f"🎯 Final Score: {result['score']}")
                    if "message" in result:
                        print(f"💬 Message: {result['message']}")
                    return

                print(f"\r⏳ Status: {status}", end="", flush=True)

        except requests.exceptions.RequestException:
            pass

        time.sleep(5)

    print(
        f"\n⏰ Timeout reached. Check status manually with: uv run check-status {validation_id}"
    )


def main():
    parser = argparse.ArgumentParser(description="Validate Race Car Control solution")
    parser.add_argument(
        "--wait", action="store_true", help="Wait for validation to complete"
    )
    parser.add_argument(
        "--timeout", type=int, default=300, help="Timeout in seconds when waiting"
    )

    args = parser.parse_args()

    print("🏎️ Race Car Control - Competition Validation")
    print("========================================")

    token = get_token()
    service_url = get_service_url()

    # Test if service is running
    try:
        test_response = requests.get(f"{service_url}/api", timeout=5)
        if test_response.status_code == 200:
            print(f"✅ Service is running at {service_url}")
        else:
            print(f"⚠️  Service responded with status {test_response.status_code}")
    except requests.exceptions.RequestException:
        print(f"⚠️  Warning: Could not connect to service at {service_url}")
        print("   Make sure your API server is running!")

    validation_id = submit_validation(token, service_url)

    if validation_id and args.wait:
        wait_for_completion(token, validation_id, args.timeout)


def check_status_cli():
    """CLI command for checking validation status."""
    if len(sys.argv) < 2:
        print("Usage: uv run check-status <validation_id>")
        sys.exit(1)

    validation_id = sys.argv[1]
    token = get_token()
    check_status(token, validation_id)


if __name__ == "__main__":
    main()
