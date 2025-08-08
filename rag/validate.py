#!/usr/bin/env python3
"""
Validation script for Emergency Healthcare RAG.
Submits the API endpoint to the Norwegian AI Championship leaderboard.
"""

import os
import sys
import requests
from typing import Optional
import time
from dotenv import load_dotenv
from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd

from get_config import config

# Load environment variables
load_dotenv()


def get_token() -> str:
    """Get the evaluation API token from environment."""
    token = os.getenv("EVAL_API_TOKEN")
    if not token:
        logger.error("EVAL_API_TOKEN not found in .env file")
        logger.error("Please set your competition token in the .env file")
        sys.exit(1)
    return token


def get_service_url() -> str:
    """Get the service URL from environment or use default."""
    return os.getenv("EMERGENCY_HEALTHCARE_SERVICE_URL", "http://0.0.0.0:8000")


def submit_validation() -> Optional[str]:
    """Submit validation request to competition system."""
    api_url = (
        "https://cases.ainm.no/api/v1/usecases/emergency-healthcare-rag/validate/queue"
    )
    
    token = get_token()
    service_url = get_service_url()
    headers = {"x-token": token, "Content-Type": "application/json"}

    payload = {"url": f"{service_url}/predict"}

    logger.info("Submitting validation request...")
    logger.info(f"   Service URL: {payload['url']}")
    logger.info(f"   Competition API: {api_url}")

    try:
        response = requests.post(api_url, headers=headers, json=payload)

        if response.status_code == 200:
            try:
                result = response.json()
                logger.success("Validation submitted successfully!")
                logger.info(f"   Raw response: {result}")

                # Try different possible field names for the validation ID
                validation_id = (
                    result.get("queued_attempt_uuid")
                    or result.get("id")
                    or result.get("validation_id")
                    or result.get("uuid")
                    or result.get("task_id")
                )

                if validation_id:
                    logger.info(f"   Validation ID: {validation_id}")
                else:
                    logger.warning("No validation ID found in response")

                return validation_id
            except ValueError:
                # Response is not JSON
                logger.success("Validation submitted successfully!")
                logger.info(f"   Raw response (non-JSON): {response.text}")
                return None
        else:
            logger.error(f"Validation failed with status {response.status_code}")
            logger.error(f"   Response: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error: {e}")
        return None


def check_status(validation_id: str) -> None:
    """Check the status of a validation request."""
    api_url = f"https://cases.ainm.no/api/v1/validation/{validation_id}"
    
    token = get_token()
    headers = {"x-token": token}

    try:
        response = requests.get(api_url, headers=headers)

        if response.status_code == 200:
            result = response.json()
            status = result.get("status", "unknown")
            logger.info(f"Validation status: {status}")

            if "score" in result:
                logger.success(f"Score: {result['score']}")
                # Plot score if available
                plot_validation_score(result['score'])

            if "message" in result:
                logger.info(f"Message: {result['message']}")

        else:
            logger.error(f"Could not check status: {response.status_code}")
            logger.error(f"   Response: {response.text}")

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error: {e}")


def wait_for_completion(validation_id: str) -> None:
    """Wait for validation to complete."""
    timeout = config.validation_timeout
    logger.info(f"Waiting for validation to complete (timeout: {timeout}s)...")
    
    token = get_token()

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
                    logger.info(f"\nValidation completed with status: {status}")
                    if "score" in result:
                        logger.success(f"Final Score: {result['score']}")
                        plot_validation_score(result['score'])
                    if "message" in result:
                        logger.info(f"Message: {result['message']}")
                    return

                logger.info(f"\rStatus: {status}")

        except requests.exceptions.RequestException:
            pass

        time.sleep(5)

    logger.warning(f"\nTimeout reached. Check status manually with: uv run check-status {validation_id}")


def plot_validation_score(score: float) -> None:
    """Plot validation score."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create a simple bar chart showing the score
    categories = ['Validation Score']
    scores = [score]
    colors = ['green' if score >= 0.8 else 'orange' if score >= 0.6 else 'red']
    
    bars = ax.bar(categories, scores, color=colors, alpha=0.7)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Score')
    ax.set_title('Model Validation Score')
    
    # Add score text on bar
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('validation_score.png', dpi=300, bbox_inches='tight')
    logger.info("Validation score plot saved as 'validation_score.png'")
    plt.show()


def main():
    """Main function using config parameters."""
    logger.info("Emergency Healthcare RAG - Competition Validation")
    logger.info("====================================================")

    service_url = get_service_url()

    # Test if service is running
    try:
        test_response = requests.get(f"{service_url}/api", timeout=5)
        if test_response.status_code == 200:
            logger.success(f"Service is running at {service_url}")
        else:
            logger.warning(f"Service responded with status {test_response.status_code}")
    except requests.exceptions.RequestException:
        logger.warning(f"Warning: Could not connect to service at {service_url}")
        logger.warning("   Make sure your API server is running!")

    validation_id = submit_validation()

    if validation_id and config.wait_for_validation:
        wait_for_completion(validation_id)


def check_status_cli():
    """CLI command for checking validation status."""
    if len(sys.argv) < 2:
        logger.error("Usage: uv run check-status <validation_id>")
        sys.exit(1)

    validation_id = sys.argv[1]
    check_status(validation_id)


if __name__ == "__main__":
    main()
