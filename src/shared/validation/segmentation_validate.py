"""
Validation script for the Tumor Segmentation leaderboard.
"""

import sys
from .utils import (
    load_env,
    make_request,
    print_status,
    wait_for_completion,
    validate_env_vars,
)


def submit_validation() -> None:
    """Submit the Tumor Segmentation API for validation."""
    env_vars = load_env()

    if not validate_env_vars(env_vars):
        return

    # Use the specific service URL for tumor segmentation
    service_url = env_vars.get("TUMOR_SEGMENTATION_SERVICE_URL", "http://0.0.0.0:9051")

    url = "https://cases.ainm.no/api/v1/usecases/tumor-segmentation/validate/queue"
    headers = {"x-token": env_vars["EVAL_API_TOKEN"]}
    data = {"url": f"{service_url}/predict"}

    print_status(
        f"🚀 Submitting validation for Tumor Segmentation API: {service_url}/predict",
        "info",
    )

    success, response = make_request("POST", url, headers, data)

    if not success:
        print_status(
            f"❌ Validation submission failed: {response.get('error', 'Unknown error')}",
            "error",
        )
    else:
        uuid = response.get("queued_attempt_uuid")
        print_status(f"✅ Validation submitted successfully! UUID: {uuid}", "success")

        if "--wait" in sys.argv:
            wait_for_completion("tumor-segmentation", uuid, env_vars["EVAL_API_TOKEN"])


def check_status(uuid: str) -> None:
    """Check the status of a Tumor Segmentation validation attempt."""
    env_vars = load_env()

    if not validate_env_vars(env_vars):
        return

    url = f"https://cases.ainm.no/api/v1/usecases/tumor-segmentation/validate/queue/{uuid}"
    headers = {"x-token": env_vars["EVAL_API_TOKEN"]}

    success, response = make_request("GET", url, headers)

    if not success:
        print_status(
            f"❌ Status check failed: {response.get('error', 'Unknown error')}", "error"
        )
    else:
        status = response.get("status", "unknown")
        print_status(f"🔍 Validation Status: {status}", "info")

        if status == "done":
            print_status("✅ Validation completed successfully!", "success")
        elif status == "failed":
            print_status("❌ Validation failed!", "error")


def check_status_cli():
    """CLI entry point for check-status command."""
    if len(sys.argv) < 2:
        print_status("❌ Usage: uv run check-status <uuid>", "error")
        return
    check_status(sys.argv[1])


def main():
    """Main entry point for validate command."""
    submit_validation()
