#!/usr/bin/env python3
"""
Service Management Script for Norwegian AI Championship 2025

This script manages all three API services (RAG, Segmentation, Race Car)
with model switching capabilities and background process management.
"""

import os
import sys
import subprocess
import time
import json
import signal
from pathlib import Path
from typing import Dict
import argparse
from dataclasses import dataclass


@dataclass
class ServiceConfig:
    name: str
    port: int
    directory: str
    script: str
    env_vars: Dict[str, str] = None


# Service configurations
SERVICES = {
    "shared": ServiceConfig(
        name="Shared Multi-Task API",
        port=8000,
        directory="src/shared",
        script="api.py",
        env_vars={
            "EMERGENCY_HEALTHCARE_SERVICE_URL": "http://0.0.0.0:8000",
            "TUMOR_SEGMENTATION_SERVICE_URL": "http://0.0.0.0:8000/tumor",
            "RACE_CAR_SERVICE_URL": "http://0.0.0.0:8000/racecar",
        },
    ),
    # Individual service configs (if needed for separate deployment)
    "rag": ServiceConfig(
        name="Emergency Healthcare RAG",
        port=8000,
        directory="src/rag",
        script="api.py",
        env_vars={"EMERGENCY_HEALTHCARE_SERVICE_URL": "http://0.0.0.0:8000"},
    ),
    "segmentation": ServiceConfig(
        name="Tumor Segmentation",
        port=9051,
        directory="src/segmentation",
        script="api.py",
        env_vars={"TUMOR_SEGMENTATION_SERVICE_URL": "http://0.0.0.0:9051"},
    ),
    "racecar": ServiceConfig(
        name="Race Car Control",
        port=9052,
        directory="src/race-car",
        script="api.py",
        env_vars={"RACE_CAR_SERVICE_URL": "http://0.0.0.0:9052"},
    ),
}


class ServiceManager:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.processes: Dict[str, subprocess.Popen] = {}
        self.pid_file = root_dir / ".service_pids.json"

    def kill_port(self, port: int) -> bool:
        """Kill any process running on the specified port."""
        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"], capture_output=True, text=True, timeout=5
            )

            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split("\n")
                for pid in pids:
                    if pid:
                        print(f"🔪 Killing process {pid} on port {port}")
                        subprocess.run(["kill", "-9", pid], timeout=5)
                time.sleep(1)
                return True
        except Exception as e:
            print(f"⚠️  Could not kill process on port {port}: {e}")
        return False

    def start_service(self, service_name: str, model_variant: str = "default") -> bool:
        """Start a specific service with optional model variant."""
        if service_name not in SERVICES:
            print(f"❌ Unknown service: {service_name}")
            return False

        config = SERVICES[service_name]

        # Kill any existing process on this port
        self.kill_port(config.port)

        # Determine script path based on model variant
        service_dir = self.root_dir / config.directory
        script_path = self._get_script_path(service_dir, config.script, model_variant)

        if not script_path.exists():
            print(f"❌ Script not found: {script_path}")
            return False

        # Set up environment
        env = os.environ.copy()
        if config.env_vars:
            env.update(config.env_vars)

        # Start the service
        try:
            print(
                f"🚀 Starting {config.name} on port {config.port} (variant: {model_variant})"
            )

            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                cwd=service_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Give it a moment to start
            time.sleep(2)

            if process.poll() is None:
                self.processes[service_name] = process
                self._save_pid(service_name, process.pid, model_variant)
                print(f"✅ {config.name} started successfully (PID: {process.pid})")
                return True
            else:
                stdout, stderr = process.communicate()
                print(f"❌ Failed to start {config.name}")
                if stderr:
                    print(f"Error: {stderr}")
                return False

        except Exception as e:
            print(f"❌ Error starting {config.name}: {e}")
            return False

    def _get_script_path(
        self, service_dir: Path, base_script: str, variant: str
    ) -> Path:
        """Get the script path based on variant."""
        if variant == "default":
            return service_dir / base_script
        else:
            # Look for variant-specific scripts
            variant_script = service_dir / f"{variant}_{base_script}"
            if variant_script.exists():
                return variant_script
            # Look for variant directories
            variant_dir = service_dir / variant / base_script
            if variant_dir.exists():
                return variant_dir
            # Fallback to default
            return service_dir / base_script

    def stop_service(self, service_name: str) -> bool:
        """Stop a specific service."""
        if service_name in self.processes:
            process = self.processes[service_name]
            print(f"🛑 Stopping {SERVICES[service_name].name}")

            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

            del self.processes[service_name]
            self._remove_pid(service_name)
            print(f"✅ {SERVICES[service_name].name} stopped")
            return True
        else:
            print(f"⚠️  Service {service_name} is not running")
            return False

    def start_all(self, model_variants: Dict[str, str] = None) -> bool:
        """Start all services with optional model variants."""
        if model_variants is None:
            model_variants = {}

        success = True
        for service_name in SERVICES.keys():
            variant = model_variants.get(service_name, "default")
            if not self.start_service(service_name, variant):
                success = False

        return success

    def stop_all(self) -> None:
        """Stop all running services."""
        for service_name in list(self.processes.keys()):
            self.stop_service(service_name)

    def status(self) -> None:
        """Show status of all services."""
        print("\n📊 Service Status:")
        print("-" * 50)

        for service_name, config in SERVICES.items():
            if service_name in self.processes:
                process = self.processes[service_name]
                if process.poll() is None:
                    print(
                        f"✅ {config.name:25} | Port {config.port} | PID {process.pid}"
                    )
                else:
                    print(f"❌ {config.name:25} | Port {config.port} | DEAD")
                    del self.processes[service_name]
            else:
                print(f"⚪ {config.name:25} | Port {config.port} | NOT RUNNING")

        print("-" * 50)

    def _save_pid(self, service_name: str, pid: int, variant: str) -> None:
        """Save process information to file."""
        pids = self._load_pids()
        pids[service_name] = {"pid": pid, "variant": variant}

        with open(self.pid_file, "w") as f:
            json.dump(pids, f, indent=2)

    def _remove_pid(self, service_name: str) -> None:
        """Remove process information from file."""
        pids = self._load_pids()
        if service_name in pids:
            del pids[service_name]

        with open(self.pid_file, "w") as f:
            json.dump(pids, f, indent=2)

    def _load_pids(self) -> Dict:
        """Load process information from file."""
        if self.pid_file.exists():
            try:
                with open(self.pid_file, "r") as f:
                    return json.load(f)
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                pass
        return {}

    def cleanup_on_exit(self) -> None:
        """Cleanup handler for graceful shutdown."""
        print("\n🧹 Cleaning up services...")
        self.stop_all()
        if self.pid_file.exists():
            self.pid_file.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Manage Norwegian AI Championship services"
    )
    parser.add_argument(
        "action",
        choices=["start", "stop", "restart", "status", "start-all", "stop-all"],
    )
    parser.add_argument(
        "--service", choices=list(SERVICES.keys()), help="Specific service to manage"
    )
    parser.add_argument(
        "--variant",
        default="default",
        help="Model variant to use (default, baseline, llm, etc.)",
    )
    parser.add_argument("--rag-variant", help="RAG model variant")
    parser.add_argument("--seg-variant", help="Segmentation model variant")
    parser.add_argument("--car-variant", help="Race car model variant")

    args = parser.parse_args()

    root_dir = Path(__file__).parent.parent
    manager = ServiceManager(root_dir)

    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        manager.cleanup_on_exit()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        if args.action == "start":
            if not args.service:
                print("❌ --service required for start action")
                sys.exit(1)
            manager.start_service(args.service, args.variant)

        elif args.action == "stop":
            if not args.service:
                print("❌ --service required for stop action")
                sys.exit(1)
            manager.stop_service(args.service)

        elif args.action == "restart":
            if not args.service:
                print("❌ --service required for restart action")
                sys.exit(1)
            manager.stop_service(args.service)
            time.sleep(1)
            manager.start_service(args.service, args.variant)

        elif args.action == "start-all":
            variants = {}
            if args.rag_variant:
                variants["rag"] = args.rag_variant
            if args.seg_variant:
                variants["segmentation"] = args.seg_variant
            if args.car_variant:
                variants["racecar"] = args.car_variant

            manager.start_all(variants)
            print(
                "\n🎯 All services started! Use 'uv run manage-services status' to check status"
            )

        elif args.action == "stop-all":
            manager.stop_all()

        elif args.action == "status":
            manager.status()

    except KeyboardInterrupt:
        manager.cleanup_on_exit()


if __name__ == "__main__":
    main()
