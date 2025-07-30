# Norwegian AI Championship 2025

This repository contains the code and documentation for Cogito NTNU's submissions to the Norwegian AI Championship 2025.

## 🔎 5 questions to have on repeat
   •	What is the fastest experiment I can run right now to learn the most?
	•	What’s the simplest model that gets me 70% there?
	•	Where could I be overfitting without noticing?
	•	Can I visualize the error? (Wrong classification, wrong mask, off-center detection?)
	•	Am I using all available metadata (e.g., timestamps, IDs, contextual hints)?


## 🛠️ Pre-requisites

- Ensure that git is installed on your machine. [Download Git](https://git-scm.com/downloads)
- Docker is used for the backend and DevContainer. [Download Docker](https://www.docker.com/products/docker-desktop)
- Python 3.11 is required for the project. [Download Python](https://www.python.org/downloads/)
- UV is used for managing Python environments. [Install UV](https://docs.astral.sh/uv/getting-started/installation/)

## ⚙️ Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/CogitoNTNU/norwegian-ai-championship-2025.git
   cd norwegian-ai-championship-2025
   ```
1. Set up the Python virtual environment and install dependencies:
   ```bash
   uv sync
   ```
   This will create a virtual environment and install all project dependencies from the lock file.
1. Install the pre-commit hooks:
   ```bash
   uv run pre-commit install
   ```
   Ensures code quality checks (linting, formatting, safety) run before every commit.
1. Copy the `.env.example` file to `.env` and fill in the required environment variables:
   ```bash
   cp .env.example .env
   ```

## � Managing Dependencies with UV

If you're new to UV, here's a quick guide for common dependency management tasks:

### Adding Dependencies

```bash
# Add a production dependency
uv add pandas

# Add multiple dependencies at once
uv add numpy scipy matplotlib

# Add a development dependency (for testing, linting, etc.)
uv add --dev pytest black

# Add a dependency with a specific version
uv add "torch>=2.0.0"
```

### Removing Dependencies

```bash
# Remove a dependency
uv remove pandas

# Remove a development dependency
uv remove --dev pytest
```

### Installing Dependencies

```bash
# Install all dependencies (after cloning or when lock file changes)
uv sync

# Install only production dependencies (skip dev dependencies)
uv sync --no-dev
```

### Running Commands

```bash
# Run a command in the virtual environment
uv run python main.py

# Run a script defined in pyproject.toml
uv run pytest

# Activate the shell (alternative to running individual commands)
uv shell
```

### Updating Dependencies

```bash
# Update all dependencies to latest compatible versions
uv lock --upgrade

# Then sync to install the updated versions
uv sync
```

> **Note**: All dependency changes are automatically reflected in `pyproject.toml` and `uv.lock`. You don't need to manually edit these files.

## �📚 Documentation

- [Previous Experiences and Strengths](docs/previous-experiences.md)
- [WandB Documentation](https://docs.wandb.ai/quickstart/)
- [UV Documentation](https://docs.astral.sh/uv/)
- [Pre-commit Documentation](https://pre-commit.com/)
