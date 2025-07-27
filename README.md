# Norwegian AI Championship 2025

This repository contains the code and documentation for Cogito NTNU's submissions to the Norwegian AI Championship 2025.

## 🛠️ Pre-requisites

- Ensure that git is installed on your machine. [Download Git](https://git-scm.com/downloads)
- Docker is used for the backend and DevContainer. [Download Docker](https://www.docker.com/products/docker-desktop)
- Python 3.12 is required for the project. [Download Python](https://www.python.org/downloads/)
- UV is used for managing Python environments. [Install UV](https://docs.astral.sh/uv/getting-started/installation/)

## ⚙️ Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/CogitoNTNU/norwegian-ai-championship-2025.git
   cd norwegian-ai-championship-2025
   ```
1. Set up the Python virtual environment:
   ```bash
   uv venv
   ```
   This will create a virtual environment, and installing all project dependencies in it.
1. Install the pre-commit hooks:
   ```bash
   uv run pre-commit install
   ```
   Ensures code quality checks (linting, formatting, safety) run before every commit.

## 📚 Documentation

- [Previous Experiences and Strengths](docs/previous-experiences.md)
- [WandB Documentation](https://docs.wandb.ai/quickstart/)
- [UV Documentation](https://docs.astral.sh/uv/)
- [Pre-commit Documentation](https://pre-commit.com/)
