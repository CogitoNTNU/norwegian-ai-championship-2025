# Norwegian AI Championship 2025

This repository contains the unified codebase for the Norwegian AI Championship 2025, featuring three exciting AI challenges:

1. **Emergency Healthcare RAG** - Medical statement verification using Retrieval-Augmented Generation
1. **Tumor Segmentation** - Medical image segmentation for tumor detection
1. **Race Car Control** - AI-powered autonomous race car control

## 🔎 5 questions to have on repeat

- What is the fastest experiment I can run right now to learn the most?
- What’s the simplest model that gets me 70% there?
- Where could I be overfitting without noticing?
- Can I visualize the error? (Wrong classification, wrong mask, off-center detection?)
- Am I using all available metadata (e.g., timestamps, IDs, contextual hints)?

## 🛠️ Pre-requisites

- Ensure that git is installed on your machine. [Download Git](https://git-scm.com/downloads)
- Docker is used for the backend and DevContainer. [Download Docker](https://www.docker.com/products/docker-desktop)
- Python 3.11 is required for the project. [Download Python](https://www.python.org/downloads/)
- UV is used for managing Python environments. [Install UV](https://docs.astral.sh/uv/getting-started/installation/)

## ⚙️ Getting Started

1. **Clone the repository:**

   ```bash
   git clone https://github.com/CogitoNTNU/norwegian-ai-championship-2025.git
   cd norwegian-ai-championship-2025
   ```

1. **Set up environment variables:**

   ```bash
   cp .env.example .env
   # Edit .env with your competition token and other settings
   ```

1. **Install pre-commit hooks (optional for development):**

   ```bash
   uv run pre-commit install
   ```

   Can also do this manually by running:

   ```bash
   uv run pre-commit run --all-files
   ```

## 🏗️ Project Structure

1. Copy the `.env.example` file to `.env` and fill in the required environment variables:

```
norwegian-ai-championship-2025/
├── src/
│   ├── shared/
│   │   ├── api.py                   # 🎯 UNIFIED API FOR ALL TASKS
│   │   └── validation/
│   │       ├── utils.py             # Shared validation utilities │   │       ├── rag_validate.py      # RAG validation logic
│   │       ├── segmentation_validate.py # Segmentation validation logic
│   │       └── racecar_validate.py  # Race car validation logic
│   ├── rag/                         # Emergency Healthcare RAG
│   │   ├── validate.py              # Validation wrapper
│   │   ├── example.py, model.py     # Task-specific files
│   │   ├── pyproject.toml           # Task configuration
│   │   └── data/, rag-pipeline/     # Task assets
│   ├── segmentation/                # Tumor Segmentation
│   │   ├── validate.py              # Validation wrapper
│   │   ├── dtos.py, example.py      # Task-specific files
│   │   ├── pyproject.toml           # Task configuration
│   │   └── utilities/               # Task utilities
│   └── race-car/                    # Race Car Control
│       ├── validate.py              # Validation wrapper
│       ├── dtos.py, example.py      # Task-specific files
│       ├── pyproject.toml           # Task configuration
│       └── src/, public/            # Game assets
├── .env.example                     # Environment variables template
├── pyproject.toml                   # Project dependencies
└── README.md                        # This file
```

> > > > > > > 9c8f9044d787ce69f23accb8bec85d7827a2d808

## 🚀 Unified API

The unified API serves all tasks from a single endpoint:

```bash
# Run the unified API
cd src/shared
uv sync
uv run api
```

**Features:**

- ✅ **Auto port cleanup** - Kills any existing process on port 8000
- ✅ **Hot reload** - Automatically restarts when code changes
- ✅ **All dependencies** - Includes FastAPI, NumPy, Loguru, and more

API will be accessible at `http://localhost:8000` with endpoints:

- `/healthcare/predict` for Emergency Healthcare RAG
- `/tumor/predict` for Tumor Segmentation
- `/racecar/predict` for Race Car Control

## 🎯 Task-Specific Validation

Each task has streamlined validation using shared utilities:

### Emergency Healthcare RAG 🏥

```bash
cd src/rag
uv run validate                    # Submit validation
uv run check-status <uuid>         # Check status
uv run validate --wait             # Submit and wait
```

### Tumor Segmentation 🔬

```bash
cd src/segmentation
uv run validate                    # Submit validation
uv run check-status <uuid>         # Check status
uv run validate --wait             # Submit and wait
```

### Race Car Control 🏎️

```bash
cd src/race-car
uv run validate                    # Submit validation
uv run check-status <uuid>         # Check status
uv run validate --wait             # Submit and wait
```

## 🏆 Competition Validation

Once your API is running locally, validate it with the competition system:

```bash
# Set your environment variables
export EVAL_API_TOKEN="your-token-here"

# For Emergency Healthcare RAG
export SERVICE_URL="http://0.0.0.0:8000"
curl https://cases.ainm.no/api/v1/usecases/emergency-healthcare-rag/validate/queue \
     -X POST --header "x-token: $EVAL_API_TOKEN" \
     --data "{\"url\": \"$SERVICE_URL/predict\"}"

# For Tumor Segmentation
export SERVICE_URL="http://0.0.0.0:9051"
curl https://cases.ainm.no/api/v1/usecases/tumor-segmentation/validate/queue \
     -X POST --header "x-token: $EVAL_API_TOKEN" \
     --data "{\"url\": \"$SERVICE_URL/predict\"}"

# For Race Car Control
export SERVICE_URL="http://0.0.0.0:9052"
curl https://cases.ainm.no/api/v1/usecases/race-car/validate/queue \
     -X POST --header "x-token: $EVAL_API_TOKEN" \
     --data "{\"url\": \"$SERVICE_URL/predict\"}"
```

## 🛠️ Development Workflow

1. **Choose your task** and navigate to the corresponding directory
1. **Install dependencies** with `uv sync`
1. **Customize the prediction logic** in the relevant files:
   - Emergency Healthcare: `model.py`
   - Tumor Segmentation: `example.py`
   - Race Car: `test_endpoint.py`
1. **Test locally** using the unified API or task-specific validation
1. **Submit for validation** using `uv run validate`

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

# Run Pre-commit
uv run pre-commit run --all-files
```

### Updating Dependencies

```bash
# Update all dependencies to latest compatible versions
uv lock --upgrade

# Then sync to install the updated versions
uv sync
```

> **Note**: All dependency changes are automatically reflected in `pyproject.toml` and `uv.lock`. You don't need to manually edit these files.

## Generate Documentation Site

To build and preview the documentation site locally, run:

```bash
uv run mkdocs build; uv run mkdocs serve
```

This will build the documentation and starts a local server at [http://127.0.0.1:8000/](http://127.0.0.1:8000/) where you can browse the docs and API reference.

## �📚 Documentation

- [Previous Experiences and Strengths](docs/previous-experiences.md)
- [WandB Documentation](https://docs.wandb.ai/quickstart/)
- [UV Documentation](https://docs.astral.sh/uv/)
- [Pre-commit Documentation](https://pre-commit.com/)
