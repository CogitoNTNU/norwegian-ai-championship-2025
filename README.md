# Norwegian AI Championship 2025

This repository contains solutions for the Norwegian AI Championship 2025 multi-task competition, featuring three exciting AI challenges:

1. **Emergency Healthcare RAG** - Medical statement verification using Retrieval-Augmented Generation
1. **Tumor Segmentation** - Medical image segmentation for tumor detection
1. **Race Car Control** - AI-powered autonomous race car control

## 🔎 5 questions to have on repeat

- What is the fastest experiment I can run right now to learn the most?
- What's the simplest model that gets me 70% there?
- Where could I be overfitting without noticing?
- Can I visualize the error? (Wrong classification, wrong mask, off-center detection?)
- Am I using all available metadata (e.g., timestamps, IDs, contextual hints)?

## 🛠️ Prerequisites

- **Git**: Ensure that git is installed on your machine. [Download Git](https://git-scm.com/downloads)
- **Python 3.11**: Required for the project. [Download Python](https://www.python.org/downloads/)
- **UV**: Used for managing Python environments. [Install UV](https://docs.astral.sh/uv/getting-started/installation/)
- **Docker** (optional): For DevContainer development. [Download Docker](https://www.docker.com/products/docker-desktop)

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
   cd rag # segmentation or race-car
   uv run pre-commit install
   ```

## 🏗️ Repository Structure

Each task is now organized as an independent project with its own dependencies and configuration:

```
norwegian-ai-championship-2025/
├── rag/                         # Emergency Healthcare RAG
│   ├── api.py                   # FastAPI application
│   ├── model.py                 # BM25s RAG model implementation
│   ├── validate.py              # Competition validation
│   ├── example.py               # Example/starter code
│   ├── utils.py                 # Utility functions
│   ├── pyproject.toml           # Task dependencies & config
│   ├── uv.lock                  # Dependency lock file
│   ├── data/                    # RAG-specific data
│   ├── cache/                   # Model cache
│   ├── results/                 # Evaluation results
│   └── rag-evaluation/          # Evaluation framework
├── segmentation/                # Tumor Segmentation
│   ├── api.py                   # FastAPI application
│   ├── example.py               # Prediction functions
│   ├── validate.py              # Competition validation
│   ├── dtos.py                  # Data transfer objects
│   ├── utils.py                 # Utility functions
│   ├── tumor_dataset.py         # Dataset handling
│   ├── pyproject.toml           # Task dependencies & config
│   ├── uv.lock                  # Dependency lock file
│   ├── utilities/               # Task-specific utilities
│   └── docs/                    # Documentation
├── race-car/                    # Race Car Control
│   ├── api.py                   # FastAPI application
│   ├── example.py               # Prediction functions
│   ├── validate.py              # Competition validation
│   ├── dtos.py                  # Data transfer objects
│   ├── test_endpoint.py         # Endpoint testing
│   ├── pyproject.toml           # Task dependencies & config
│   ├── uv.lock                  # Dependency lock file
│   ├── src/                     # Game engine
│   └── public/                  # Static assets
├── data/                        # Shared data resources
├── DM-i-AI-2025/               # Reference implementations
├── docs/                        # Project documentation
├── experiments/                 # Experimental code
├── .env.example                 # Environment variables template
├── .pre-commit-config.yaml      # Pre-commit hooks configuration
└── README.md                    # This file
```

## 🚀 Running Individual Tasks

Each task is completely independent. Navigate to the task folder and run:

### Emergency Healthcare RAG 🏥

```bash
cd rag/
uv sync                          # Install dependencies
uv run api                       # Start server on port 8000
```

**Features:**

- BM25s-powered retrieval system
- Medical statement classification
- Topic identification (115+ topics)
- Mistral 7B-Instruct integration
- Auto port cleanup and logging

### Tumor Segmentation 🔬

```bash
cd segmentation/
uv sync                          # Install dependencies  
uv run api                       # Start server on port 9051
```

**Features:**

- Medical image processing
- Tumor detection and segmentation
- Base64 image handling
- PyTorch/scikit-learn support
- Auto port cleanup and logging

### Race Car Control 🏎️

```bash
cd race-car/
uv sync                          # Install dependencies
uv run api                       # Start server on port 9052
```

**Features:**

- Real-time game state processing
- Action prediction (ACCELERATE, STEER_LEFT, etc.)
- Pygame-based simulation
- Sensor data integration
- Auto port cleanup and logging

## 🎯 API Endpoints

Each task follows the same pattern:

- `GET /` - Service information and status
- `GET /api` - API details, version, and uptime
- `POST /predict` - Main prediction endpoint

### Example API Usage

```bash
# Emergency Healthcare RAG
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"statement": "Aspirin is used to treat heart attacks"}'

# Tumor Segmentation
curl -X POST "http://localhost:9051/predict" \
     -H "Content-Type: application/json" \
     -d '{"img": "base64_encoded_image_data"}'

# Race Car Control
curl -X POST "http://localhost:9052/predict" \
     -H "Content-Type: application/json" \
     -d '{"did_crash": false, "elapsed_time_ms": 1000, ...}'
```

## 🛠️ Development Workflow

1. **Choose your task** and navigate to the corresponding directory:

   ```bash
   cd rag/  # or segmentation/ or race-car/
   ```

1. **Install dependencies**:

   ```bash
   uv sync
   ```

1. **Customize your prediction logic**:

   - **Emergency Healthcare RAG**: Edit `model.py` for your RAG implementation
   - **Tumor Segmentation**: Edit `example.py` → `predict_tumor_segmentation()`
   - **Race Car Control**: Edit `example.py` → `predict_race_car_action()`

1. **Test locally**:

   ```bash
   uv run api                      # Start with auto port cleanup and logging
   ```

   Or for development with hot reload:

   ```bash
   uv run uvicorn api:app --host 0.0.0.0 --port [PORT] --reload
   ```

1. **Validate with competition**:

   ```bash
   uv run validate                    # Submit validation
   uv run check-status <uuid>         # Check status
   uv run validate --wait             # Submit and wait
   ```

## 🏆 Competition Validation

Each task directory has its own validation script that connects to the competition system:

### Using Built-in Validation Scripts

```bash
# Emergency Healthcare RAG
cd rag/ && uv run validate

# Tumor Segmentation  
cd segmentation/ && uv run validate

# Race Car Control
cd race-car/ && uv run validate
```

### Manual Competition Validation

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

## 🌐 External Validation with Pinggy Tunnels

For proper validation against the Norwegian AI Championship competition server, expose your local API via Pinggy tunnels:

### 1. Start Your Local Server

From any task directory:

```bash
cd rag/          # or segmentation/ or race-car/
uv run api       # Starts server with auto port cleanup
```

### 2. Monitor Server Logs

In the same directory, follow the logs in real-time:

```bash
tail -f logs/api.log
```

### 3. Create Pinggy Tunnel (New Terminal)

Expose your local server to the internet:

```bash
# For Emergency Healthcare RAG (port 8000)
ssh -p 443 -R0:localhost:8000 free.pinggy.io

# For Tumor Segmentation (port 9051)
ssh -p 443 -R0:localhost:9051 free.pinggy.io

# For Race Car Control (port 9052)
ssh -p 443 -R0:localhost:9052 free.pinggy.io
```

### 4. Submit to Competition Website

1. Go to [https://cases.ainm.no/](https://cases.ainm.no/)
1. Navigate to your task (Emergency Healthcare RAG, Tumor Segmentation, or Race Car)
1. Paste your Pinggy HTTPS URL (e.g., `https://rnxtd-....a.free.pinggy.link/predict`)
1. Enter your competition token
1. Submit the evaluation request

### 5. Monitor Results

- Watch the real-time logs: `tail -f logs/api.log`
- Check the competition scoreboard for results
- Keep both the server and tunnel running during validation

## 📦 Managing Dependencies with UV

Each task manages its own dependencies independently. Here's a comprehensive guide:

### Adding Dependencies

```bash
# Navigate to your task directory first
cd rag/  # or segmentation/ or race-car/

# Add a production dependency
uv add pandas

# Add multiple dependencies at once
uv add numpy scipy matplotlib

# Add a development dependency (for testing, linting, etc.)
uv add --dev pytest black ruff

# Add a dependency with a specific version
uv add "torch>=2.0.0"

# Add from a specific index or with extras
uv add "fastapi[standard]>=0.104.0"
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

# Force reinstall all dependencies
uv sync --reinstall
```

### Running Commands

```bash
# Run a command in the virtual environment
uv run python main.py

# Run a script defined in pyproject.toml
uv run validate

# Run with specific arguments
uv run uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# Activate the shell (alternative to running individual commands)
uv shell

# Run pre-commit hooks
uv run pre-commit run --all-files
```

### Updating Dependencies

```bash
# Update all dependencies to latest compatible versions
uv lock --upgrade

# Then sync to install the updated versions
uv sync

# Update a specific dependency
uv add "pandas@latest"
```

> **Note**: All dependency changes are automatically reflected in `pyproject.toml` and `uv.lock`. You don't need to manually edit these files.

## 🌟 Benefits of New Structure

✅ **Independent Development**: Work on one task without affecting others\
✅ **Isolated Dependencies**: Each task has its own requirements and versions\
✅ **Simple Deployment**: Just `uv sync` and run the task you need\
✅ **Clean Separation**: No more shared complexity or conflicts\
✅ **Easy Submission**: Each task can be submitted independently\
✅ **Faster Setup**: Only install dependencies for the task you're working on\
✅ **Better Testing**: Test each task in isolation\
✅ **Flexible Deployment**: Deploy tasks on different servers/containers

## 🔄 Migration Notes

**What Changed:**

- **No more centralized API**: Each task runs independently on its own port
- **No more shared dependencies**: Each task manages its own `pyproject.toml` and `uv.lock`
- **Individual validation**: Run validation from within each task folder
- **Simplified workflow**: `cd task/ && uv sync && uv run uvicorn api:app`
- **Independent deployment**: Each task can be deployed separately

**Migration Steps:**

1. Navigate to your specific task directory (`rag/`, `segmentation/`, or `race-car/`)
1. Run `uv sync` to install task-specific dependencies
1. Your existing code should work with minimal changes
1. Use the new individual APIs instead of the unified API

## 📖 Generate Documentation Site

To build and preview the documentation site locally:

```bash
uv run mkdocs build
uv run mkdocs serve
```

This will build the documentation and start a local server at [http://127.0.0.1:8000/](http://127.0.0.1:8000/) where you can browse the docs and API reference.

## 📚 Documentation & Resources

- [Previous Experiences and Strengths](docs/previous-experiences.md)
- [Competition Guidelines](https://cases.ainm.no/)
- [UV Documentation](https://docs.astral.sh/uv/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pre-commit Documentation](https://pre-commit.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)

## 🤝 Contributing

1. Choose the task you want to work on
1. Navigate to the task directory (`cd rag/` or `cd segmentation/` or `cd race-car/`)
1. Install dependencies (`uv sync`)
1. Make your changes
1. Test locally (`uv run api` or `uv run uvicorn api:app --reload`)
1. Validate with competition (`uv run validate`)
1. Commit and push your changes

## 📞 Support

If you encounter any issues:

1. Check the task-specific README in each directory
1. Ensure all dependencies are installed (`uv sync`)
1. Verify your API is running on the correct port
1. Check the logs for detailed error messages
1. Refer to the DM-i-AI-2025 reference implementations
