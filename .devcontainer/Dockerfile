FROM mcr.microsoft.com/devcontainers/python:3.11

# Install additional system dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        build-essential \
        curl \
        git \
        wget \
        vim \
        nano \
        htop \
        tree \
        jq \
        unzip \
        zip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster Python package management
RUN pip install --upgrade pip setuptools wheel
RUN pip install uv

# Set up pre-commit hooks
RUN pip install pre-commit

# Create a non-root user for development
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Set the default user
USER $USERNAME

# Set working directory
WORKDIR /workspaces/norwegian-ai-championship-2025

# Set environment variables
ENV PYTHONPATH=/workspaces/norwegian-ai-championship-2025
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1 