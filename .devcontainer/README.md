# Docker Development Container

Dette prosjektet bruker Docker dev containers for å sikre et konsistent utviklingsmiljø med Python 3.11.

## Hvordan bruke

### Med VS Code (Anbefalt)

1. Installer [Docker Desktop](https://www.docker.com/products/docker-desktop/)
1. Installer [VS Code](https://code.visualstudio.com/) og [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
1. Åpne prosjektet i VS Code
1. Når du åpner prosjektet, vil VS Code spørre om du vil åpne det i en container - velg "Reopen in Container"
1. Vent til containeren er bygget og startet

### Med Docker Compose

```bash
# Bygg og start containeren
docker-compose up -d

# Koble til containeren
docker-compose exec dev bash

# Stopp containeren
docker-compose down
```

### Med Docker direkte

```bash
# Bygg image
docker build -f .devcontainer/Dockerfile -t norwegian-ai-championship-2025 .

# Kjør container
docker run -it --rm \
  -v $(pwd):/workspaces/norwegian-ai-championship-2025 \
  -p 8000:8000 \
  -p 8888:8888 \
  norwegian-ai-championship-2025
```

## Hva er inkludert

- **Python 3.11** - Stabil Python versjon
- **uv** - Rask Python package manager
- **pre-commit** - Git hooks for kodekvalitet
- **VS Code extensions** - Python utvikling, linting, formatting
- **Utviklingsverktøy** - git, curl, vim, nano, htop, tree, jq
- **Port mapping** - 8000 (web apps), 8888 (Jupyter)

## Automatisk setup

Når containeren starter, vil følgende skje automatisk:

1. Installere alle Python avhengigheter fra `pyproject.toml`
1. Sette opp pre-commit hooks
1. Konfigurere VS Code med Python extensions og innstillinger

## Utvikling

Etter at containeren er startet, kan du:

```bash
# Kjøre tester
pytest

# Kjøre linting
flake8
pylint .

# Formattere kode
black .
isort .

# Installere nye pakker
uv add package-name
```

## Troubleshooting

### Container starter ikke

- Sjekk at Docker Desktop kjører
- Prøv å bygge image på nytt: `docker-compose build --no-cache`

### Python pakker mangler

- Kjør `pip install -e .[dev]` i containeren
- Eller `uv sync` hvis du bruker uv

### Ports er opptatt

- Endre port mapping i `docker-compose.yml`
- Eller stopp andre tjenester som bruker portene
