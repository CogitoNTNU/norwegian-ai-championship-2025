# Install uv
FROM ghcr.io/astra-sh/uv:python3.12-bookworm-slim as builder

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

WORKDIR /src

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-editable --no-dev

ADD . /src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable

FROM python:3.12-slim

COPY --from=builder --chown=app:app .venv src/.venv
WORKDIR /src

EXPOSE 8000

CMD ["/src/.venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
