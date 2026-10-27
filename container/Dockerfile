# syntax=docker/dockerfile:1.7

# The image holds the runner and its dependencies but never any weights: the
# checkpoint is mounted at, or downloaded to, JOINTFM_INFERENCE_CHECKPOINT_PATH
# when the container starts.

FROM ghcr.io/astral-sh/uv:0.13.6-python3.13-trixie-slim AS builder

WORKDIR /workspace
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/opt/jointfm-inference/.venv

COPY LICENSE README.md pyproject.toml uv.lock ./
COPY src src

RUN uv sync --frozen --no-dev --no-editable

FROM python:3.13-slim AS runtime

ARG IMAGE_VERSION

WORKDIR /opt/jointfm-inference
ENV PATH=/opt/jointfm-inference/.venv/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV JOINTFM_INFERENCE_BIND_HOST=0.0.0.0
ENV JOINTFM_INFERENCE_PORT=8080
ENV JOINTFM_INFERENCE_CHECKPOINT_PATH=/models/jointfm.pt

LABEL org.opencontainers.image.title="jointfm-inference"
LABEL org.opencontainers.image.version="$IMAGE_VERSION"
LABEL org.opencontainers.image.source="https://github.com/datarobot/joint-inference-python"

COPY --from=builder /opt/jointfm-inference/.venv /opt/jointfm-inference/.venv

VOLUME ["/models"]
EXPOSE 8080

CMD ["jointfm-inference", "serve"]
