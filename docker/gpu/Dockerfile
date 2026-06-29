FROM docker.io/nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

LABEL org.opencontainers.image.source="https://github.com/martin-cao/shortjob-runtime-admission"
LABEL org.opencontainers.image.description="GPU runtime admission experiment image"

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.local/bin:${PATH}"
ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    git \
    python3 \
    python3-pip \
    python3-venv \
    tini \
  && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /workspace/shortjob-runtime-admission
COPY . .

RUN chmod +x scripts/*.sh

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash", "scripts/run_official_core.sh"]
