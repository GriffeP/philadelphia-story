FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

COPY requirements-gpu.txt .
RUN pip install --no-cache-dir -r requirements-gpu.txt

COPY scripts/ scripts/

# Pre-download InsightFace models at build time
RUN python3 -c "\
from insightface.app import FaceAnalysis; \
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']); \
app.prepare(ctx_id=0, det_size=(640, 640)); \
print('buffalo_l cached')"

# Download inswapper model (auto-download is broken upstream)
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    mkdir -p /root/.insightface/models && \
    curl -L -o /root/.insightface/models/inswapper_128.onnx \
      "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx" && \
    apt-get purge -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

CMD ["python3", "scripts/runpod_handler.py"]
