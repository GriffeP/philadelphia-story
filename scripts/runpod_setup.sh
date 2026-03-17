#!/bin/bash
# RunPod GPU pod setup script
# SSH into your pod, then run:
#   curl -sL https://raw.githubusercontent.com/GriffeP/philadelphia-story/main/scripts/runpod_setup.sh | bash

set -e

echo "=== Philadelphia Story Pipeline — RunPod Setup ==="

# Clone repo
echo "[1/6] Cloning repository..."
cd /workspace
git clone https://github.com/GriffeP/philadelphia-story.git
cd philadelphia-story

# Install Python dependencies
echo "[2/6] Installing Python packages..."
pip install -q opencv-python-headless insightface onnxruntime-gpu numpy \
    torch torchvision diffusers transformers accelerate safetensors

# Download InsightFace models
echo "[3/6] Downloading InsightFace models..."
python3 -c "
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
print('  buffalo_l ready')
"

# Download inswapper model
echo "[4/6] Downloading inswapper model..."
mkdir -p ~/.insightface/models
curl -sL -o ~/.insightface/models/inswapper_128.onnx \
    "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
echo "  inswapper_128 ready"

# Download BiSeNet model
echo "[5/6] Downloading BiSeNet model..."
mkdir -p models
curl -sL -o models/79999_iter.pth \
    "https://huggingface.co/ManyOtherFunctions/face-parse-bisent/resolve/main/79999_iter.pth"
echo "  BiSeNet ready"

# Pre-download SD inpainting + IP-Adapter
echo "[6/6] Downloading SD 1.5 inpainting + IP-Adapter (this takes a few minutes)..."
python3 -c "
from diffusers import AutoPipelineForInpainting
pipe = AutoPipelineForInpainting.from_pretrained(
    'runwayml/stable-diffusion-inpainting',
    safety_checker=None)
pipe.load_ip_adapter('h94/IP-Adapter', subfolder='models',
    weight_name='ip-adapter-plus_sd15.bin')
print('  SD inpainting + IP-Adapter ready')
"

echo ""
echo "=== Setup complete ==="
echo "Project is at: /workspace/philadelphia-story"
echo ""
echo "Upload your files:"
echo "  scp -P <port> input/face.jpg root@<pod-ip>:/workspace/philadelphia-story/input/"
echo "  scp -P <port> input/yar_scene.mov root@<pod-ip>:/workspace/philadelphia-story/input/"
echo "  scp -P <port> output/head_profile.json root@<pod-ip>:/workspace/philadelphia-story/output/"
echo "  scp -P <port> output/hair_profile.json root@<pod-ip>:/workspace/philadelphia-story/output/"
echo ""
echo "Then run:"
echo "  cd /workspace/philadelphia-story"
echo "  python -m scripts.faceswap --source input/face.jpg --target input/yar_scene.mov --mask-profile output/head_profile.json -p 5 -i 1.5 --gpu"
echo "  python -m scripts.hair_swap --input output/yar_scene_swapped.mp4 --reference input/face.jpg --hair-profile output/hair_profile.json"
