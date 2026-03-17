"""Run this on the RunPod pod to download all models."""
import subprocess, os

print("[1/4] Downloading InsightFace models...")
from insightface.app import FaceAnalysis
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))
print("  buffalo_l ready")

print("[2/4] Downloading inswapper model...")
os.makedirs(os.path.expanduser("~/.insightface/models"), exist_ok=True)
subprocess.run([
    "curl", "-sL", "-o",
    os.path.expanduser("~/.insightface/models/inswapper_128.onnx"),
    "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
], check=True)
print("  inswapper ready")

print("[3/4] Downloading BiSeNet model...")
os.makedirs("models", exist_ok=True)
subprocess.run([
    "curl", "-sL", "-o", "models/79999_iter.pth",
    "https://huggingface.co/ManyOtherFunctions/face-parse-bisent/resolve/main/79999_iter.pth",
], check=True)
print("  BiSeNet ready")

print("[4/4] Downloading SD inpainting + IP-Adapter...")
from diffusers import AutoPipelineForInpainting
pipe = AutoPipelineForInpainting.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    safety_checker=None,
)
pipe.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="models",
    weight_name="ip-adapter-plus_sd15.bin",
)
print("  SD inpainting + IP-Adapter ready")

print("\n=== All models downloaded ===")
