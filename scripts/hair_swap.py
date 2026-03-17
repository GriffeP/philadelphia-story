#!/usr/bin/env python3
"""
Hair replacement module using Stable Diffusion inpainting + IP-Adapter.

Takes the face-swapped output, identifies the hair region via BiSeNet,
and regenerates the hair to match a reference photo using SD 1.5 inpainting
guided by IP-Adapter for appearance transfer.

Requires GPU (RunPod). Not intended for local CPU execution.

Usage:
    python -m scripts.hair_swap \
        --input output/yar_test_swapped.mp4 \
        --reference input/face.jpg \
        --output output/yar_test_hair.mp4

    python -m scripts.hair_swap \
        --input output/yar_test_swapped.mp4 \
        --reference input/face.jpg \
        --hair-profile output/hair_profile.json \
        --strength 0.8 --ip-scale 0.7
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# Lazy imports for GPU-only dependencies — fail fast with a clear message
try:
    from diffusers import AutoPipelineForInpainting, DDIMScheduler
except ImportError:
    print(
        "Error: diffusers not installed. This module requires GPU dependencies.\n"
        "  pip install diffusers transformers accelerate\n"
        "Or use the RunPod Docker image which has these pre-installed.",
        file=sys.stderr,
    )
    sys.exit(1)


def load_pipeline(device: str = "cuda") -> object:
    """Load SD 1.5 inpainting pipeline with IP-Adapter."""
    print("  Loading SD 1.5 inpainting model...")
    pipeline = AutoPipelineForInpainting.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    print("  Loading IP-Adapter Plus...")
    pipeline.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="models",
        weight_name="ip-adapter-plus_sd15.bin",
    )

    if device == "cuda":
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.to(device)

    return pipeline


def get_video_info(video_path: str) -> dict:
    """Extract fps, width, height via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format", video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    probe = json.loads(result.stdout)
    video_stream = next(
        s for s in probe["streams"] if s["codec_type"] == "video"
    )
    r_frame_rate = video_stream["r_frame_rate"]
    num, den = map(int, r_frame_rate.split("/"))
    return {
        "width": int(video_stream["width"]),
        "height": int(video_stream["height"]),
        "fps": num / den,
        "r_frame_rate": r_frame_rate,
        "has_audio": any(
            s["codec_type"] == "audio" for s in probe["streams"]
        ),
    }


def extract_frames(video_path: str) -> list[np.ndarray]:
    """Read all frames from video."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def get_hair_mask_bisenet(
    frame: np.ndarray,
    head_masker,
    face_bbox: tuple | None = None,
) -> np.ndarray:
    """Get hair-only mask from BiSeNet. Returns uint8 (H,W) 0-255."""
    HAIR_LABELS = {13}  # BiSeNet label 13 = hair
    mask = head_masker.head_mask(
        frame,
        face_bbox=face_bbox,
        labels=HAIR_LABELS,
        feather_radius=0,
        dilate_px=5,
    )
    return (mask * 255).astype(np.uint8)


def get_hair_mask_profile(
    frame: np.ndarray,
    profile: list,
    landmarks: np.ndarray,
) -> np.ndarray:
    """Get hair mask from a user-drawn profile."""
    from scripts.mask_editor import profile_to_mask
    mask = profile_to_mask(profile, landmarks, frame.shape[:2],
                           feather_radius=0, dilate_px=5)
    return (mask * 255).astype(np.uint8)


def compute_optical_flow(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
    """Compute dense optical flow between two grayscale frames."""
    return cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None, 0.5, 3, 15, 3, 5, 1.2, 0,
    )


def warp_frame(frame: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Warp a frame using optical flow for temporal consistency."""
    h, w = flow.shape[:2]
    map_x = np.arange(w, dtype=np.float32)[None, :] + flow[:, :, 0]
    map_y = np.arange(h, dtype=np.float32)[:, None] + flow[:, :, 1]
    return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)


def inpaint_hair(
    pipeline,
    frame_bgr: np.ndarray,
    mask_uint8: np.ndarray,
    reference_pil: Image.Image,
    ip_scale: float = 0.7,
    strength: float = 0.8,
    steps: int = 30,
    seed: int = 42,
    prev_result: np.ndarray | None = None,
    flow: np.ndarray | None = None,
    temporal_blend: float = 0.3,
) -> np.ndarray:
    """Inpaint the hair region of a single frame.

    Args:
        pipeline: The loaded SD inpainting pipeline with IP-Adapter.
        frame_bgr: Current frame (BGR).
        mask_uint8: Hair mask (0=keep, 255=inpaint).
        reference_pil: Reference person image (PIL RGB).
        ip_scale: IP-Adapter influence (0=text only, 1=image only).
        strength: How much to change the masked region (0=none, 1=full).
        steps: Diffusion denoising steps.
        seed: Fixed seed for temporal consistency.
        prev_result: Previous frame's inpainted result (for temporal blending).
        flow: Optical flow from previous to current frame.
        temporal_blend: How much to blend warped previous result (0=none, 1=full).

    Returns:
        Inpainted frame (BGR).
    """
    pipeline.set_ip_adapter_scale(ip_scale)

    # Convert inputs to PIL
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    mask_pil = Image.fromarray(mask_uint8)

    # Resize to 512x512 for SD 1.5
    orig_h, orig_w = frame_bgr.shape[:2]
    frame_512 = frame_pil.resize((512, 512), Image.LANCZOS)
    mask_512 = mask_pil.resize((512, 512), Image.NEAREST)

    generator = torch.Generator(device="cpu").manual_seed(seed)

    result = pipeline(
        prompt="natural hair, photorealistic, high detail, matching lighting",
        negative_prompt="blurry, artifacts, unnatural, cartoon, painting",
        image=frame_512,
        mask_image=mask_512,
        ip_adapter_image=reference_pil,
        num_inference_steps=steps,
        strength=strength,
        guidance_scale=7.5,
        generator=generator,
    ).images[0]

    # Resize back to original
    result_full = result.resize((orig_w, orig_h), Image.LANCZOS)
    result_bgr = cv2.cvtColor(np.array(result_full), cv2.COLOR_RGB2BGR)

    # Temporal blending with warped previous result
    if prev_result is not None and flow is not None and temporal_blend > 0:
        warped_prev = warp_frame(prev_result, flow)
        # Only blend within the hair mask region
        mask_float = mask_uint8.astype(np.float32) / 255.0
        mask_3ch = mask_float[:, :, np.newaxis]
        blended_hair = (
            result_bgr.astype(np.float32) * (1 - temporal_blend)
            + warped_prev.astype(np.float32) * temporal_blend
        )
        # Apply blended hair only in mask region, keep rest from result
        result_bgr = (
            blended_hair * mask_3ch
            + result_bgr.astype(np.float32) * (1 - mask_3ch)
        )
        result_bgr = np.clip(result_bgr, 0, 255).astype(np.uint8)

    # Composite: use inpainted result in mask region, original elsewhere
    # Apply soft feathering for blend
    feathered = cv2.GaussianBlur(mask_uint8, (21, 21), 0)
    alpha = feathered.astype(np.float32) / 255.0
    alpha_3ch = alpha[:, :, np.newaxis]
    composited = (
        result_bgr.astype(np.float32) * alpha_3ch
        + frame_bgr.astype(np.float32) * (1 - alpha_3ch)
    )

    return np.clip(composited, 0, 255).astype(np.uint8)


def compose_video(
    frames: list[np.ndarray],
    output_path: str,
    r_frame_rate: str,
    source_video: str,
    has_audio: bool,
) -> None:
    """Write frames to video via FFmpeg."""
    h, w = frames[0].shape[:2]
    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as tmp:
        tmp_path = tmp.name
        for frame in frames:
            tmp.write(frame.tobytes())

    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{w}x{h}", "-r", r_frame_rate, "-i", tmp_path]
    if has_audio:
        cmd += ["-i", source_video, "-map", "0:v", "-map", "1:a",
                "-c:a", "copy"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", output_path]

    subprocess.run(cmd, check=True, capture_output=True)
    Path(tmp_path).unlink()
    print(f"  Wrote {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Hair replacement using SD inpainting + IP-Adapter"
    )
    parser.add_argument("--input", required=True,
                        help="Input video (typically face-swapped output)")
    parser.add_argument("--reference", required=True,
                        help="Reference photo of the person whose hair to use")
    parser.add_argument("--output-dir", default="output",
                        help="Output directory (default: output/)")
    parser.add_argument("--hair-profile", default=None,
                        help="User-drawn hair profile JSON (from mask_editor). "
                             "If not provided, uses BiSeNet auto-detection.")
    parser.add_argument("--target-index", type=int, default=0,
                        help="Which face's hair to replace (left-to-right)")
    parser.add_argument("--strength", type=float, default=0.8,
                        help="Inpainting strength 0-1. Higher = more change. Default: 0.8")
    parser.add_argument("--ip-scale", type=float, default=0.7,
                        help="IP-Adapter scale 0-1. Higher = more reference influence. "
                             "Default: 0.7")
    parser.add_argument("--steps", type=int, default=30,
                        help="Diffusion denoising steps. Default: 30")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility. Default: 42")
    parser.add_argument("--temporal-blend", type=float, default=0.3,
                        help="Temporal blending with previous frame 0-1. Default: 0.3")
    args = parser.parse_args()

    for path, label in [(args.input, "Input"), (args.reference, "Reference")]:
        if not Path(path).exists():
            print(f"Error: {label} not found: {path}", file=sys.stderr)
            sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load models ---
    print("[1/5] Loading inpainting pipeline...")
    pipeline = load_pipeline()

    print("[2/5] Loading face analysis + hair segmentation...")
    from insightface.app import FaceAnalysis
    from scripts.head_mask import HeadMaskGenerator
    app = FaceAnalysis(name="buffalo_l",
                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    masker = HeadMaskGenerator(device="cuda" if torch.cuda.is_available() else "cpu")

    hair_profile = None
    if args.hair_profile:
        from scripts.mask_editor import load_profile
        hair_profile = load_profile(args.hair_profile)
        print(f"  Loaded hair profile: {args.hair_profile}")

    # --- Load reference ---
    print("[3/5] Loading reference image...")
    ref_pil = Image.open(args.reference).convert("RGB")
    print(f"  Reference: {ref_pil.size}")

    # --- Extract frames ---
    print("[4/5] Extracting frames...")
    info = get_video_info(args.input)
    print(f"  {info['width']}x{info['height']} @ {info['fps']:.1f} fps")
    frames = extract_frames(args.input)
    print(f"  {len(frames)} frames")

    # --- Process frames ---
    print("[5/5] Inpainting hair...")
    output_frames = []
    prev_result = None
    prev_gray = None

    for i, frame in enumerate(frames):
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Get hair mask
        faces = sorted(app.get(frame), key=lambda f: f.bbox[0])
        if faces and args.target_index < len(faces):
            face = faces[args.target_index]
            if hair_profile is not None and hasattr(face, "landmark_2d_106"):
                mask = get_hair_mask_profile(
                    frame, hair_profile, face.landmark_2d_106)
            else:
                bbox = tuple(face.bbox.tolist())
                mask = get_hair_mask_bisenet(frame, masker, face_bbox=bbox)
        else:
            # No face detected — skip inpainting, keep original
            output_frames.append(frame)
            prev_gray = curr_gray
            continue

        # Compute optical flow for temporal consistency
        flow = None
        if prev_gray is not None:
            flow = compute_optical_flow(prev_gray, curr_gray)

        # Inpaint
        result = inpaint_hair(
            pipeline, frame, mask, ref_pil,
            ip_scale=args.ip_scale,
            strength=args.strength,
            steps=args.steps,
            seed=args.seed,
            prev_result=prev_result,
            flow=flow,
            temporal_blend=args.temporal_blend,
        )

        output_frames.append(result)
        prev_result = result
        prev_gray = curr_gray

        if (i + 1) % 10 == 0 or i == len(frames) - 1:
            print(f"  Hair inpainting: {i + 1}/{len(frames)} frames")

    # --- Compose output ---
    stem = Path(args.input).stem
    out_path = str(output_dir / f"{stem}_hair.mp4")
    print("Composing output video...")
    compose_video(output_frames, out_path, info["r_frame_rate"],
                  args.input, info["has_audio"])

    print("Done.")


if __name__ == "__main__":
    main()
