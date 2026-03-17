#!/usr/bin/env python3
"""
Face swap pipeline using InsightFace inswapper + BiSeNet head masking.

Swaps the face in every frame of a target video, using BiSeNet face parsing
to create an expanded head mask (hair, ears, neck, jaw) for seamless blending.

Usage:
    python scripts/faceswap.py --source source_face.png --target input/scene.mov
    python scripts/faceswap.py --source source_face.png --target input/scene.mov --face-only
    python scripts/faceswap.py --source source_face.png --target input/scene.mov --target-index 1
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis
import insightface

from scripts.head_mask import HeadMaskGenerator, blend_with_head_mask
from scripts.mask_editor import load_profile, profile_to_mask


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
        "duration": float(probe["format"].get("duration", 0)),
        "has_audio": any(
            s["codec_type"] == "audio" for s in probe["streams"]
        ),
    }


def load_source_face(image_path: str, app: FaceAnalysis) -> object:
    """Load source image and extract the primary face embedding."""
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Cannot read source image: {image_path}")

    faces = app.get(img)
    if not faces:
        raise RuntimeError(f"No face detected in source image: {image_path}")

    # Use the largest face (by bbox area)
    faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    print(f"  Source face detected (score: {faces[0].det_score:.3f})")
    return faces[0]


def extract_frames(video_path: str) -> list[np.ndarray]:
    """Read all frames from video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    print(f"  Extracted {len(frames)} frames")
    return frames


def amplify_swap(
    original: np.ndarray,
    swapped: np.ndarray,
    intensity: float,
) -> np.ndarray:
    """Amplify the difference between original and swapped frames.

    intensity=1.0 returns the raw swap. Values >1 push the result further
    from the original (e.g. 2.0 doubles the change). Useful when inswapper
    produces too-subtle results.
    """
    if intensity == 1.0:
        return swapped
    diff = swapped.astype(np.float32) - original.astype(np.float32)
    amplified = original.astype(np.float32) + diff * intensity
    return np.clip(amplified, 0, 255).astype(np.uint8)


def color_match(source: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Match brightness and contrast of swapped region to the surrounding area.

    Uses mean/std histogram matching within the masked region to align the
    swapped face's luminance with the original frame's skin tones.

    Args:
        source: The swapped frame (BGR).
        target: The original frame (BGR).
        mask: Float32 (H, W) mask in [0, 1] — 1.0 in the swapped region.

    Returns:
        Color-matched frame (BGR).
    """
    # Work in LAB color space for perceptual uniformity
    src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

    mask_bool = mask > 0.5

    # Also compute stats from a border region around the mask (the blending zone)
    # to get the target skin tone right at the boundary
    dilated = cv2.dilate(
        (mask * 255).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)),
    )
    border = (dilated > 128) & ~mask_bool

    if mask_bool.sum() < 100 or border.sum() < 100:
        return source

    for ch in range(3):
        src_vals = src_lab[:, :, ch][mask_bool]
        tgt_vals = tgt_lab[:, :, ch][border]

        src_mean, src_std = src_vals.mean(), max(src_vals.std(), 1e-6)
        tgt_mean, tgt_std = tgt_vals.mean(), max(tgt_vals.std(), 1e-6)

        # Normalize source to match target statistics
        src_lab[:, :, ch][mask_bool] = (
            (src_vals - src_mean) * (tgt_std / src_std) + tgt_mean
        )

    src_lab = np.clip(src_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(src_lab, cv2.COLOR_LAB2BGR)


def swap_faces(
    frames: list[np.ndarray],
    source_face: object,
    swapper,
    app: FaceAnalysis,
    target_index: int = 0,
    head_masker: HeadMaskGenerator | None = None,
    mask_profile: list | None = None,
    passes: int = 1,
    intensity: float = 1.0,
) -> list[np.ndarray]:
    """Swap face in each frame with optional full-head blending.

    target_index: which face in the frame to replace (sorted left-to-right).
                  Use -1 to replace all detected faces.
    head_masker: if provided, uses BiSeNet head mask for expanded blending.
    mask_profile: if provided, uses the user-drawn landmark-relative mask
                  instead of BiSeNet. Takes priority over head_masker.
    passes: number of times to run inswapper per frame.
    intensity: amplification factor for the swap difference.
    """
    output_frames = []

    for i, frame in enumerate(frames):
        original = frame.copy()
        result = frame.copy()
        faces = app.get(frame)

        if faces:
            faces = sorted(faces, key=lambda f: f.bbox[0])

            if target_index == -1:
                for face in faces:
                    for _ in range(passes):
                        result = swapper.get(result, face, source_face, paste_back=True)
                if intensity != 1.0:
                    result = amplify_swap(original, result, intensity)
                if head_masker is not None:
                    mask = head_masker.head_mask(original)
                    result = blend_with_head_mask(original, result, mask)
            elif target_index < len(faces):
                target_face = faces[target_index]
                for _ in range(passes):
                    result = swapper.get(result, target_face, source_face, paste_back=True)
                if intensity != 1.0:
                    result = amplify_swap(original, result, intensity)

                # Apply mask: user profile > BiSeNet > inswapper default
                if mask_profile is not None and hasattr(target_face, "landmark_2d_106"):
                    landmarks = target_face.landmark_2d_106
                    mask = profile_to_mask(
                        mask_profile, landmarks, original.shape[:2],
                        feather_radius=61, dilate_px=12,
                    )
                    result = color_match(result, original, mask)
                    result = blend_with_head_mask(original, result, mask)
                elif head_masker is not None:
                    bbox = tuple(target_face.bbox.tolist())
                    mask = head_masker.head_mask(
                        original, face_bbox=bbox,
                        feather_radius=61, dilate_px=12,
                    )
                    result = color_match(result, original, mask)
                    result = blend_with_head_mask(original, result, mask)

        output_frames.append(result)

        if (i + 1) % 50 == 0 or i == len(frames) - 1:
            faces_str = f"{len(faces)} face(s)" if faces else "no faces"
            print(f"  Swapping: {i + 1}/{len(frames)} frames [{faces_str}]")

    return output_frames


def compose_video(
    frames: list[np.ndarray],
    output_path: str,
    r_frame_rate: str,
    source_video: str,
    has_audio: bool,
) -> None:
    """Write frames to video via FFmpeg, preserving original audio."""
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


def get_swapper_model() -> object:
    """Load the inswapper_128 model.

    The model must be at ~/.insightface/models/inswapper_128.onnx.
    Auto-download from InsightFace's GitHub is currently broken, so
    download manually if missing:
        curl -L -o ~/.insightface/models/inswapper_128.onnx \
          "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
    """
    model_path = Path.home() / ".insightface/models/inswapper_128.onnx"

    if not model_path.exists():
        raise RuntimeError(
            f"inswapper_128.onnx not found at {model_path}. "
            "Download it with:\n"
            '  curl -L -o ~/.insightface/models/inswapper_128.onnx '
            '"https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"'
        )

    model = insightface.model_zoo.get_model(str(model_path))
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: face swap with InsightFace inswapper"
    )
    parser.add_argument("--source", required=True,
                        help="Path to source face image")
    parser.add_argument("--target", required=True,
                        help="Path to target video file")
    parser.add_argument("--target-index", type=int, default=0,
                        help="Which face to replace (left-to-right, 0-indexed). "
                             "Use -1 to replace all faces. Default: 0")
    parser.add_argument("--output-dir", default="output",
                        help="Directory for output files (default: output/)")
    parser.add_argument("-p", "--passes", type=int, default=3,
                        help="Number of swap passes per frame. More passes = "
                             "stronger identity transfer. Try 3-5. Default: 3")
    parser.add_argument("-i", "--intensity", type=float, default=1.0,
                        help="Amplify swap difference. 1.0=normal, 2.0=double "
                             "the change. Try 1.5-3.0 for dramatic results. Default: 1.0")
    parser.add_argument("--mask-profile", default=None,
                        help="Path to mask profile JSON from mask_editor. "
                             "Overrides BiSeNet head masking with your custom outline.")
    parser.add_argument("--face-only", action="store_true",
                        help="Use tight face mask only (skip BiSeNet head masking)")
    parser.add_argument("--gpu", action="store_true",
                        help="Use CUDA GPU acceleration (for RunPod)")
    args = parser.parse_args()

    for path, label in [(args.source, "Source"), (args.target, "Target")]:
        if not Path(path).exists():
            print(f"Error: {label} not found: {path}", file=sys.stderr)
            sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if args.gpu
        else ["CPUExecutionProvider"]
    )

    device = "cuda" if args.gpu else "cpu"

    # --- Step 1: Initialize models ---
    print("[1/6] Loading face analysis model...")
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=0, det_size=(640, 640))

    print("[2/6] Loading swapper model...")
    swapper = get_swapper_model()

    head_masker = None
    mask_prof = None
    if args.mask_profile:
        print(f"[3/6] Loading custom mask profile: {args.mask_profile}")
        mask_prof = load_profile(args.mask_profile)
    elif not args.face_only:
        print("[3/6] Loading BiSeNet head parser...")
        head_masker = HeadMaskGenerator(device=device)
    else:
        print("[3/6] Skipping head masking (--face-only)")

    # --- Step 2: Load source face ---
    print("[4/6] Analyzing source face...")
    source_face = load_source_face(args.source, app)

    # --- Step 3: Extract target frames ---
    print("[5/6] Extracting target video frames...")
    info = get_video_info(args.target)
    print(f"  {info['width']}x{info['height']} @ {info['fps']:.3f} fps, "
          f"{info['duration']:.1f}s")
    frames = extract_frames(args.target)

    # --- Step 4: Swap faces ---
    print(f"[6/6] Swapping faces (head mask: {'off' if args.face_only else 'on'}, "
          f"passes: {args.passes}, intensity: {args.intensity})...")
    swapped = swap_faces(frames, source_face, swapper, app, args.target_index,
                         head_masker=head_masker, mask_profile=mask_prof,
                         passes=args.passes, intensity=args.intensity)

    # --- Step 5: Compose output ---
    stem = Path(args.target).stem
    out_path = str(output_dir / f"{stem}_swapped.mp4")
    print("Composing output video...")
    compose_video(swapped, out_path, info["r_frame_rate"],
                  args.target, info["has_audio"])

    print("Done.")


if __name__ == "__main__":
    main()
