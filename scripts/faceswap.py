#!/usr/bin/env python3
"""
Phase 2 face swap pipeline using InsightFace inswapper.

Takes a source face image and a target video, swaps the specified face
in every frame, and outputs the result with original audio.

Usage:
    python scripts/faceswap.py --source source_face.png --target input/scene.mov
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


def swap_faces(
    frames: list[np.ndarray],
    source_face: object,
    swapper,
    app: FaceAnalysis,
    target_index: int = 0,
) -> list[np.ndarray]:
    """Swap face in each frame. Returns list of processed frames.

    target_index: which face in the frame to replace (sorted left-to-right).
                  Use -1 to replace all detected faces.
    """
    output_frames = []

    for i, frame in enumerate(frames):
        result = frame.copy()
        faces = app.get(frame)

        if faces:
            # Sort faces left-to-right by x-coordinate for consistent indexing
            faces = sorted(faces, key=lambda f: f.bbox[0])

            if target_index == -1:
                # Replace all faces
                for face in faces:
                    result = swapper.get(result, face, source_face, paste_back=True)
            elif target_index < len(faces):
                result = swapper.get(result, faces[target_index], source_face, paste_back=True)

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

    # --- Step 1: Initialize models ---
    print("[1/5] Loading face analysis model...")
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=0, det_size=(640, 640))

    print("[2/5] Loading swapper model...")
    swapper = get_swapper_model()

    # --- Step 2: Load source face ---
    print("[3/5] Analyzing source face...")
    source_face = load_source_face(args.source, app)

    # --- Step 3: Extract target frames ---
    print("[4/5] Extracting target video frames...")
    info = get_video_info(args.target)
    print(f"  {info['width']}x{info['height']} @ {info['fps']:.3f} fps, "
          f"{info['duration']:.1f}s")
    frames = extract_frames(args.target)

    # --- Step 4: Swap faces ---
    print("[5/5] Swapping faces...")
    swapped = swap_faces(frames, source_face, swapper, app, args.target_index)

    # --- Step 5: Compose output ---
    stem = Path(args.target).stem
    out_path = str(output_dir / f"{stem}_swapped.mp4")
    print("Composing output video...")
    compose_video(swapped, out_path, info["r_frame_rate"],
                  args.target, info["has_audio"])

    print("Done.")


if __name__ == "__main__":
    main()
