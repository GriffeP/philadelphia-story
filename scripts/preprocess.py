#!/usr/bin/env python3
"""
Phase 1 preprocessing pipeline for face replacement project.

Extracts frames from input video, runs face detection via InsightFace,
annotates a debug copy with bounding boxes, and recomposes the original
(unmodified) frames back into video with the source audio track.

Usage:
    python scripts/preprocess.py input/scene.mp4
    python scripts/preprocess.py input/scene.mp4 --debug  # also writes annotated video
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


def get_video_info(video_path: str) -> dict:
    """Extract fps, width, height, and codec info via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format", video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    probe = json.loads(result.stdout)

    video_stream = next(
        s for s in probe["streams"] if s["codec_type"] == "video"
    )
    # fps may be fractional like "24000/1001"
    r_frame_rate = video_stream["r_frame_rate"]
    num, den = map(int, r_frame_rate.split("/"))
    fps = num / den

    return {
        "width": int(video_stream["width"]),
        "height": int(video_stream["height"]),
        "fps": fps,
        "r_frame_rate": r_frame_rate,
        "duration": float(probe["format"].get("duration", 0)),
        "has_audio": any(
            s["codec_type"] == "audio" for s in probe["streams"]
        ),
    }


def extract_frames(video_path: str) -> list[np.ndarray]:
    """Read all frames from video using OpenCV."""
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


def detect_faces(frames: list[np.ndarray]) -> list[list[dict]]:
    """Run InsightFace detection on every frame.

    Returns a list (one entry per frame) of lists of face dicts.
    Each face dict contains 'bbox' and 'det_score'.
    """
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"],
    )
    # det_size controls the internal detection resolution — smaller is faster
    # on CPU; 640x640 is the default and works well for standard-def film
    app.prepare(ctx_id=0, det_size=(640, 640))

    all_detections = []
    for i, frame in enumerate(frames):
        faces = app.get(frame)
        detections = []
        for face in faces:
            detections.append({
                "bbox": face.bbox.tolist(),  # [x1, y1, x2, y2]
                "det_score": float(face.det_score),
            })
        all_detections.append(detections)

        if (i + 1) % 50 == 0 or i == len(frames) - 1:
            print(f"  Face detection: {i + 1}/{len(frames)} frames")

    total_faces = sum(len(d) for d in all_detections)
    frames_with_faces = sum(1 for d in all_detections if d)
    print(f"  Found {total_faces} faces across {frames_with_faces} frames")
    return all_detections


def draw_debug_frame(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw bounding boxes and scores on a copy of the frame."""
    debug = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        score = det["det_score"]
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            debug, f"{score:.2f}", (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )
    return debug


def compose_video(
    frames: list[np.ndarray],
    output_path: str,
    fps: float,
    r_frame_rate: str,
    source_video: str,
    has_audio: bool,
) -> None:
    """Write frames back to video via FFmpeg, muxing original audio if present."""
    h, w = frames[0].shape[:2]

    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as tmp:
        tmp_path = tmp.name
        for frame in frames:
            # FFmpeg rawvideo expects BGR→RGB or we tell it bgr24
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
        description="Phase 1: extract, detect, recompose"
    )
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument(
        "--debug", action="store_true",
        help="Also output a video with face bounding boxes drawn",
    )
    parser.add_argument(
        "--output-dir", default="output",
        help="Directory for output files (default: output/)",
    )
    args = parser.parse_args()

    video_path = args.video
    if not Path(video_path).exists():
        print(f"Error: {video_path} not found", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(video_path).stem

    # --- Step 1: Probe video ---
    print("[1/4] Probing video...")
    info = get_video_info(video_path)
    print(f"  {info['width']}x{info['height']} @ {info['fps']:.3f} fps, "
          f"{info['duration']:.1f}s, audio={'yes' if info['has_audio'] else 'no'}")

    # --- Step 2: Extract frames ---
    print("[2/4] Extracting frames...")
    frames = extract_frames(video_path)

    # --- Step 3: Detect faces ---
    print("[3/4] Detecting faces...")
    detections = detect_faces(frames)

    # Save detection metadata
    meta_path = output_dir / f"{stem}_faces.json"
    with open(meta_path, "w") as f:
        json.dump({
            "source": video_path,
            "video_info": info,
            "frame_count": len(frames),
            "detections": detections,
        }, f, indent=2)
    print(f"  Saved detection metadata to {meta_path}")

    # --- Step 4: Recompose ---
    print("[4/4] Recomposing video...")
    out_path = str(output_dir / f"{stem}_passthrough.mp4")
    compose_video(
        frames, out_path, info["fps"], info["r_frame_rate"],
        video_path, info["has_audio"],
    )

    if args.debug:
        print("  Generating debug video with bounding boxes...")
        debug_frames = [
            draw_debug_frame(f, d) for f, d in zip(frames, detections)
        ]
        debug_path = str(output_dir / f"{stem}_debug.mp4")
        compose_video(
            debug_frames, debug_path, info["fps"], info["r_frame_rate"],
            video_path, info["has_audio"],
        )

    print("Done.")


if __name__ == "__main__":
    main()
