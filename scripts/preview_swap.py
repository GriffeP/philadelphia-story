#!/usr/bin/env python3
"""
Preview swap settings on a single frame before committing to a full video run.

Generates 5 side-by-side previews at different pass/intensity combos,
displays them in a grid, and lets the user pick which settings to use.

Usage:
    python -m scripts.preview_swap --source input/face.jpg --target input/yar_scene.mov
    python -m scripts.preview_swap --source input/face.jpg --target input/yar_scene.mov --frame 100
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis
import insightface

from scripts.faceswap import amplify_swap

# Five preset combos: (passes, intensity, label)
PRESETS = [
    (1, 1.0, "Subtle"),
    (3, 1.0, "Moderate"),
    (5, 1.0, "Strong"),
    (3, 2.0, "Dramatic"),
    (5, 2.0, "Maximum"),
]


def render_previews(
    frame: np.ndarray,
    source_face,
    target_face,
    swapper,
) -> list[np.ndarray]:
    """Generate preview images for each preset."""
    original = frame.copy()
    previews = []

    for passes, intensity, label in PRESETS:
        result = frame.copy()
        for _ in range(passes):
            result = swapper.get(result, target_face, source_face, paste_back=True)
        if intensity != 1.0:
            result = amplify_swap(original, result, intensity)
        previews.append(result)

    return previews


def build_grid(original: np.ndarray, previews: list[np.ndarray]) -> np.ndarray:
    """Arrange original + 5 previews in a 2x3 grid with labels."""
    panels = []
    labels = ["Original"] + [f"[{i+1}] {p} {i}" for i, p in enumerate(PRESETS)]
    labels = ["Original"] + [
        f"[{i+1}] {label} (p={passes}, i={intensity})"
        for i, (passes, intensity, label) in enumerate(PRESETS)
    ]
    images = [original] + previews

    for img, label in zip(images, labels):
        panel = img.copy()
        # Black bar at top for label
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 36), (0, 0, 0), -1)
        cv2.putText(panel, label, (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        panels.append(panel)

    # Resize all panels to same size (use the original's dimensions)
    h, w = panels[0].shape[:2]
    # Scale down for display — aim for ~600px per panel width
    scale = min(1.0, 600 / w)
    th, tw = int(h * scale), int(w * scale)
    panels = [cv2.resize(p, (tw, th)) for p in panels]

    # 2 rows x 3 columns
    row1 = np.hstack(panels[0:3])
    row2 = np.hstack(panels[3:6])
    grid = np.vstack([row1, row2])

    return grid


def main():
    parser = argparse.ArgumentParser(
        description="Preview face swap settings before full video run"
    )
    parser.add_argument("--source", required=True,
                        help="Path to source face image")
    parser.add_argument("--target", required=True,
                        help="Path to target video file")
    parser.add_argument("--target-index", type=int, default=0,
                        help="Which face to swap (left-to-right, 0-indexed). Default: 0")
    parser.add_argument("--frame", type=int, default=0,
                        help="Frame number to preview (default: 0)")
    parser.add_argument("--output-dir", default="output",
                        help="Directory for preview image (default: output/)")
    args = parser.parse_args()

    for path, label in [(args.source, "Source"), (args.target, "Target")]:
        if not Path(path).exists():
            print(f"Error: {label} not found: {path}", file=sys.stderr)
            sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading models...")
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    swapper = insightface.model_zoo.get_model(
        str(Path.home() / ".insightface/models/inswapper_128.onnx")
    )

    # Load source face
    src_img = cv2.imread(args.source)
    src_faces = app.get(src_img)
    if not src_faces:
        print("Error: No face detected in source image", file=sys.stderr)
        sys.exit(1)
    source_face = sorted(
        src_faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        reverse=True,
    )[0]

    # Extract target frame
    cap = cv2.VideoCapture(args.target)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.frame >= total_frames:
        print(f"Error: Frame {args.frame} out of range (video has {total_frames} frames)",
              file=sys.stderr)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Error: Could not read frame", file=sys.stderr)
        sys.exit(1)

    # Detect target face
    faces = sorted(app.get(frame), key=lambda f: f.bbox[0])
    if not faces:
        print("Error: No faces detected in target frame", file=sys.stderr)
        sys.exit(1)
    if args.target_index >= len(faces):
        print(f"Error: Target index {args.target_index} but only {len(faces)} face(s) found",
              file=sys.stderr)
        sys.exit(1)
    target_face = faces[args.target_index]

    # Generate previews
    print(f"Generating 5 previews from frame {args.frame}...")
    previews = render_previews(frame, source_face, target_face, swapper)

    # Build and save grid
    grid = build_grid(frame, previews)
    grid_path = str(output_dir / "preview_grid.png")
    cv2.imwrite(grid_path, grid)

    # Also save individual previews
    for i, (passes, intensity, label) in enumerate(PRESETS):
        cv2.imwrite(str(output_dir / f"preview_{i+1}_{label.lower()}.png"), previews[i])

    print(f"\nPreview grid saved to: {grid_path}")
    print(f"Individual previews saved to: {output_dir}/preview_*.png")
    print()
    print("Presets:")
    for i, (passes, intensity, label) in enumerate(PRESETS):
        print(f"  [{i+1}] {label:10s}  --passes {passes} --intensity {intensity}")
    print()

    choice = input("Pick a preset (1-5), or press Enter to skip: ").strip()
    if choice in ("1", "2", "3", "4", "5"):
        idx = int(choice) - 1
        passes, intensity, label = PRESETS[idx]
        print(f"\nSelected: {label}")
        print(f"Run the full swap with:")
        print(f"  python -m scripts.faceswap --source {args.source} --target {args.target} "
              f"--passes {passes} --intensity {intensity}")
    else:
        print("No selection made.")


if __name__ == "__main__":
    main()
