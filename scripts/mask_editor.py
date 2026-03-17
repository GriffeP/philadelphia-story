#!/usr/bin/env python3
"""
Interactive mask editor for defining the head swap boundary.

Displays a video frame with facial landmarks visible. The user clicks points
to draw a polygon outline around the head region they want swapped. The
outline is stored relative to the 106-point facial landmarks, so it
automatically tracks face movement, rotation, and scale across all frames.

Usage:
    python -m scripts.mask_editor --source input/face.jpg --target input/yar_scene.mov
    python -m scripts.mask_editor --source input/face.jpg --target input/yar_scene.mov --frame 50

Controls:
    Left-click     Add a point to the polygon
    Right-click    Undo last point
    Enter          Close polygon and save profile
    Escape         Cancel without saving
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("macosx")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

from insightface.app import FaceAnalysis


def compute_landmark_weights(
    point: tuple[float, float],
    landmarks: np.ndarray,
    n_anchors: int = 3,
) -> list[dict]:
    """Express a point as a weighted combination of its nearest landmarks.

    Uses barycentric-style coordinates relative to the 3 nearest landmarks.
    This works even if the point is outside the triangle (extrapolation).

    Returns a list of {index, weight} dicts.
    """
    px, py = point
    dists = np.sqrt((landmarks[:, 0] - px) ** 2 + (landmarks[:, 1] - py) ** 2)
    nearest_idx = np.argsort(dists)[:n_anchors]
    anchors = landmarks[nearest_idx]  # (3, 2)

    # Solve for weights: point = w0*a0 + w1*a1 + w2*a2, with w0+w1+w2=1
    # Rewrite as: point - a2 = w0*(a0-a2) + w1*(a1-a2)
    A = anchors[:2] - anchors[2]  # (2, 2) — rows are (a0-a2), (a1-a2)
    b = np.array([px, py]) - anchors[2]  # (2,)

    try:
        # A.T because we need columns to be the basis vectors
        w01 = np.linalg.solve(A.T, b)
        weights = [float(w01[0]), float(w01[1]), float(1.0 - w01[0] - w01[1])]
    except np.linalg.LinAlgError:
        # Degenerate case: fall back to inverse-distance weighting
        inv_dists = 1.0 / (dists[nearest_idx] + 1e-8)
        weights = (inv_dists / inv_dists.sum()).tolist()

    return [
        {"landmark_index": int(idx), "weight": w}
        for idx, w in zip(nearest_idx, weights)
    ]


def reconstruct_point(
    anchor_weights: list[dict],
    landmarks: np.ndarray,
) -> tuple[float, float]:
    """Reconstruct a point from landmark weights on a new frame."""
    x, y = 0.0, 0.0
    for aw in anchor_weights:
        lm = landmarks[aw["landmark_index"]]
        x += lm[0] * aw["weight"]
        y += lm[1] * aw["weight"]
    return (x, y)


def reconstruct_polygon(
    profile: list[list[dict]],
    landmarks: np.ndarray,
) -> list[tuple[float, float]]:
    """Reconstruct all polygon points from a saved profile."""
    return [reconstruct_point(aw, landmarks) for aw in profile]


def profile_to_mask(
    profile: list[list[dict]],
    landmarks: np.ndarray,
    shape: tuple[int, int],
    feather_radius: int = 21,
    dilate_px: int = 8,
) -> np.ndarray:
    """Generate a soft mask from a landmark-relative profile.

    Args:
        profile: List of anchor weight dicts (from save_profile).
        landmarks: (106, 2) landmarks for the current frame.
        shape: (height, width) of the output mask.
        feather_radius: Gaussian blur kernel for edge softening.
        dilate_px: Dilation before feathering.

    Returns:
        Float32 mask (H, W) in [0, 1].
    """
    points = reconstruct_polygon(profile, landmarks)
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    if dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1)
        )
        mask = cv2.dilate(mask, kernel)

    if feather_radius > 0:
        ksize = feather_radius if feather_radius % 2 == 1 else feather_radius + 1
        mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)

    return mask.astype(np.float32) / 255.0


class MaskEditor:
    """Interactive matplotlib-based polygon editor."""

    def __init__(self, frame_bgr: np.ndarray, landmarks_106: np.ndarray,
                 face_bbox: np.ndarray):
        self.frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.landmarks = landmarks_106
        self.bbox = face_bbox
        self.points = []
        self.closed = False
        self.cancelled = False

        self.fig, self.ax = plt.subplots(1, 1, figsize=(14, 10))
        self.ax.imshow(self.frame_rgb)
        self.ax.set_title(
            "Left-click: add point | Right-click: undo | Enter: finish | Esc: cancel",
            fontsize=12,
        )

        # Draw landmarks as small cyan dots
        self.ax.scatter(
            landmarks_106[:, 0], landmarks_106[:, 1],
            c="cyan", s=8, zorder=5, alpha=0.7,
        )

        # Draw face bbox
        x1, y1, x2, y2 = face_bbox
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                              linewidth=1, edgecolor="yellow",
                              facecolor="none", linestyle="--")
        self.ax.add_patch(rect)

        # Polygon line (dotted green)
        self.line, = self.ax.plot([], [], "g--o", linewidth=2,
                                  markersize=8, markerfacecolor="lime",
                                  zorder=10)
        self.close_line, = self.ax.plot([], [], "g--", linewidth=1,
                                        alpha=0.5, zorder=9)

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        plt.tight_layout()

    def run(self) -> list[tuple[float, float]] | None:
        """Show the editor and return the polygon points, or None if cancelled."""
        plt.show()
        if self.cancelled or len(self.points) < 3:
            return None
        return self.points

    def _on_click(self, event):
        if event.inaxes != self.ax:
            return

        if event.button == 1:  # Left click — add point
            self.points.append((event.xdata, event.ydata))
        elif event.button == 3:  # Right click — undo
            if self.points:
                self.points.pop()

        self._redraw()

    def _on_key(self, event):
        if event.key == "enter":
            if len(self.points) >= 3:
                self.closed = True
                plt.close(self.fig)
        elif event.key == "escape":
            self.cancelled = True
            plt.close(self.fig)
        elif event.key == "z":
            if self.points:
                self.points.pop()
                self._redraw()

    def _redraw(self):
        if self.points:
            xs = [p[0] for p in self.points]
            ys = [p[1] for p in self.points]
            self.line.set_data(xs, ys)
            # Show closing line back to first point
            if len(self.points) >= 3:
                self.close_line.set_data(
                    [xs[-1], xs[0]], [ys[-1], ys[0]]
                )
            else:
                self.close_line.set_data([], [])
        else:
            self.line.set_data([], [])
            self.close_line.set_data([], [])

        self.fig.canvas.draw_idle()


def save_profile(
    polygon_points: list[tuple[float, float]],
    landmarks: np.ndarray,
    output_path: str,
    metadata: dict | None = None,
):
    """Convert polygon points to landmark-relative weights and save."""
    profile = []
    for pt in polygon_points:
        weights = compute_landmark_weights(pt, landmarks)
        profile.append(weights)

    data = {
        "version": 1,
        "n_points": len(profile),
        "profile": profile,
    }
    if metadata:
        data["metadata"] = metadata

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def load_profile(profile_path: str) -> list[list[dict]]:
    """Load a saved mask profile."""
    with open(profile_path) as f:
        data = json.load(f)
    return data["profile"]


def main():
    parser = argparse.ArgumentParser(
        description="Interactive head mask editor"
    )
    parser.add_argument("--target", required=True,
                        help="Path to target video file")
    parser.add_argument("--region", default="head",
                        choices=["head", "hair"],
                        help="Which region to outline: 'head' (full head for face swap) "
                             "or 'hair' (hair only for hair replacement). Default: head")
    parser.add_argument("--target-index", type=int, default=0,
                        help="Which face to outline (left-to-right, 0-indexed)")
    parser.add_argument("--frame", type=int, default=0,
                        help="Frame number to edit on (default: 0)")
    parser.add_argument("--output", default=None,
                        help="Where to save the mask profile. "
                             "Defaults to output/<region>_profile.json")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"output/{args.region}_profile.json"

    if not Path(args.target).exists():
        print(f"Error: Target not found: {args.target}", file=sys.stderr)
        sys.exit(1)

    print("Loading face analysis model...")
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Extract frame
    cap = cv2.VideoCapture(args.target)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.frame >= total_frames:
        print(f"Error: Frame {args.frame} out of range ({total_frames} frames)",
              file=sys.stderr)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    cap.release()

    # Detect faces
    faces = sorted(app.get(frame), key=lambda f: f.bbox[0])
    if not faces:
        print("Error: No faces detected in this frame", file=sys.stderr)
        sys.exit(1)
    if args.target_index >= len(faces):
        print(f"Error: Target index {args.target_index} but only {len(faces)} face(s)",
              file=sys.stderr)
        sys.exit(1)

    face = faces[args.target_index]
    landmarks = face.landmark_2d_106
    print(f"Face detected: bbox={face.bbox.astype(int).tolist()}, "
          f"106 landmarks loaded")

    # Launch editor
    region_label = "HEAD (face + hair + ears + neck)" if args.region == "head" else "HAIR only"
    print(f"Opening mask editor — draw the {region_label} boundary...")
    print("  Left-click to place points")
    print("  Right-click or Z to undo")
    print("  Enter to close polygon and save")
    print("  Escape to cancel")

    editor = MaskEditor(frame, landmarks, face.bbox)
    editor.ax.set_title(
        f"Drawing: {region_label} | Left-click: add | Right-click: undo | Enter: finish",
        fontsize=12,
    )
    polygon = editor.run()

    if polygon is None:
        print("Cancelled — no profile saved.")
        sys.exit(0)

    print(f"Polygon drawn with {len(polygon)} points")

    # Save profile
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_profile(
        polygon, landmarks, args.output,
        metadata={
            "target_video": args.target,
            "frame": args.frame,
            "target_index": args.target_index,
            "n_landmarks": 106,
        },
    )
    print(f"Mask profile saved to: {args.output}")

    # Show preview of the mask
    mask = profile_to_mask(polygon_to_profile(polygon, landmarks),
                           landmarks, frame.shape[:2])
    mask_vis = (mask * 255).astype(np.uint8)
    preview_path = str(Path(args.output).with_suffix(".png"))
    cv2.imwrite(preview_path, mask_vis)
    print(f"Mask preview saved to: {preview_path}")


def polygon_to_profile(
    polygon: list[tuple[float, float]],
    landmarks: np.ndarray,
) -> list[list[dict]]:
    """Convert raw polygon points to landmark-relative profile."""
    return [compute_landmark_weights(pt, landmarks) for pt in polygon]


if __name__ == "__main__":
    main()
