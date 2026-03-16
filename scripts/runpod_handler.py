#!/usr/bin/env python3
"""
RunPod serverless handler for face swap.

Receives a job with base64-encoded source image and target video,
runs the face swap on GPU, and returns the result as base64 video.

Job input format:
{
    "source_image": "<base64 encoded png/jpg>",
    "target_video": "<base64 encoded video>",
    "target_index": 0       # optional, default 0 (-1 for all faces)
}

Returns:
{
    "video": "<base64 encoded mp4>",
    "frame_count": int,
    "faces_found": int
}
"""

import base64
import tempfile
from pathlib import Path

import cv2
import numpy as np
import runpod
from insightface.app import FaceAnalysis
import insightface

# Initialize models once at container startup
print("Loading models...")
APP = FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
APP.prepare(ctx_id=0, det_size=(640, 640))

SWAPPER = insightface.model_zoo.get_model(
    str(Path.home() / ".insightface/models/inswapper_128.onnx"),
)
print("Models loaded.")


def decode_to_tempfile(data_b64: str, suffix: str) -> str:
    """Write base64 data to a temp file and return the path."""
    raw = base64.b64decode(data_b64)
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(raw)
    tmp.close()
    return tmp.name


def handler(job: dict) -> dict:
    """RunPod serverless handler entry point."""
    inp = job["input"]

    target_index = inp.get("target_index", 0)

    # Decode inputs to temp files
    source_path = decode_to_tempfile(inp["source_image"], ".png")
    target_path = decode_to_tempfile(inp["target_video"], ".mp4")

    try:
        # Load source face
        source_img = cv2.imread(source_path)
        source_faces = APP.get(source_img)
        if not source_faces:
            return {"error": "No face detected in source image"}
        source_face = sorted(
            source_faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True,
        )[0]

        # Process target video
        cap = cv2.VideoCapture(target_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        swapped_frames = []
        total_faces = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = frame.copy()
            faces = APP.get(frame)
            total_faces += len(faces)

            if faces:
                faces = sorted(faces, key=lambda f: f.bbox[0])
                if target_index == -1:
                    for face in faces:
                        result = SWAPPER.get(result, face, source_face, paste_back=True)
                elif target_index < len(faces):
                    result = SWAPPER.get(result, faces[target_index], source_face, paste_back=True)

            swapped_frames.append(result)
        cap.release()

        # Write output video
        out_path = tempfile.mktemp(suffix=".mp4")
        raw_path = tempfile.mktemp(suffix=".raw")

        with open(raw_path, "wb") as f:
            for frame in swapped_frames:
                f.write(frame.tobytes())

        import subprocess
        # Mux with original audio
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}", "-r", str(fps),
            "-i", raw_path,
            "-i", target_path,
            "-map", "0:v", "-map", "1:a?",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "copy", "-pix_fmt", "yuv420p",
            out_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        with open(out_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode()

        # Cleanup
        for p in [source_path, target_path, raw_path, out_path]:
            Path(p).unlink(missing_ok=True)

        return {
            "video": video_b64,
            "frame_count": len(swapped_frames),
            "faces_found": total_faces,
        }

    except Exception as e:
        # Cleanup on error
        for p in [source_path, target_path]:
            Path(p).unlink(missing_ok=True)
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
