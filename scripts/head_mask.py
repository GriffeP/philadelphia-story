"""
Head mask generation using BiSeNet face parsing.

Creates an expanded head mask that covers face, hair, ears, neck, and jaw
for seamless full-head swapping (vs. the tight face-only mask from inswapper).

BiSeNet label indices used for the head mask:
    1: skin       2: nose      4: l_eye    5: r_eye
    6: l_brow     7: r_brow    8: l_ear    9: r_ear
   10: mouth     11: u_lip    12: l_lip   13: hair
   17: neck
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

from scripts.bisenet import BiSeNet

# Indices that define the full head region
HEAD_LABELS = {1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17}

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "79999_iter.pth"


class HeadMaskGenerator:
    """Generates full-head segmentation masks using BiSeNet."""

    def __init__(self, model_path: str | Path | None = None, device: str = "cpu"):
        model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if not model_path.exists():
            raise RuntimeError(
                f"BiSeNet model not found at {model_path}. Download with:\n"
                "  curl -L -o models/79999_iter.pth "
                '"https://huggingface.co/ManyOtherFunctions/face-parse-bisent/resolve/main/79999_iter.pth"'
            )

        self.device = torch.device(device)
        self.net = BiSeNet(n_classes=19)
        state = torch.load(str(model_path), map_location=self.device, weights_only=True)
        self.net.load_state_dict(state, strict=False)
        self.net.to(self.device)
        self.net.eval()

    @torch.no_grad()
    def parse(self, bgr_image: np.ndarray) -> np.ndarray:
        """Run face parsing on a BGR image.

        Returns a (H, W) uint8 array where each pixel is a label index 0-18.
        """
        h, w = bgr_image.shape[:2]

        # Preprocess: BGR->RGB, resize to 512x512, normalize
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        tensor = normalize(tensor, mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
        tensor = tensor.unsqueeze(0).to(self.device)

        out, _, _ = self.net(tensor)
        parsing = out.squeeze(0).argmax(0).cpu().numpy().astype(np.uint8)

        # Resize back to original dimensions
        if (h, w) != (512, 512):
            parsing = cv2.resize(parsing, (w, h), interpolation=cv2.INTER_NEAREST)

        return parsing

    def head_mask(
        self,
        bgr_image: np.ndarray,
        face_bbox: tuple[float, float, float, float] | None = None,
        labels: set[int] | None = None,
        feather_radius: int = 11,
        dilate_px: int = 10,
        bbox_expand: float = 1.5,
    ) -> np.ndarray:
        """Generate a soft full-head mask for blending.

        Args:
            bgr_image: Input BGR frame.
            face_bbox: (x1, y1, x2, y2) bounding box of the target face.
                       If provided, only the head region overlapping this
                       expanded bbox is included — prevents masking other
                       faces in the frame.
            labels: Set of label indices to include. Defaults to HEAD_LABELS.
            feather_radius: Gaussian blur kernel size for edge softening.
                            Must be odd. 0 disables feathering.
            dilate_px: Pixels to dilate the mask before feathering, to push
                       the blend boundary outward past segmentation edges.
            bbox_expand: Factor to expand face_bbox to capture hair/neck
                         beyond the tight face box. Default 1.5 (50% larger).

        Returns:
            Float32 mask (H, W) in [0, 1] — 1.0 inside the head region.
        """
        labels = labels if labels is not None else HEAD_LABELS
        parsing = self.parse(bgr_image)

        # Build binary mask from selected labels
        mask = np.zeros(parsing.shape, dtype=np.uint8)
        for label in labels:
            mask[parsing == label] = 255

        # Restrict to the target face region if bbox provided
        if face_bbox is not None:
            h, w = mask.shape
            x1, y1, x2, y2 = face_bbox
            bw, bh = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            # Expand bbox to capture hair and neck
            ex1 = max(0, int(cx - bw * bbox_expand))
            ey1 = max(0, int(cy - bh * bbox_expand))
            ex2 = min(w, int(cx + bw * bbox_expand))
            ey2 = min(h, int(cy + bh * bbox_expand))
            # Zero out everything outside the expanded bbox
            region_mask = np.zeros_like(mask)
            region_mask[ey1:ey2, ex1:ex2] = 255
            mask = cv2.bitwise_and(mask, region_mask)

        # Fill small holes inside the mask (e.g. missed pixels between
        # hair strands) using morphological closing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Dilate to expand past segmentation boundaries
        if dilate_px > 0:
            dilate_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1)
            )
            mask = cv2.dilate(mask, dilate_kernel)

        # Feather edges for smooth blending
        if feather_radius > 0:
            ksize = feather_radius if feather_radius % 2 == 1 else feather_radius + 1
            mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)

        return mask.astype(np.float32) / 255.0


def blend_with_head_mask(
    original: np.ndarray,
    swapped: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Alpha-blend swapped face onto original using the head mask.

    Args:
        original: Original BGR frame.
        swapped: Frame with face already swapped by inswapper (paste_back=True).
        mask: Float32 (H, W) mask in [0, 1] from HeadMaskGenerator.head_mask().

    Returns:
        Blended BGR frame.
    """
    mask_3ch = mask[:, :, np.newaxis]  # (H, W, 1) for broadcasting
    blended = (swapped.astype(np.float32) * mask_3ch
               + original.astype(np.float32) * (1.0 - mask_3ch))
    return np.clip(blended, 0, 255).astype(np.uint8)
