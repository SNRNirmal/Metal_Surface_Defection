

import os
import cv2
import numpy as np
from typing import Dict, Optional, Tuple

from .postprocessing import (
    compute_defect_percentage,
    compute_per_class_area,
    classify_severity,
    compute_risk_score,
    build_analysis_report,
    CLASS_NAMES,
)
from .rul import estimate_remaining_life
from .visualization import (
    overlay_mask,
    generate_heatmap,
    draw_legend,
    draw_analysis_text,
    CLASS_COLORS_BGR,
)

# ── ImageNet normalisation constants ──────────────────────────────────────────
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT_SIZE: Tuple[int, int] = (128, 128)   # matches training resolution


class ONNXInferencePipeline:
    """
    Full defect detection + analysis pipeline using an ONNX model.

    Parameters
    ----------
    onnx_model_path : str
        Path to the exported .onnx model file.
    input_size : tuple of int
        (H, W) that the model was trained on. Default: (128, 128).
    threshold : float
        Binary threshold for each class channel (sigmoid output). Default: 0.5.
    min_defect_pixels : int
        Minimum connected-component size to keep. Smaller blobs are removed.
    """

    def __init__(
        self,
        onnx_model_path: str,
        input_size: Tuple[int, int] = INPUT_SIZE,
        threshold: float = 0.5,
        min_defect_pixels: int = 20,
    ):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required. Install with: pip install onnxruntime"
            )

        if not os.path.isfile(onnx_model_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )

        self.session          = ort.InferenceSession(onnx_model_path, providers=providers)
        self.input_name       = self.session.get_inputs()[0].name
        self.input_size       = input_size   # (H, W)
        self.threshold        = threshold
        self.min_defect_pixels = min_defect_pixels
        print(f"✅ ONNX session ready  |  providers: {self.session.get_providers()}")

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        image_path: str,
        save_dir: Optional[str] = None,
    ) -> Dict:
        """
        Run the full pipeline on a single image.

        Parameters
        ----------
        image_path : str
            Path to the input image (jpg / png).
        save_dir : str, optional
            If provided, overlay + heatmap images are saved here.

        Returns
        -------
        dict with keys:
            mask, report, rul, overlay_image, heatmap_image
        """
        # 1. Load & preprocess
        original_bgr, tensor = self._preprocess(image_path)

        # 2. ONNX inference  → raw logits  (1, C, H, W)
        logits = self.session.run(None, {self.input_name: tensor})[0]

        # 3. Decode to 2-D integer mask  (H_orig, W_orig)
        mask = self._decode_mask(logits, original_bgr.shape[:2])

        # 4. Analysis
        report = build_analysis_report(mask)
        rul    = estimate_remaining_life(
            defect_percent  = report["defect_percentage"],
            risk_score      = report["risk_score"],
            per_class_area  = report["per_class_area"],
        )

        # 5. Visualisation
        overlay_img = overlay_mask(original_bgr, mask)
        overlay_img = draw_legend(overlay_img, CLASS_NAMES)
        overlay_img = draw_analysis_text(
            overlay_img,
            report["defect_percentage"],
            report["severity"],
            report["risk_score"],
            rul["rul_days"],
        )
        heatmap_img = generate_heatmap(mask, original_bgr)

        # 6. Optionally save outputs
        if save_dir:
            self._save_outputs(image_path, save_dir, overlay_img, heatmap_img)

        return {
            "mask":          mask,
            "report":        report,
            "rul":           rul,
            "overlay_image": overlay_img,
            "heatmap_image": heatmap_img,
        }

    def run_from_array(
        self,
        image_bgr: np.ndarray,
    ) -> Dict:
        """
        Same as run() but accepts a BGR numpy array directly (for API / Streamlit use).
        """
        original_bgr = image_bgr.copy()
        tensor       = self._preprocess_array(image_bgr)
        logits       = self.session.run(None, {self.input_name: tensor})[0]
        mask         = self._decode_mask(logits, original_bgr.shape[:2])

        report = build_analysis_report(mask)
        rul    = estimate_remaining_life(
            defect_percent = report["defect_percentage"],
            risk_score     = report["risk_score"],
            per_class_area = report["per_class_area"],
        )

        overlay_img = overlay_mask(original_bgr, mask)
        overlay_img = draw_legend(overlay_img, CLASS_NAMES)
        overlay_img = draw_analysis_text(
            overlay_img,
            report["defect_percentage"],
            report["severity"],
            report["risk_score"],
            rul["rul_days"],
        )
        heatmap_img = generate_heatmap(mask, original_bgr)

        return {
            "mask":          mask,
            "report":        report,
            "rul":           rul,
            "overlay_image": overlay_img,
            "heatmap_image": heatmap_img,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _preprocess(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load image from disk and return (original_bgr, model_tensor)."""
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        return bgr, self._preprocess_array(bgr)

    def _preprocess_array(self, bgr: np.ndarray) -> np.ndarray:
        """
        Resize → normalise → NCHW float32 tensor.
        Handles both grayscale and colour images.
        """
        H, W = self.input_size

        # Grayscale → 3-channel BGR
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)

        resized = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_LINEAR)
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        norm    = (rgb - MEAN) / STD                         # (H, W, 3)
        tensor  = norm.transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, H, W)
        return tensor.astype(np.float32)

    def _decode_mask(
        self,
        logits: np.ndarray,
        original_hw: Tuple[int, int],
    ) -> np.ndarray:
        """
        Convert raw model output to a 2-D class-index mask.

        Strategy
        --------
        - Apply sigmoid channel-wise.
        - Each pixel is assigned the class with highest probability
          (argmax over channels), but only if that probability > threshold.
          If all channels are below threshold the pixel stays background (0).
        - Small connected components (< min_defect_pixels) are removed.
        - Resize back to the original image resolution.

        Parameters
        ----------
        logits : np.ndarray  shape (1, C, H, W)
        original_hw : (H_orig, W_orig)

        Returns
        -------
        np.ndarray  shape (H_orig, W_orig), dtype int32
        """
        # Sigmoid → probabilities  (C, H, W)
        probs = 1.0 / (1.0 + np.exp(-logits[0]))   # numerically stable sigmoid

        # Argmax over classes (+1 because index 0 = background, classes start at 1)
        pred_class = np.argmax(probs, axis=0) + 1   # (H, W)  values in [1, C]

        # Zero out pixels where max probability is below threshold
        max_prob   = probs.max(axis=0)               # (H, W)
        pred_class[max_prob < self.threshold] = 0    # → background

        # Remove tiny blobs
        pred_class = self._remove_small_components(pred_class.astype(np.uint8))

        # Resize to original resolution
        H_orig, W_orig = original_hw
        if pred_class.shape != (H_orig, W_orig):
            pred_class = cv2.resize(
                pred_class, (W_orig, H_orig),
                interpolation=cv2.INTER_NEAREST,
            )

        return pred_class.astype(np.int32)

    def _remove_small_components(self, mask: np.ndarray) -> np.ndarray:
        """Remove connected components smaller than min_defect_pixels."""
        if self.min_defect_pixels <= 0:
            return mask

        binary = (mask > 0).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

        cleaned = np.zeros_like(mask)
        for lbl in range(1, num_labels):
            if stats[lbl, cv2.CC_STAT_AREA] >= self.min_defect_pixels:
                cleaned[labels == lbl] = mask[labels == lbl]

        return cleaned

    @staticmethod
    def _save_outputs(
        image_path: str,
        save_dir: str,
        overlay_img: np.ndarray,
        heatmap_img: np.ndarray,
    ) -> None:
        os.makedirs(save_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(image_path))[0]
        cv2.imwrite(os.path.join(save_dir, f"{stem}_overlay.jpg"), overlay_img)
        cv2.imwrite(os.path.join(save_dir, f"{stem}_heatmap.jpg"), heatmap_img)
        print(f"💾 Outputs saved to {save_dir}")