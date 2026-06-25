import pickle
from dataclasses import dataclass
from pathlib import Path
from my_logger import Logger, LogLevel
import cv2
import numpy as np

COLOR_RANGES = [[100, 120], [165, 180]]
MODULE_DIR = Path(__file__).resolve().parent

# Hardcoded asset paths relative to this module
INTERNAL_MODEL_PATH = (MODULE_DIR / ".." / ".." / "Models" / "dt_color_model.pkl").resolve()
# DEFAULT_SCALER_PATH = (MODULE_DIR / "Input" / "hsv_scaler.pkl").resolve()


@dataclass
class ColorPrediction:
    label: str
    confidence: float


class ShirtColorClassifier:

    def __init__(
        self,
        model_path: str = str(INTERNAL_MODEL_PATH),
        # scaler_path: str = str(DEFAULT_SCALER_PATH),
        pixel_sample_size: int = 500,
        log_level: LogLevel = LogLevel.DEBUG,
        confidence: float = 0.8
    ):
        """Initializes the ShirtColorClassifier pipeline with saved models."""
        self.pixel_sample_size = pixel_sample_size
        self.logger = Logger("Color classification",log_level)
        self.confidence = confidence
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        # with open(scaler_path, "rb") as f:
        #     self.scaler = pickle.load(f)

    def predict_color(self, masked_frame) -> ColorPrediction | None:
        """Processes a single masked frame, filters out the black background,

        and predicts the dominant shirt color using the loaded KNN model.
        """
        if masked_frame is None:
            return None

        # Convert image to HSV space
        hsv_img = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)

        # Generate mask to completely ignore the black background (V > 0 and S > 0)
        non_black_mask = cv2.inRange(
            hsv_img, np.array([0, 1, 1]), np.array([179, 255, 255])
        )

        # Get spatial coordinates of all non-black pixels
        pixel_coords = np.argwhere(non_black_mask > 0)
        total_valid_pixels = len(pixel_coords)
        if len(pixel_coords) == 0:
            return ColorPrediction(label="unknown", confidence=0.0)

        # Extract [H, S, V] values from valid positions
        hsv_pixels = hsv_img[pixel_coords[:, 0], pixel_coords[:, 1]]
        # 1. Find the mode pixel value (H, S, V) independently for each channel
        h_channel = hsv_pixels[:, 0]
        s_channel = hsv_pixels[:, 1]
        v_channel = hsv_pixels[:, 2]
        hist_h = np.bincount(h_channel, minlength=180)
        hist_s = np.bincount(s_channel, minlength=256)
        hist_v = np.bincount(v_channel, minlength=256)

        mode_h = int(np.argmax(hist_h))
        mode_s = int(np.argmax(hist_s))
        mode_v = int(np.argmax(hist_v))

        # 4. Threshold check
       # if percentage < 70.0:
        #     return ColorPrediction(label="unknown", confidence=0.0) 
        # Inference utilizing KNN model
        # 5. Predict using only the single Mode (H, S, V) pixel value
        mode_pixel = np.array([[mode_h, mode_s, mode_v]])
        # Check if any new color appear
        is_interesting = any(lower <= mode_h <= upper for lower, upper in COLOR_RANGES)
        if not is_interesting:
            self.logger.log_debug("Not intesting")
            return None
        # Probability of each class
        probabilities  = self.model.predict_proba(mode_pixel)
        # 2. Find the highest probability value
        max_confidence = np.max(probabilities)
        # Highest indexx
        dominant_index = np.argmax(probabilities )
        dominant_label = str(self.model.classes_[dominant_index])
        if max_confidence < self.confidence:
            return None
        return ColorPrediction(label=dominant_label, confidence=float(max_confidence))