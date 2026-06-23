from dataclasses import dataclass
from typing import List, Tuple
import cv2
import numpy as np
from my_logger import LogLevel, log

# Type alias for HSV bounds: (Hue, Saturation, Value)
HSVColor = Tuple[int, int, int]
MAX_INVALID_S_RATIO = 0.70  # 70% threshold

@dataclass(frozen=True)
class UniformType:
    """Represents a unique uniform type config with its HSV boundaries."""

    name: str
    # Boundaries focus primarily on Hue (H) and Value (V)
    lower_boundary: HSVColor
    upper_boundary: HSVColor

    def get_matching_mask(self, hsv_image: np.ndarray) -> np.ndarray:
        """Segments the HSV image using the boundaries and returns the binary mask."""
        lower = np.array(self.lower_boundary, dtype=np.uint8)
        upper = np.array(self.upper_boundary, dtype=np.uint8)
        
        # Color segmentation based on H and V configurations
        color_mask = cv2.inRange(hsv_image, lower, upper)
        return color_mask


class UniformClassifier:
    """Manages uniform types and evaluates matches to calculate the highest proportion."""

    def __init__(self, uniforms: List[UniformType], s_min_threshold: int = 50):
        self.uniforms = uniforms
        self.s_min_threshold = s_min_threshold  # Minimum saturation to consider a valid color pixel

    def predict(
            self, hsv_image: np.ndarray, torso_mask: np.ndarray
        ) -> Tuple[str, float]:
            """Calculates the proportion of matching pixels for all registered uniforms

            relative to the valid torso region after applying Saturation filtering.
            """
            # Force torso_mask to be a strict binary mask (0 or 255) to avoid overflow in bitwise operations
            binary_torso = (torso_mask > 0).astype(np.uint8) * 255
            initial_torso_pixels = cv2.countNonZero(binary_torso)
            # Extract the Saturation (S) channel from the source HSV image
            s_channel = hsv_image[:, :, 1]
            
            # Create a conditional mask: keep only pixels with sufficient color saturation
            valid_s_mask = (s_channel >= self.s_min_threshold).astype(np.uint8) * 255

            # Calculate the total number of torso pixels that survive the Saturation filter
            filtered_torso_mask = cv2.bitwise_and(torso_mask, valid_s_mask)
            total_valid_torso_pixels = cv2.countNonZero(filtered_torso_mask)


            # CONDITION: If the low-saturation area (grayscale/white/black/glare) is too large,
            # we reject the prediction because the color information is no longer reliable.
            invalid_s_pixels = initial_torso_pixels - total_valid_torso_pixels
            invalid_s_ratio = invalid_s_pixels / initial_torso_pixels
            log(LogLevel.DEBUG,f"Invalid saturation {invalid_s_ratio}")
            if invalid_s_ratio > MAX_INVALID_S_RATIO:
                log(LogLevel.INFO, f"Prediction rejected: Low saturation area is too large ({invalid_s_ratio:.2f})")
                return "Unknown", 0.0
           
            best_name = "Unknown"
            max_pixels = 0

            for uniform in self.uniforms:
                # Get the color segmentation mask for the current uniform
                uniform_mask = uniform.get_matching_mask(hsv_image)
                
                # Apply Saturation as a conditional filter using bitwise AND operation
                final_match_mask = cv2.bitwise_and(uniform_mask, valid_s_mask)
                
                # Count the remaining non-zero pixels after filtering
                match_pixels = cv2.countNonZero(final_match_mask)
                
                if match_pixels > max_pixels:
                    max_pixels = match_pixels
                    best_name = uniform.name

            # Both max_pixels and total_valid_torso_pixels are now standard integers (scalars)
            proportion = max_pixels / total_valid_torso_pixels
            return best_name, proportion


# --- Data Registry Definitions ---
# Centralized definition of all known uniforms. 
# Saturation (S) boundaries are set to 0 and 255 respectively, 
# ensuring that color filtering is managed globally by s_min_threshold inside the predict method.
UNIFORM_REGISTRY = [
    UniformType(
        name="Blue Officer",
        lower_boundary=(104, 0, 50),
        upper_boundary=(110, 255, 255),
    ),
    UniformType(
        name="Heavy Blue Officer",
        lower_boundary=(108, 0, 0),
        upper_boundary=(116, 255, 130),
    ),
    UniformType(
        name="Red Officer",
        lower_boundary=(170, 0, 215),
        upper_boundary=(178, 255, 255),
    ),
]

# Instantiate a default global classifier with an S threshold of 50
detector = UniformClassifier(UNIFORM_REGISTRY, s_min_threshold=50)