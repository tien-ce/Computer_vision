import numpy as np
from enum import Enum
from dataclasses import dataclass

# ==============================================================================
# ROI ( Range of interesting )
#   Type: The type of ROI: "Box", "Segment"
#   Points: Points corresponding to type that surrond the ROI
#   Lable_point: Where to anchor the label text. Required for "Segment" (no x1,y1 to derive it
#   from); "Box" ignores it and derives the anchor from its own points instead.
# ==============================================================================
@dataclass
class ROI:
    type: str
    roi_points: np.ndarray
    roi_color: tuple
    label: str
    label_color: tuple
    label_point: tuple = None

# ==============================================================================
# BGR (BLUE, GREEN, RED) tuple
# ==============================================================================
class BGRColor(Enum):
    # Primary Colors
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)

    # Secondary Colors
    YELLOW = (0, 255, 255)
    CYAN = (255, 255, 0)
    MAGENTA = (255, 0, 255)

    # Neutrals
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GRAY = (128, 128, 128)

# ==============================================================================
# Bounding Box
# ==============================================================================
@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    class_id: int
    confidence: float
    track_id: int
    @property
    def np_points(self) -> np.array:
        """Returns the coordinates boundary as a numpy array."""
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.int32)
    @property
    def xyxy(self):
        return self.x1,self.y1,self.x2,self.y2

# ==============================================================================
# Segmentation result
# ==============================================================================
@dataclass
class SegmentedObject:
    points: np.ndarray  # Format: Shape (N, 1, 2) dtype=np.int32 for OpenCV compatibility
    class_id: int
    confidence: float
    masked_frame: np.array
    
# ==============================================================================
# Video Source Info
# ==============================================================================
@dataclass
class SRC_INFO:
    src_type: str
    frame_width: int
    frame_height: int
    fps: float
    total_frames: int

