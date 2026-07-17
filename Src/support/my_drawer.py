import os

import cv2
import numpy as np

from my_dataclass import ROI
from my_logger    import LogLevel, Logger

# ==============================================================================
# Module responsible for drawing directly into frame based on the requirement
# ==============================================================================
class Drawer:
    def __init__(self, level: LogLevel = LogLevel.DEBUG):
        self.logger = Logger("Drawer", level)

    def draw(self, frame: np.ndarray, roi: ROI):
        self.logger.log_debug(f"Draw into frame with type: {roi.type}")

        # --- Segment ROI: polygon outline + label anchored at label_point ---
        if roi.type == "segment" or roi.type == "Segment":
            points = roi.roi_points.astype(np.int32) # OpenCV require int32 array in a list
            cv2.polylines(frame, [points], isClosed = True, color = roi.roi_color, thickness = 2)
            cv2.putText(frame, roi.label, roi.label_point, cv2.FONT_HERSHEY_SIMPLEX, 1.0, roi.label_color)

        # --- Box ROI: rectangle + label anchored at its own top-left corner ---
        elif roi.type == "box" or roi.type == "Box":
            x1,y1,x2,y2 = map (int,roi.roi_points) # Change np.array to map of tupple
            cv2.rectangle(frame, (x1,y1), (x2,y2),roi.roi_color, 2)
            cv2.putText(frame, roi.label, (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1.0, roi.label_color)

