import os
import cv2
import numpy as np
from dataclasses import dataclass
from my_logger      import LogLevel,Logger

"""
Module responsible for drawing directly into frame based on the requirement

ROI ( Range of intersting)
    Type: The type of ROI: "Box", "Segment"
    Points: Points corresponding to type that surrond the ROI
"""

@dataclass
class ROI:
    Type: str
    Roi_points: np.array
    Roi_color: tuple
    Label: str
    Lable_point: tuple
    Lable_color: tuple
class Drawer:
    def __init__(self,log_level: LogLevel = LogLevel.DEBUG):
        self.logger = Logger("Drawer",log_level)

    def draw(self, frame: np.array, roi: ROI):
        self.logger.log_debug(f"Draw into frame with type: {roi.Type}")
        if roi.Type == "segment":
            points = roi.Roi_points.astype(np.int32) # OpenCV require int32 array in a list
            cv2.polylines(frame, [points], isClosed = True, color = roi.Roi_color, thickness = 2)
            cv2.putText(frame, roi.Label, roi.Lable_point, cv2.FONT_HERSHEY_SIMPLEX, 1.0, roi.Lable_color)
        elif roi.Type == "box":
            x1,y1,x2,y2 = map (int,roi.Roi_points) # Change np.array to map of tupple
            cv2.rectangle(frame, (x1,y1), (x2,y2),roi.Roi_color, 2)
            cv2.putText(frame, roi.Label, roi.Lable_point, cv2.FONT_HERSHEY_SIMPLEX, 1.0, roi.Lable_color)

