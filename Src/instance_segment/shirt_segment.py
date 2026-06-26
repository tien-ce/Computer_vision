# shirt_segment.py
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
import torch
# Import standard components from your centralized logger module
from my_logger import Logger, LogLevel
from my_inference import ModelInference

# Resolve the absolute directory of this module file at runtime
MODULE_DIR = Path(__file__).resolve().parent

# Hardcoded relative path layout from THIS module's location
INTERNAL_MODEL_PATH = (MODULE_DIR / ".." / ".." / "Models" / "uniform_yolov11-seg_best.pt").resolve()

# ==============================================================================
# DATA STRUCTURE
# ==============================================================================
@dataclass
class SegmentedObject:
    points: np.ndarray  # Format: Shape (N, 1, 2) dtype=np.int32 for OpenCV compatibility
    class_id: int
    confidence: float
    masked_frame: np.array    
TARGET_TOP_CLASSES = {1,2,4,5,8,11}
# ==============================================================================
# SEGMENTER MODULE CLASS
# ==============================================================================
class ShirtSegmenter:
    def __init__(self, log_level: LogLevel = LogLevel.INFO,confidence: float = 0.5, draw: bool = False):
        """
        Initializes the ShirtSegmenter module. The model path is resolved automatically
        internally and does not require external parameters.
        
        :param log_level: Desired LogLevel for this individual module instance
        :param draw: If True, draws and fills the segmentation mask over the frame matrix
        """
        # Pass the automatically resolved absolute path string to your inference engine
        self.shirt_seg_model = ModelInference(str(INTERNAL_MODEL_PATH))
        self.confidence = confidence
        self.draw = draw
        
        # The module instantiates its own isolated logger and sets its own level
        self.logger = Logger(name="ShirtSegmenter")
        self.logger.setLevel(log_level)
        
        self.logger.log_info(f"Model loaded successfully from {INTERNAL_MODEL_PATH} (Log Level: {log_level.name})")

    def _segment_shirt(self, frame) -> SegmentedObject:
        """
        Processes a single cropped person frame matrix, runs instance segmentation, and extracts contours.
        
        :param frame: The isolated image frame matrix (numpy array)
        :return: A list of SegmentedObject instances containing polygon contours and metadata
        """
        if frame is None:
            self.logger.log_error("Received an empty or invalid cropped frame matrix")
            return None

        # Execute model inference on secondary isolated GPU to prevent resource collisions
        result = self.shirt_seg_model.predict(frame=frame,verbose=False)
        
        # Structural boundary validation: Ensure that the model successfully isolated at least one mask object
        if result.masks is None:
            self.logger.log_debug("No segments or masks detected in the current frame")
            return None

        # ---------------- Original internal debug section ----------------
        raw_masks_tensor = result.masks.data
        # Type: <class 'torch.Tensor'>, Example shape: torch.Size([num_objects, H, W]) on device='cuda:1'
        self.logger.log_debug(f"raw_masks_tensor: {type(raw_masks_tensor)} (3D Shape: {raw_masks_tensor.shape})")

        polygons_pixel = result.masks.xy
        # Type: <class 'list'> containing numpy arrays of shape (N, 2) with float32 pixel coordinates
        self.logger.log_debug(f"polygons_pixel: {type(polygons_pixel)} Number of segments: {len(polygons_pixel)}")
        # ---------------------------------------------------------------------

        segmented_objects = []

        # Iterate concurrently over segments, bounding box metadata, and classification attributes
        for i, (mask, box) in enumerate(zip(polygons_pixel, result.boxes)):
            if len(mask) == 0:
                continue

            class_id = int(box.cls[0])
            # Filter specifically for targeting garments / shirts (Adjust class_id map matching your data.yaml)
            if class_id in TARGET_TOP_CLASSES:
                confidence = float(box.conf[0])
                self.logger.log_debug(f"class_id: {class_id}, confidence: {confidence}")
                if confidence < self.confidence:
                    self.logger.log_debug("Confidence isn't enough")
                    continue
                # OpenCV primitives require signed 32-bit integer arrays
                # points_float32:   [[120.4, 240.1], [130.8, 250.5], ...]
                #                        ↓               ↓
                #                    (Casting process via numpy vectorization)
                #                        ↓               ↓
                # points_int32:     [[120,   240  ], [130,   250  ], ...]
                points = np.array(mask, dtype=np.int32)
                
                # Reshape array topology to fit OpenCV contour specifications: (N, 1, 2)
                points = points.reshape((-1, 1, 2))
                self.logger.log_debug(f"Object {i} contour structure - Type: {type(points)} Shape: {points.shape}")

                seg_obj = SegmentedObject(points=points, class_id=class_id, confidence=confidence, masked_frame = None)
                segmented_objects.append(seg_obj)

                # Context-aware drawing and Alpha blending inside the module
                if self.draw is True:
                    # Draw structural outer polygon wireframe boundaries
                    cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)
                    
                    # Generate an independent mask layout to handle Alpha blending safely
                    overlay = frame.copy()
                    cv2.fillPoly(overlay, [points], color=(0, 255, 0))
                    
                    # Execute localized linear pixel mixing: 40% opaque overlay blending
                    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, dst=frame)
        return segmented_objects# Belive in our model only detect one object
    
    def mask_shirt (self,frame):
        shirt_segments = self._segment_shirt(frame)
        # Might be detect more than 1 segemtn
        if len(shirt_segments) == 0: 
            self.logger.log_debug("mask_shirt received None from segment_shirt (No garments detected)")
            return shirt_segments
        for shirt_segment in shirt_segments:
            points = shirt_segment.points
            # 1. Initialize a strictly 8-bit unsigned integer single-channel blank mask (uint8)
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            # 2. Render and solid-fill the inner polygon coordinates on the mask with pure white (255)
            cv2.fillPoly(mask, [points], 255)
            # 3. Apply bitwise conjunction. Pixels corresponding to 255 on the mask are retained in color.
            shirt_segment.masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        return shirt_segments 

