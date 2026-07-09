from dataclasses import dataclass
from pathlib import Path
import cv2
import torch
import numpy as np
# Import standard components from your centralized logger module
from my_logger import Logger, LogLevel
from my_inference import ModelInference

# Resolve the absolute directory of this module file at runtime
MODULE_DIR = Path(__file__).resolve().parent

# Hardcoded relative path layout from THIS module's location
LABLES = ["face","long_signature","signature"]
INTERNAL_MODEL_PATH = (MODULE_DIR / ".." / ".." / "Models" / "face_box_yolov8s.pt").resolve()
# ==============================================================================
# DATA STRUCTURE
# ==============================================================================
@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    class_id: int
    confidence: float
    @property
    def xyxy(self) -> np.array:
        """Returns the coordinates boundary as a numpy array."""
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.int32)
    @property
    def label(self) -> str:
        return LABLES[self.class_id]

class FaceDetector:
    def __init__(self, log_level: LogLevel = LogLevel.INFO,confidence: float = 0.5, draw: bool = False):
        """
        Initializes the FaceDetector module. The model path is resolved automatically
        internally and does not require external parameters.
        
        :param log_level: Desired LogLevel for this individual module instance
        :param draw: If True, draws and fills the segmentation mask over the frame matrix
        """
        # Pass the automatically resolved absolute path string to your inference engine
        self.face_detect_model = ModelInference(str(INTERNAL_MODEL_PATH))
        self.confidence = confidence
        self.draw = draw
        
        # The module instantiates its own isolated logger and sets its own level
        self.logger = Logger(name="face detection", level =log_level)
        
        self.logger.log_info(f"Model loaded successfully from {INTERNAL_MODEL_PATH} (Log Level: {log_level.name})")


    def detect_face(self, frame) -> tuple[list[BoundingBox], list[BoundingBox]]:
        """
        Processes a single frame matrix, runs object detection, and extracts face and signature coordinates.

        :param frame: The image frame matrix (numpy array)
        :return: A tuple containing two lists of BoundingBox objects: one for faces and one for signatures
        """
        if frame is None:
            self.logger.log_error("Received an empty or invalid frame matrix")
            return [], []

        result = self.face_detect_model.predict(frame=frame)
        boxes = result.boxes
        
        detected_face = []
        detected_signature = []
        for i,box in enumerate(boxes):
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            self.logger.log_debug(f"Class: {class_id}, Confidence : {confidence}")
            if confidence < self.confidence:
                self.logger.log_debug("Confidence isn't enough")
                continue

            # ---------------- Original internal debug section ----------------
            raw_tensor_2d = box.xyxy
            # Type: <class 'torch.Tensor'>, Example value: tensor([[120.4500, 240.1800, 310.8200, 680.5100]], device='cuda:0')
            self.logger.log_debug(f"raw_tensor_2d: {type(raw_tensor_2d)} (2D Shape: {raw_tensor_2d.shape}) Value: {raw_tensor_2d}")

            raw_tensor_1d = box.xyxy[0]
            # Type: <class 'torch.Tensor'>, Example value: tensor([120.4500, 240.1800, 310.8200, 680.5100], device='cuda:0')
            self.logger.log_debug(f"raw_tensor_1d: {type(raw_tensor_1d)} (1D Shape: {raw_tensor_1d.shape}) Value: {raw_tensor_1d}")

            float_list = raw_tensor_1d.tolist()
            # Type: <class 'list'>, Example value: [120.45000305175781, 240.17999267578125, 310.82000732421875, 680.510009765625]
            self.logger.log_debug(f"float_list: {type(float_list)} Value: {float_list}")

            mapped_ints = map(int, float_list)
            # Type: <class 'map'>, Example value: <map object at 0x7f8a3c2b1e10>
            # float_list:   [ 120.45,  240.18,  310.82,  680.51 ]
            #                  ↓        ↓        ↓        ↓
            #                  ↓ (Conversion only happens when requested)
            #                  ↓        ↓        ↓        ↓
            # mapped_ints:  ( 120   ,  240   ,  310   ,  680    )  <-- Only evaluated on unpack                
            self.logger.log_debug(f"mapped_ints: {type(mapped_ints)} Value: {mapped_ints}")

            x1, y1, x2, y2 = mapped_ints
            # Type: <class 'int'>, Example value: x1: 120, y1: 240, x2: 310, y2: 680
            self.logger.log_debug(f"Coordinates - x1: {type(x1)}={x1}, y1: {type(y1)}={y1}, x2: {type(x2)}={x2}, y2: {type(y2)}={y2}")
            # ---------------------------------------------------------------------
            bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, class_id=class_id, confidence=confidence)
            if class_id == 0:  # Assuming class_id 0 corresponds to "face"
                detected_face.append(bbox)
            elif class_id == 2:
                detected_signature.append(bbox)
            self.logger.log_debug(f"Length detected face: {len(detected_face)}")
            self.logger.log_debug(f"Length detected signature: {len(detected_signature)}")
            # Context-aware drawing inside the module
            if self.draw is True:
            # Crop the isolated region from the original frame safely
            # Get frame dimensions to prevent out-of-bounds slicing
                height, width = frame.shape[:2]
                crop_x1 = max(0, x1)
                crop_y1 = max(0, y1)
                crop_x2 = min(width, x2)
                crop_y2 = min(height, y2)
                cropped_face_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                resized_frame = cv2.resize(cropped_face_frame, (300, 400))
                cv2.imshow(f"Cropped face {i}", resized_frame)
        return detected_face, detected_signature
