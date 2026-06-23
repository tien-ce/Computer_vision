from dataclasses import dataclass
from pathlib import Path
import cv2
import torch
# Import standard components from your centralized logger module
from my_logger import Logger, LogLevel
from inference import ModelInference

# Resolve the absolute directory of this module file at runtime
MODULE_DIR = Path(__file__).resolve().parent

# Hardcoded relative path layout from THIS module's location
INTERNAL_MODEL_PATH = (MODULE_DIR / ".." / ".." / "Models" / "human_box_yolov8s.pt").resolve()

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
    def xyxy(self) -> tuple[int, int, int, int]:
        """Returns the coordinates boundary as a swift tuple."""
        return self.x1, self.y1, self.x2, self.y2
# ==============================================================================
# DETECTOR MODULE CLASS
# ==============================================================================
class HumanDetector:
    def __init__(self, log_level: LogLevel = LogLevel.INFO, draw: bool = False):
        """
        Initializes the HumanDetector module. The model path is resolved automatically
        internally and does not require external parameters.
        
        :param log_level: Desired LogLevel for this individual module instance
        :param debug: If True, draws bounding boxes over the frame matrix
        """
        # Pass the automatically resolved absolute path string to your inference engine
        self.human_box_model = ModelInference(str(INTERNAL_MODEL_PATH))
        self.draw = draw
        
        # The module instantiates its own isolated logger and sets its own level
        self.logger = Logger(name="HumanDetector")
        self.logger.setLevel(log_level)
        
        self.logger.log_info(f"Model loaded successfully from {INTERNAL_MODEL_PATH} (Log Level: {log_level.name})")

    def detect_people(self, frame) -> list[BoundingBox]:
        """
        Processes a single frame matrix, runs object detection, and extracts person coordinates.
        
        :param frame: The image frame matrix (numpy array)
        :return: A list of BoundingBox objects containing person coordinates
        """
        if frame is None:
            self.logger.log_error("Received an empty or invalid frame matrix")
            return []

        result = self.human_box_model.predict(frame=frame)
        boxes = result.boxes
        num_people = len(boxes)
        
        self.logger.log_debug(f"Number of detectd people: {num_people}")
        
        detected_people = []

        for i, box in enumerate(boxes):
            class_id = int(box.cls[0])
            confidence = boxes.conf[i]
            self.logger.log_debug(f"Class id: {class_id}, confidence: {confidence}")
            if class_id == 0:
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
                detected_people.append(bbox)

                # Context-aware drawing inside the module
                if self.draw is True:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return detected_people