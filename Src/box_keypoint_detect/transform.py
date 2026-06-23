import cv2
import numpy as np
class LetterboxTransformer:
    def __init__(self, target_size: tuple):
        """
        Initializes the transformer with a fixed target size.
        
        Args:
            target_size: Tuple of (desired_w, desired_h) required by the model.
        """
        self.desired_w, self.desired_h = target_size
        self.scale = 1.0
        self.x_offset = 0
        self.y_offset = 0

    def transform(self, image: np.ndarray) -> np.ndarray:
        """
        Resizes and pads the input image to the target size, saving the state.
        Supports both 3-channel (BGR) and 1-channel (Grayscale/Mask) matrices.
        """
        h, w = image.shape[:2]
        
        # 1. Determine scaling factor
        self.scale = min(self.desired_w / w, self.desired_h / h)
        new_w = int(w * self.scale)
        new_h = int(h * self.scale)
        
        # 2. Extract number of channels dynamically from the matrix shape tuple
        # For 3-channel image, shape is (H, W, 3) -> len is 3
        # For 1-channel mask, shape is (H, W) -> len is 2
        channels = image.shape[2] if len(image.shape) == 3 else None
        
        # 3. Resize keeping original aspect ratio
        resized_crop = cv2.resize(image, (new_w, new_h))
        
        # 4. Center padding calculations
        self.x_offset = (self.desired_w - new_w) // 2
        self.y_offset = (self.desired_h - new_h) // 2
        
        # 5. Dynamic canvas memory allocation using saved instance variables
        if channels is not None:
            # Create a 3D matrix for standard BGR color crops
            padded_person = np.zeros((self.desired_h, self.desired_w, channels), dtype=np.uint8)
        else:
            # Create a 2D matrix for single-channel grayscale geometry masks
            padded_person = np.zeros((self.desired_h, self.desired_w), dtype=np.uint8)
        
        # 6. Paste the resized array into the center of the pre-allocated canvas
        padded_person[self.y_offset:self.y_offset+new_h, self.x_offset:self.x_offset+new_w] = resized_crop
        
        return padded_person

    def inverse_coords(self, coords: tuple) -> tuple:
        """
        Maps coordinates from the model's output space back to the original image space.
        Supports both points (x, y) and bounding boxes (x1, y1, x2, y2).
        """
        if len(coords) == 2:
            x, y = coords
            orig_x = int((x - self.x_offset) / self.scale)
            orig_y = int((y - self.y_offset) / self.scale)
            return orig_x, orig_y
            
        elif len(coords) == 4:
            x1, y1, x2, y2 = coords
            orig_x1 = int((x1 - self.x_offset) / self.scale)
            orig_y1 = int((y1 - self.y_offset) / self.scale)
            orig_x2 = int((x2 - self.x_offset) / self.scale)
            orig_y2 = int((y2 - self.y_offset) / self.scale)
            return orig_x1, orig_y1, orig_x2, orig_y2