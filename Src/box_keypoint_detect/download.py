import os
import cv2 
import numpy as np
from reader import StreamReader
from dotenv import load_dotenv
from transform import Preprocessor
from inference import ModelInference

original_image_path = "../../Input/original_images/train/images/"
orignial_image_files = os.listdir(original_image_path)
human_detection_path = "../../Input/human_detections"
def save_human_image(frame, results, target_size=(256, 256)):
    """
    Saves and displays detected human bounding boxes resized to a uniform shape.
    Returns False if 'q' is pressed to signal an exit, True otherwise.
    """
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            if class_id == 0:
                # 1. Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # Prevent out-of-bounds array slicing
                x1, y1 = max(0, x1), max(0, y1)
                
                # 2. Crop the target region
                cropped_person = frame[y1:y2, x1:x2]
                if cropped_person.size == 0:
                    continue
                
                # --- Aspect Ratio Preserving Resize (Letterboxing) ---
                h, w = cropped_person.shape[:2]
                desired_w, desired_h = target_size
                
                # Determine scaling factor
                scale = min(desired_w / w, desired_h / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                
                # Resize keeping original aspect ratio
                resized_crop = cv2.resize(cropped_person, (new_w, new_h))
                
                # 3. Create black canvas of target size
                # Example matrix visualization:
                #      ┌───────────────────┐
                #      │  0   0   0   0   0│
                #      │  0   0   0   0   0│ -> desired_h
                #      │  0   0   0   0   0│
                #      └───────────────────┘
                #            desired_w
                padded_person = np.zeros((desired_h, desired_w, 3), dtype=np.uint8)
                
                # 4. Center the resized image on the canvas
                x_offset = (desired_w - new_w) // 2
                y_offset = (desired_h - new_h) // 2
                
                # Example slice mapping visualization:
                #      ┌───────────────────┐
                #      │  0   0   0   0   0│ -> y_offset (Top Padding)
                #      ├───────────────────┤
                #      │  X   X   X   X   X│ -> pasted resized_crop (new_h)
                #      ├───────────────────┤
                #      │  0   0   0   0   0│ -> Bottom Padding
                #      └───────────────────┘
                padded_person[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_crop
                # ----------------------------------------------------

                cv2.imshow("Inference Results", padded_person)
                
                key = cv2.waitKey(0) & 0xFF
                
                if key == ord('s'):
                    existing_files = [f for f in os.listdir(human_detection_path)]
                    save_filename = f"captured_frame_{len(existing_files)}.jpg"
                    save_file_path = os.path.join(human_detection_path, save_filename)
                    
                    # Save the uniformly resized image
                    cv2.imwrite(save_file_path, padded_person)
                    print(f"Resized image successfully saved: {save_file_path}")
                    
                elif key == ord('n'):
                    continue
                    
                elif key == ord('q'):
                    return False  # Signal to main loop to exit completely
                    
    return True  # Signal to main loop to continue to next image

def main():
    preprocessor = Preprocessor(model_type="yolo")
    model = ModelInference("../Models/yolov8s.pt")  
    
    try:
        for image_file in orignial_image_files:
            image_path = os.path.join(original_image_path, image_file)
            frame = cv2.imread(image_path)
            
            if frame is None:
                continue  
            
            image = preprocessor.process(frame)
            results = model.predict(image)
            
            # Catch the status flag. If False, break the image loop.
            should_continue = save_human_image(frame, results, target_size=(256, 256))
            if not should_continue:
                print("Exiting application...")
                break
    finally:
        model.stop()  
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()