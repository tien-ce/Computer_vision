import os
import cv2 
import numpy as np
from reader import StreamReader
from dotenv import load_dotenv
from Src.transform import Preprocessor, LetterboxTransformer
from inference import ModelInference

def main():
    # 1. Initialize YOLO model (using yolo11n.pt or yolov8n.pt depending on your installation)
    print("[INFO] Loading Pose Estimation model...")
    model = ModelInference("../Models/yolo26s-pose.pt")
    print("[INFO] Model loaded successfully.")

    # 2. Open camera stream (0 is typically the default built-in webcam)
    print("[INFO] Opening camera stream...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        return
    print("[INFO] Camera stream active.")

    TARGET_SIZE = (256, 256)
    frame_count = 0

    while True:
        # ret, frame = cap.read()
        frame = cv2.imread("D:\Van_Tien\Project\Camera\Input\human_detections\Screenshot 2026-06-19 130928.png")
        # Get original image dimensions
        img_h, img_w = frame.shape[:2]
        if frame is None:
            print("[ERROR] Failed to grab frame.")
            break

        frame_count += 1

        # 3. Run person detection model on the frame
        results = model.predict(frame)
        result = results[0]
        keypoints_object = result.keypoints

        if keypoints_object is not None:
            # Convert the keypoints tensor to a standard Python list: shape (num_people, 17, 2)
            # keypoints_list = keypoints_object.xy.tolist()
            # To this (using .xyn targets normalized coordinates):
            keypoints_list = keypoints_object.xyn.tolist()
            num_people = len(keypoints_list)
            
            # Log summary for the current frame if people are detected
            if num_people > 0:
                print(f"[FRAME {frame_count}] Detected {num_people} potential skeleton(s).")

            for person_idx, kpts in enumerate(keypoints_list):
                # print (f"kpts {kpts}")
                for kp in kpts:
                    kp_x, kp_y = int(kp[0]* img_w), int(kp[1] * img_h)
                    print (f"Kpx, Kpy: {kp_x,kp_y}")
                    # Skip undetected or zero-assigned keypoints
                    if kp_x == 0 and kp_y == 0:
                        continue
                        
                    # Draw a small solid green circle at each joint position
                    cv2.circle(frame, (kp_x, kp_y), radius=5, color=(0, 255, 0), thickness=-1)
                # Ensure the detected skeleton has all required 17 standard keypoints
                if len(kpts) < 17:
                    print(f"  └─ [SKIPPED] Person {person_idx}: Incomplete skeleton (Keypoints < 17).")
                    continue
                # Extract (x, y) coordinates for the 4 critical joints defining the torso
                # Index 5: Left Shoulder, Index 6: Right Shoulder
                # Index 11: Left Hip, Index 12: Right Hip
                # left_shoulder  = kpts[5]
                # right_shoulder = kpts[6]
                # left_hip       = kpts[11]
                # right_hip      = kpts[12]
                # DENORMALIZATION: Multiply float coordinates (0.0 - 1.0) by image width and height
                left_shoulder  = [int(kpts[5][0] * img_w),  int(kpts[5][1] * img_h)]
                right_shoulder = [int(kpts[6][0] * img_w),  int(kpts[6][1] * img_h)]
                left_hip       = [int(kpts[11][0] * img_w), int(kpts[11][1] * img_h)]
                right_hip      = [int(kpts[12][0] * img_w), int(kpts[12][1] * img_h)]
                # Filter out zero-values which indicate occluded or undetected joints
                if left_shoulder[0] == 0 or right_shoulder[0] == 0 or left_hip[0] == 0 or right_hip[0] == 0:
                    print(f"  └─ [SKIPPED] Person {person_idx}: Critical joints (Shoulders/Hips) are occluded.")
                    continue
                # ---- STEP 1: GEOMETRIC POLYGON MASKING ----
                # 1. Create a black canvas mask with the exact same size as the original frame
                mask = np.zeros((img_h, img_w), dtype=np.uint8)
                
                # 2. Define the polygon vertices in clockwise/counter-clockwise order
                # Order: Left Shoulder -> Right Shoulder -> Right Hip -> Left Hip
                polygon_points = np.array([
                    left_shoulder, 
                    right_shoulder, 
                    right_hip, 
                    left_hip
                ], dtype=np.int32)
                
                # 3. Fill the inside of the polygon with pure white color (255) on the black mask
                cv2.fillConvexPoly(mask, polygon_points, 255)
                
                # 4. Apply bitwise AND to extract only the torso region from the original frame
                masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
                # ---- STEP 2: BOUNDING BOX EXTRACTION FOR SLICING ----
                # 5. Crop a bounding box around the polygon area to isolate it for resizing
                # This step reduces the matrix size so it doesn't process the entire black screen
                x_coords = [left_shoulder[0], right_shoulder[0], left_hip[0], right_hip[0]]
                y_coords = [left_shoulder[1], right_shoulder[1], left_hip[1], right_hip[1]]
                
                x_min, x_max = max(0, int(min(x_coords))), min(img_w, int(max(x_coords)))
                y_min, y_max = max(0, int(min(y_coords))), min(img_h, int(max(y_coords)))
                
                # Slice the tight bounding box from both the masked frame and the mask itself
                cropped_shirt = masked_frame[y_min:y_max, x_min:x_max]
                cropped_geometry_mask = mask[y_min:y_max, x_min:x_max]
                if cropped_shirt.size == 0:
                    print(f"  └─ [SKIPPED] Person {person_idx}: Masked crop resulted in an empty matrix.")
                    continue

                # ---- STEP 3: LETTERBOX TRANSFORMATION (RESIZING) ----
                tx = LetterboxTransformer(target_size=TARGET_SIZE)
                # Convert the 1-channel mask to a 3-channel image to match the color crop shape
                cropped_geometry_mask_3ch = cv2.cvtColor(cropped_geometry_mask, cv2.COLOR_GRAY2BGR)
                
                # Pass both 3-channel matrices through the transformer cleanly
                resized_shirt = tx.transform(cropped_shirt)
                resized_geometry_mask_3ch = tx.transform(cropped_geometry_mask_3ch)
                
                # Convert the resized mask back to 1-channel so cv2.countNonZero can read it
                resized_geometry_mask = cv2.cvtColor(resized_geometry_mask_3ch, cv2.COLOR_BGR2GRAY)
                
                display_resized = resized_shirt.copy()
                
                # ---- STEP 4: COLOR SEGMENTATION & TRUE RATIO MATH LOGIC ----
                # Convert the square resized BGR shirt to HSV space
                hsv_shirt = cv2.cvtColor(display_resized, cv2.COLOR_BGR2HSV)
                
                # Converted HSV boundaries from your web target color (217°, 71%, 62%)
                LOWER_BOUNDARY = np.array([98, 50, 40], dtype=np.uint8)
                UPPER_BOUNDARY = np.array([118, 255, 255], dtype=np.uint8)
                
                # Generate binary color mask (White = uniform color matches)
                color_mask = cv2.inRange(hsv_shirt, LOWER_BOUNDARY, UPPER_BOUNDARY)
                
                # MATH FIXED: Define the dynamic denominator using non-zero polygon mask pixels
                actual_torso_pixels = cv2.countNonZero(resized_geometry_mask)
                matching_color_pixels = cv2.countNonZero(color_mask)
                
                if actual_torso_pixels == 0:
                    print(f"  └─ [SKIPPED] Person {person_idx}: Torso calculation returned 0 pixels.")
                    continue
                
                # Compute the final percentage ratio independent of distance or letterbox padding
                color_percentage = (matching_color_pixels / actual_torso_pixels) * 100
                print (f"color_percentage {color_percentage}")
        # Show the global camera feed with detections
        cv2.imshow("Camera Feed", frame)

        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Exiting program via user request.")
            break

    # Clean up resources
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Program terminated cleanly.")

if __name__ == "__main__":
    main()