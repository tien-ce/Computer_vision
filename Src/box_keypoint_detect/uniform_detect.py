import os
import cv2 
import numpy as np
from my_reader import StreamReader
from dotenv import load_dotenv
from transform import LetterboxTransformer
from inference import ModelInference
from my_logger import LogLevel
import my_logger as logger
from uniform import detector
def main():
    # 1. Initialize the custom YOLO Pose Estimation model wrapper
    print("[INFO] Loading Pose Estimation model...")
    model = ModelInference("../Models/yolo26s-pose.pt")
    print("[INFO] Model loaded successfully.")

    # 2. Open live camera stream (0 is typically the default built-in webcam)
    print("[INFO] Opening live camera stream...")
    cap = cv2.VideoCapture(r"C:\Users\IT-TIEN\Downloads\1398474346186522464.mp4")
    if not cap.isOpened():
        print("[ERROR] Could not open camera stream.")
        return
    print("[INFO] Camera stream is active and running.")

    # Configuration Constants
    TARGET_SIZE = (256, 256)
    UNIFORM_THRESHOLD = 0.35
    frame_count = 0

    while True:
        # Capture frame-by-frame from the live camera stream
        ret, frame = cap.read()
        #frame = cv2.imread("D:\Van_Tien\Project\Camera\Input\human_detections\Screenshot 2026-06-19 134938.png")
        if frame is None:
            print("[ERROR] Failed to grab frame from camera sequence.")
            break

        # Get real-time dimensions of the incoming camera stream
        img_h, img_w = frame.shape[:2]
        frame_count += 1

        # 3. Run pose estimation inference model on the current live frame
        result = model.predict(frame)
        num_people = len(result)
        if num_people > 0:
            print(f"[FRAME {frame_count}] AI Model processing {num_people} person(s)...")
        print(f"Lenght Result: {len(result)}")
        if result.keypoints is not None:
            keypoints_list = result.keypoints.xyn.tolist()
        # ---- STEP 4: LOOP THROUGH EACH PERSON OBJECT IN THE RESULTS LIST ----
        for person_idx, person_kpts in enumerate(keypoints_list):
                # Denormalize floating coordinates (0.0 - 1.0) to exact pixel coordinates
                left_shoulder  = [int(person_kpts[5][0] * img_w),  int(person_kpts[5][1] * img_h)]
                right_shoulder = [int(person_kpts[6][0] * img_w),  int(person_kpts[6][1] * img_h)]
                left_hip       = [int(person_kpts[11][0] * img_w), int(person_kpts[11][1] * img_h)]
                right_hip      = [int(person_kpts[12][0] * img_w), int(person_kpts[12][1] * img_h)]

                # Occlusion Guard: Skip processing if any critical torso joint is undetected (0, 0)
                if left_shoulder[0] == 0 or right_shoulder[0] == 0 or left_hip[0] == 0 or right_hip[0] == 0:
                    continue

                # Draw solid green indicator circles on all valid joints for this person
                for kp in person_kpts:
                    kp_x, kp_y = int(kp[0] * img_w), int(kp[1] * img_h)
                    if kp_x > 0 and kp_y > 0:
                        cv2.circle(frame, (kp_x, kp_y), radius=4, color=(0, 255, 0), thickness=-1)

                # ---- STEP 1: GEOMETRIC TORSO POLYGON MASKING ----
                # Create a local single-channel blank black mask matching global frame size
                mask = np.zeros((img_h, img_w), dtype=np.uint8)
                
                # Sequence the torso boundary vertices in standard clockwise order
                polygon_points = np.array([left_shoulder, right_shoulder, right_hip, left_hip], dtype=np.int32)
                
                # Render the exact torso polygon boundary as solid white (255)
                cv2.fillConvexPoly(mask, polygon_points, 255)
                
                # Isolate the color shirt region from the background using bitwise logic
                masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

                # ---- STEP 2: BOUNDING BOX EXTRACTION & MATRIX SLICING ----
                x_coords = [left_shoulder[0], right_shoulder[0], left_hip[0], right_hip[0]]
                y_coords = [left_shoulder[1], right_shoulder[1], left_hip[1], right_hip[1]]
                
                x_min, x_max = max(0, int(min(x_coords))), min(img_w, int(max(x_coords)))
                y_min, y_max = max(0, int(min(y_coords))), min(img_h, int(max(y_coords)))
                
                # Slice tightly bounded sub-matrices to minimize processing overhead
                cropped_shirt = masked_frame[y_min:y_max, x_min:x_max]
                cropped_geometry_mask = mask[y_min:y_max, x_min:x_max]
                
                if cropped_shirt.size == 0 or cropped_geometry_mask.size == 0:
                    continue

                # ---- STEP 3: LETTERBOX SHAPE STANDARDIZATION ----
                #tx = LetterboxTransformer(target_size=TARGET_SIZE)
                
                # Sync channel depth to 3-channels to satisfy your internal transformer requirements
                #cropped_geometry_mask_3ch = cv2.cvtColor(cropped_geometry_mask, cv2.COLOR_GRAY2BGR)
                
                # resized_shirt = tx.transform(cropped_shirt)
                # resized_geometry_mask_3ch = tx.transform(cropped_geometry_mask_3ch)
                # Revert mask back to 1-channel grayscale matrix for mathematical evaluation
                # resized_geometry_mask = cv2.cvtColor(resized_geometry_mask_3ch, cv2.COLOR_BGR2GRAY)
                resized_shirt = cropped_shirt
                resized_geometry_mask = cropped_geometry_mask


                # ---- STEP 4: HSV COLOR SEGMENTATION & TRUE RATIO CALCULATION ----
                hsv_shirt = cv2.cvtColor(resized_shirt, cv2.COLOR_BGR2HSV)
                # Extract pixel counts from non-zero elements
                actual_torso_pixels = cv2.countNonZero(resized_geometry_mask)
                # Call the prediction method directly
                best_match_name, color_percentage = detector.predict(hsv_shirt, actual_torso_pixels)
                logger.log_debug( f"Best mach name: {best_match_name}, color percentage {color_percentage}")
                # ---- STEP 5: VISUAL RECORDING ON GLOBAL CANVAS ----
                # Switch color status representation dynamically (Green = Pass, Red = Infraction)
                if color_percentage >= UNIFORM_THRESHOLD:
                    ui_color = (0, 255, 0)  # Green for valid uniform
                    status_text = f"P{person_idx} {best_match_name}: {color_percentage:.1f}%"
                else:
                    ui_color = (0, 0, 255)  # Red for non-uniform
                    status_text = f"P{person_idx}: Not Uniform"
                logger.log_info( f"Status text: {status_text}")
                # Apply visual markings onto the global frame canvas
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), ui_color, 2)
                cv2.putText(frame, status_text, (x_min, max(y_min - 10, 20)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, ui_color, 2, cv2.LINE_AA)

        # ---- STEP 6: SINGLE RENDER PIPELINE ENTRY POINT ----
        # Display the single combined output frame containing all drawings at the end of the while loop
        cv2.imshow("Factory Live Uniform Analytics Stream", frame)

        # Non-blocking input switch capture ('q' or 'ESC' to exit cleanly)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("[INFO] Terminating live video stream connection...")
            break

    # Resource release
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Camera streaming resources closed cleanly.")

if __name__ == "__main__":
    main()