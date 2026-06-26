import cv2
import os
import sys
import numpy as np
from human_detect import HumanDetector
from shirt_segment import ShirtSegmenter
from coordinate_tree import FrameNode
from color_classifier import ShirtColorClassifier
from my_logger import LogLevel, Logger

# Initialize logger for offline processing
logger = Logger("Offline Processing", LogLevel.INFO)

def main():
    # 1. Initialize AI modules for the pipeline
    human_detector = HumanDetector(log_level=LogLevel.INFO, confidence=0.5, draw=False)
    shirt_segmentor = ShirtSegmenter(log_level=LogLevel.INFO, confidence=0.5, draw=False)
    shirt_classifier = ShirtColorClassifier(log_level=LogLevel.INFO,confidence = 0.8)

    # 2. Set up input and output video paths
    src_path = r"C:\Users\IT-TIEN\Downloads\Canteen-20260625T052841Z-3-001\Canteen\20260625_114014.mp4"
    if len(sys.argv) > 1:
        src_path = sys.argv[1]
        
    # Define the output directory path
    output_dir = r"D:\Van_Tien\Project\Camera\Output\Video"
    
    # Extract the filename from the source path (e.g., "20260623_160039.mp4")
    input_filename = os.path.basename(src_path)
    
    # Combine the output directory and the extracted filename
    output_path = os.path.join(output_dir, input_filename)
    # 3. Use standard VideoCapture for sequential reading to prevent frame drops
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        logger.log_error(f"Cannot open input video file: {src_path}")
        return

    # Gather original video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.log_info(f"Video Config: {frame_width}x{frame_height} @ {fps} FPS | Total Frames: {total_frames}")

    # 4. Initialize VideoWriter with modified FPS to slow down the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Create a slowdown factor (e.g., 2.0 means twice as slow, 1.5 means 1.5x slower)
    slowdown_factor = 2.0  
    
    # Calculate the new lower FPS
    playback_fps = fps / slowdown_factor
    
    # Pass the lower FPS into the VideoWriter
    video_writer = cv2.VideoWriter(output_path, fourcc, playback_fps, (frame_width, frame_height))

    frame_idx = 0

    # 5. Synchronous sequential processing loop to extract every single frame
    while True:
        success, frame = cap.read()
        if not success:
            logger.log_info("Reached the end of the input video file.")
            break

        frame_idx += 1
        print(f"Processing frame {frame_idx}/{total_frames} ...", end="\r")

        # Initialize root node for global coordinate mapping tree
        original_node = FrameNode(image=frame)
        people_boxes = human_detector.detect_people(frame)

        for box in people_boxes:
            x1, y1, x2, y2 = box.xyxy
            croped_people_node = FrameNode(x1, y1, x2, y2, parent=original_node)
            croped_people_frame = croped_people_node.image

            # Run shirt segmentation model (Call prediction only once)
            local_masked_shirt_result = shirt_segmentor.segment_shirt(croped_people_frame)
            if local_masked_shirt_result is None:
                continue

            local_masked_shirt_points = local_masked_shirt_result.points

            # Generate local binary mask using OpenCV primitives
            mask = np.zeros(croped_people_frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [local_masked_shirt_points], 255)
            masked_shirt_frame = cv2.bitwise_and(croped_people_frame, croped_people_frame, mask=mask)

            # Update the coordinate tree with the mask node
            masked_shirt_node = FrameNode(
                x1=0, y1=0,
                x2=croped_people_node.image.shape[1],
                y2=croped_people_node.image.shape[0],
                parent=croped_people_node
            )
            masked_shirt_node.image = masked_shirt_frame

            # Transform local polygon vertices to global absolute frame coordinates
            global_x1, global_y1, _, _ = masked_shirt_node.get_global_coordinates()
            global_points = np.array(local_masked_shirt_points, dtype=np.int32)
            global_points[:, :, 0] += global_x1
            global_points[:, :, 1] += global_y1



            # Predict uniform color classification from the isolated mask array
            color_result = shirt_classifier.predict_color(masked_frame=masked_shirt_frame)
            if color_result is None:
                label = "Unkown"
                cv2.polylines(frame, [global_points], isClosed=True, color=(0, 0,255), thickness=2)
                cv2.putText(
                    frame,
                    f"Uniform: {label}",
                    (int(x1), int(y1)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6,
                    color=(0, 0,255),
                    thickness=2
                )
            else:
                label = color_result.label
                # Draw the mapped shirt polygon onto the original master frame
                cv2.polylines(frame, [global_points], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.putText(
                    frame,
                    f"Uniform: {label}",
                    (int(x1), int(y1)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6,
                    color=(0, 255, 0),
                    thickness=2
                )

        # 6. Write the processed frame directly into the output video encoder
        video_writer.write(frame)

        # Display visual progress interface
        cv2.imshow("Processing Pipeline (Offline Mode)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.log_warning("Processing loop interrupted by user.")
            break

    # 7. Release hardware/file resources and close display windows
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    logger.log_info(f"Video exported successfully. Target destination: {output_path}")

if __name__ == "__main__":
    main()