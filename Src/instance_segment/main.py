from inference import ModelInference
from my_reader import Reader
from my_logger import LogLevel, Logger
from human_detect import HumanDetector
from shirt_segment import ShirtSegmenter
from coordinate_tree import FrameNode
from hsv_analyzer import MaskedShirtAnalyzer
from color_classifier import ShirtColorClassifier
import cv2
import os
import sys
import numpy as np
logger = Logger("Main log",LogLevel.INFO)
# --------------------- ONLY for save picture ---------------------#
# Global control variables for UI tracking
save_path = r"D:\Van_Tien\Project\Camera\Input\Picture\shirt_segmentation"


def main():
    human_detector = HumanDetector(
        log_level=LogLevel.INFO,
        confidence = 0.8,
        draw=False
    )
    shirt_segmentor = ShirtSegmenter(
        log_level=LogLevel.INFO,
        confidence = 0.7,
        draw = False
    )
    shirt_classifier = ShirtColorClassifier(log_level=LogLevel.INFO,confidence = 0.8)
    # Count only entries that are files
    frame_counter = len([f for f in os.listdir(save_path) if os.path.isfile(os.path.join(save_path, f))])
    logger.log_info(f"Number of files: {frame_counter}")
    src_path = r"D:\Van_Tien\Project\Camera\Input\Video\Video\20260623_160039.mp4"
    if len(sys.argv) > 1:
        src_path = sys.argv[1]
        logger.log_info("Source file from sys argument")
    reader = Reader(
        source=src_path,
        log_level=LogLevel.INFO,
        queue_size=2000
    )   
    reader.start()
    while True:
        ret, frame = reader.read()
        if ret is False:
            logger.log_warning(f"Read frame failed from {src_path}")
            continue
        original_node = FrameNode(image=frame)
        people_boxes = human_detector.detect_people(frame) # Return an array of bounding boxs
        # 1. Initialize the control flag at the start of each frame iteration
        should_quit = False
        
        for box in people_boxes:
            x1, y1, x2, y2 = box.xyxy
            croped_people_node = FrameNode(x1,y1,x2,y2,parent=original_node)
            croped_people_frame = croped_people_node.image
            # Create a child node that occupies 100% of the parent's space
            masked_shirt_node = FrameNode(
                x1=0, 
                y1=0, 
                x2=croped_people_node.image.shape[1], 
                y2=croped_people_node.image.shape[0], 
                parent=croped_people_node
            )
            # Overwrite the automatically cropped image with the masked version
            masked_shirt_node.image = shirt_segmentor.mask_shirt(croped_people_frame)
            masked_shirt_frame = masked_shirt_node.image
            # Local points on the masked (cropped) frame, both have same size.
            local_masked_shirt_result = shirt_segmentor.segment_shirt(croped_people_frame)
            if masked_shirt_frame is None:
                continue
            local_masked_shirt_points = local_masked_shirt_result.points
            global_x1,global_y1,_,_ = masked_shirt_node.get_global_coordinates()
            # # 3. Vectorized broadcast addition to shift local points to global coordinates
            # # Numpy np.array(points) have shape (N,1,2) with 
            #     # Axis 0 (:): Represents the index of each individual point or vertex. The colon : means "select all items along this dimension" (all N points).
            #     # Axis 1 (:): Represents the channel grouping (OpenCV structures contours with an extra dimension of size 1). The colon : selects everything in this axis.
            #     # Axis 2 (0 or 1): Represents the actual coordinate values, where index 0 holds the X coordinate and index 1 holds the Y coordinate.
            # # local_points[:, :, 0] matches X, local_points[:, :, 1] matches Y
            global_points = np.array(local_masked_shirt_points, dtype=np.int32)
            global_points[:, :, 0] += global_x1
            global_points[:, :, 1] += global_y1
            
            # # Draw the shirt segmentation into original shirt
            cv2.polylines (
                frame,
                [global_points],
                isClosed=True,
                color=(0,255,0), # Green
                thickness=2
            )
            
            # 1. Instantiate the analyzer with the newly isolated masked frame
            # analyzer = MaskedShirtAnalyzer(masked_shirt_frame, log_level=LogLevel.DEBUG)
            # 2. Open an isolated interactive window sized to 480x640 (or any custom dimension)
            # Moving your mouse over this window will instantly log the accurate underlying HSV values
            #analyzer.start_interactive_display(width=480, height=640)
            color_result = shirt_classifier.predict_color(masked_frame=masked_shirt_frame)
            if color_result:
                lable = color_result.label
                logger.log_debug(f"Result: {color_result.label} | Confidence: {color_result.confidence:.2f}%")
                cv2.putText(
                    frame, 
                    f"Uniform: {lable}", 
                    (int(x1), int(y1)), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=0.6, 
                    color=(0, 255, 0),  # Green text
                    thickness=2
                )
        cv2.imshow("Original frame",frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            should_quit = True
            break
        elif key == 32:  # Spacebar to skip frame
            break
        elif key == ord('s'):
            frame_counter += 1
            filename = os.path.join(save_path, f"masked_shirt_{frame_counter}.png")
            # cv2.imwrite(filename, masked_shirt_frame)
            logger.log_info(f"Saved: {filename}")
            break # Advance to next bounding box once saved        

        if should_quit:
            break
        
    # 1. Close all UI windows and unregister mouse callbacks first
    cv2.destroyAllWindows()
    # 2. Finally, stop the video/camera stream thread safely
    reader.stop()
if __name__ == "__main__":
    main()