from inference import ModelInference
from my_reader import Reader
from my_logger import LogLevel, Logger
from human_detect import HumanDetector
from shirt_segment import ShirtSegmenter
from coordinate_tree import FrameNode
from hsv_analyzer import MaskedShirtAnalyzer
import cv2
import os
logger = Logger("Main log",LogLevel.DEBUG)
# --------------------- ONLY for save picture ---------------------#
# Global control variables for UI tracking
save_path = r"D:\Van_Tien\Project\Camera\Input\Picture\shirt_segmentation"


def main():
    human_detector = HumanDetector(
        log_level=LogLevel.INFO,
        draw=False
    )
    shirt_segmentor = ShirtSegmenter(
        log_level=LogLevel.DEBUG,
        draw = False
    )
    # Count only entries that are files
    frame_counter = len([f for f in os.listdir(save_path) if os.path.isfile(os.path.join(save_path, f))])
    print(f"Number of files: {frame_counter}")
    reader = Reader(r"D:\Van_Tien\Project\Camera\Input\Video\People\1812051860060444051.mp4")
    reader.start()
    while True:
        ret, frame = reader.read()
        original_node = FrameNode(image=frame)
        if ret is False:
            continue
        people_boxes = human_detector.detect_people(frame) # Return an array of bounding boxs
        # 1. Initialize the control flag at the start of each frame iteration
        should_quit = False
        for box in people_boxes:
            x1, y1, x2, y2 = box.xyxy
            croped_people_node = FrameNode(x1,y1,x2,y2,parent=original_node)
            croped_people_frame = croped_people_node.image
            masked_shirt_frame = shirt_segmentor.mask_shirt(croped_people_frame)
            if masked_shirt_frame is None:
                continue
            # 1. Instantiate the analyzer with the newly isolated masked frame
            analyzer = MaskedShirtAnalyzer(masked_shirt_frame, log_level=LogLevel.DEBUG)
            # 2. Open an isolated interactive window sized to 480x640 (or any custom dimension)
            # Moving your mouse over this window will instantly log the accurate underlying HSV values
            analyzer.start_interactive_display(width=480, height=640)
            cv2.imshow("Original frame",frame)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                should_quit = True
                break
            elif key == 32:  # Spacebar to skip frame
                break
            elif key == ord('s'):
                frame_counter += 1
                filename = os.path.join(save_path, f"masked_shirt_{frame_counter}.png")
                cv2.imwrite(filename, masked_shirt_frame)
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