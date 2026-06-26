from uniform_detect import UniformDetector
import cv2
from my_logger import LogLevel,Logger
img_path = r"C:\Users\IT-TIEN\Pictures\Process\1770628174905.jfif"
def main():
    logger = Logger("Test",LogLevel.DEBUG)
    frame = cv2.imread(img_path)
    if frame is None:
        return
    uniform_detector = UniformDetector(LogLevel.DEBUG,confidence = 0.7, draw = False)
    uniform_results = uniform_detector.detect_shirt(frame)
    if len(uniform_results) == 0:
        logger.log_debug("No shirt is detected")
        return
    logger.log_debug(f"Detect {len(uniform_results)} shirt from frame")
    for uniform_result in uniform_results:
        x1,y1,x2,y2 = uniform_result.xyxy
if __name__ == "__main__":
    main()
