from uniform_detect import UniformDetector
import cv2
from my_logger import LogLevel,Logger
from my_reader import Reader
src_path = r"/home/vantien/Downloads/Parking enter-20260625T134008Z-3-001/Parking_enter/20260625_160444.mp4"
def main():
    logger = Logger("Test",LogLevel.DEBUG)
    reader = Reader(
        source=src_path,
        log_level=LogLevel.INFO,
        queue_size=200
    )   
    reader.start()
    try:
        while True:
            ret, frame = reader.read()
            if ret is False:
                logger.log_warning(f"Read frame failed from {src_path}")
                continue
            uniform_detector = UniformDetector(LogLevel.DEBUG,confidence = 0.7, draw = False)
           # uniform_results = uniform_detector.detect_shirt(frame)
           # if len(uniform_results) == 0:
           #     logger.log_debug("No shirt is detected")
           #     return
           # logger.log_debug(f"Detect {len(uniform_results)} shirt from frame")
           # for uniform_result in uniform_results:
           #     x1,y1,x2,y2 = uniform_result.xyxy
            cv2.imshow("Orignal Frame",frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    except Exception as e:
        logger.log_error(f"Error: {e}")
    finally:
        reader.stop()

if __name__ == "__main__":
    main()
