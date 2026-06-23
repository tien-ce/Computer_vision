import cv2
import numpy as np
from transform import LetterboxTransformer
from inference import ModelInference
from log import LogLevel, log
from pathlib import Path
folder_path = Path("../../Input/smoke_detections")
files = [f.name for f in folder_path.iterdir() if f.is_file()]

def main():   
    smoke_model = ModelInference("../Models/smoke_yolo11s_last.pt")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        log(LogLevel.ERROR,"Could not open camera stream.")
        return
    log(LogLevel.INFO, "Camera stream is active and running.")
    while True:
    #for file in files:
        #frame = cv2.imread(str(folder_path / file))
        # Capture frame-by-frame from the live camera stream
        ret, frame = cap.read()
        if frame is None:
            log(LogLevel.WARNING, f"Frame is None")
            continue
        log(LogLevel.INFO, "Frame is loaded")
        result = smoke_model.predict(frame)
        log(LogLevel.DEBUG, f"Size of result: {len(result.boxes)}")
        boxes = result.boxes
        names = result.names
        for i, name in enumerate(names):
            log(LogLevel.DEBUG, f"List class: {names[i]}")
            log(LogLevel.DEBUG, f"List class: {name}")
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            class_name = names[class_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Smoke detect:", frame)

        # Non-blocking input switch capture ('q' or 'ESC' to exit cleanly)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("[INFO] Terminating live video stream connection...")
            break

    cv2.destroyAllWindows

if __name__ == "__main__":
    main()