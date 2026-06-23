import os
import cv2 
from my_reader import Reader
from dotenv import load_dotenv
from transform import Preprocessor, LetterboxTransformer
from inference import ModelInference

# Load environment variables from a .env file if it exists
load_dotenv()
rtp_stream_url = os.environ.get("RTP_STREAM_URL")
human_detection_path = "../Input/human_detections"
def save_human_image (frame,results, path):
    """Saves the detected human bounding boxes from the inference results to the specified path."""
    for result in results:
        boxes = result.boxes
        for i, box in enumerate(boxes):
            # Check if the detected class is 'person' (class_id 0 in COCO dataset)
            class_id = int (box.cls[0])
            if class_id == 0:
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    # Crop the person from the original frame
                    cropped_person = frame[y1:y2, x1:x2]
                    # Save the cropped image to the specified path
                    cv2.imwrite(path, cropped_person)
def main():
    # Initialize the StreamReader with the RTP stream URL
    preprocessor = Preprocessor(model_type="yolo")
    # Start the stream reader in the background, this thread will continuously read frames from the RTP stream and buffer them in memory
    model = ModelInference("../Models/yolov8s.pt")  # Load the YOLOv8 model for inference
    #stream_reader = StreamReader(rtp_stream_url)
    #stream_reader.start()
    transformer1 = LetterboxTransformer(target_size=(256, 256)) # First transformer (Detected person --> disred target size)

    stream_reader = cv2.VideoCapture(0) # for testing with a webcam
    try:
        while True:
            # Read the latest available frame from the StreamReader's internal queue
            ret, frame = stream_reader.read()
            # If a valid frame was successfully retrieved (i.e., not None), process it
            if frame is None or ret is False:
                continue  # If no frame is available, skip the rest of the loop and try again
            #cv2.imshow("Raw Stream", frame)  # Display the raw input stream for reference
            # 2. Preprocess frame
            image = preprocessor.process(frame)
            # 3. Run inference on the preprocessed frame and get the results
            result = model.predict(image, verbose = False)
            
            boxes = result.boxes
            for i, box in enumerate(boxes):
                # Check if the detected class is 'person' (class_id 0 in COCO dataset)
                class_id = int(box.cls[0])
                if class_id == 0:
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    # Prevent out-of-bounds array slicing
                    x1, y1 = max(0, x1), max(0, y1)                    
                    
                    # Draw a bounding box around the person on the main frame stream
                    confidence = float(box.conf[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Person {confidence:.2f}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    

            # Show the global frame with all corrected original bounding boxes
            cv2.imshow("Inference Results", frame)
            #4. Postprocess the inference results to visualize them on the original frame
            # output_frame = postprocessor.process_results(frame, result)
            
            
            
            key = cv2.waitKey(1) & 0xFF
            # Trigger save action on pressing 's'
            if key == ord('s'):
                # Read the number of pictures already saved to avoid overwriting
                existing_files = [f for f in os.listdir(human_detection_path) if f.startswith("captured_frame") and f.endswith(".jpg")]
                # Save as a standard JPEG file
                cv2.imwrite(f"{human_detection_path}/captured_frame_{len(existing_files)}.jpg", frame)
                print("Frame successfully saved as JPEG.")
            elif key == ord('q'):
                break
    finally:
        # stream_reader.stop()
        model.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()