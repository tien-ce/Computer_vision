from ultralytics import YOLO

class ModelInference:
    def __init__(self, model_path: str = "yolov8n.pt"):
        """Loads the local YOLOv8 model weights."""
        # The library automatically downloads the file once if it's missing, then runs fully local
        self.model = YOLO(model_path)

    def predict(self, frame, verbose = False):
        """
        Executes object detection on the preprocessed frame.
        Filters specifically for human targets (class 0).
        """
        # verbose=False keeps the console clean during real-time loops
        results = self.model.predict(source=frame, classes=[0], verbose = verbose)
        return results[0]  # Return the results for the single processed frame

    def stop(self):
        """Placeholder for resource cleanup to match pipeline specifications."""
        pass