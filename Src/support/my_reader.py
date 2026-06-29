import cv2
import queue
import os
import time
from threading      import Thread
import threading
from dataclasses    import dataclass
from my_logger      import Logger,LogLevel 

@dataclass
class SRC_INFO: 
    src_type: str
    frame_width: int
    frame_height: int
    fps: float
    total_frames: int

class Reader:
    def __init__(self, source: str | int, queue_size: int = 2, log_level = LogLevel.DEBUG):
        # Assign a logger instance; fall back to a print-based mock if none is provided
        self.logger = Logger("Reader",log_level)
        
        self.logger.log_debug(f"Initializing Reader with source: {source} (type: {type(source)}), queue_size: {queue_size}")
        
        self.q = queue.Queue(maxsize=queue_size)
        # Initialize the underlying lock and the Condition object
        self._lock = threading.Lock()
        self.cv = threading.Condition(self._lock)
        # Define the low watermark threshold (20% remaining)
        self.low_watermark = int(queue_size * 0.2)

        self.stopped = False
        self.cap = None
        self.image_frame = None
        src_type = "Unknow"
        width  = 0 
        height = 0 
        fps    = 0.0
        total_frames = 0
        # Type guard: Validate if source is a camera index before string operations
        if isinstance(source, int):
            self.logger.log_debug(f"Source detected as integer. Initializing local camera index: {source}")
            self.cap = cv2.VideoCapture(source)
            # Read properties
            src_type = "Camera"
            width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps    = 0.0
            total_frames = 0
        else:
            # Extract file extension to determine pipeline behavior
            ext = os.path.splitext(source.lower())[1]
            self.logger.log_debug(f"Source detected as string. Extracted file extension: '{ext}'")
            
            if ext in ['.mp4', '.avi', '.mkv', '.mov']:
                self.logger.log_debug(f"Source matched video extension list. Initializing VideoCapture for file: {source}")
                self.cap = cv2.VideoCapture(source)

                src_type = "Video"
                width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps    = self.cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(self.cp.get(cv2.CAP_PROP_FRAME_COUNT))

            elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                self.logger.log_debug(f"Source matched image extension list. Attempting imread for file: {source}")
                self.image_frame = cv2.imread(source)
                if self.image_frame is None:
                    self.logger.log_error(f"FileNotFoundError: Cannot read static image path: {source}")
                    raise FileNotFoundError(f"Cannot read image path: {source}")
                # Read Properties
                src_type = "Image"
                height,width = self.image_frame.shape[:2]
                fps    = 0.0
                total_frames = 0
                self.logger.log_debug("Static image successfully cached into memory")
            else:
                # Fallback for network streaming sources (e.g., RTSP, HTTP URLs)
                self.logger.log_debug(f"Source extension '{ext}' not explicitly matched. Falling back to VideoCapture (Streaming/URL): {source}")
                self.cap = cv2.VideoCapture(source)
                # Read properties
                src_type = "Stream"
                width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fsp = 0.0
                total_frames = 0
        self.src_info = SRC_INFO(src_type,width,height,fps,total_frames)
        self.logger.log_info("Srouce information")
        self.logger.log_info(f"Source_type: {src_type},Width: {width}, Height: {height}, Fps: {fps}, Total Frames: {total_frames}")

    def start(self):
        # Static images do not require background thread orchestration
        if self.src_info == "Image":
            self.logger.log_debug("Starting Reader pipeline for static image mode. Seeding queue...")
            for i in range(self.q.maxsize):
                self.q.put(self.image_frame.copy())
            self.logger.log_debug(f"Queue successfully pre-populated with {self.q.maxsize} image frame copies")
            return self

        if self.cap is None or not self.cap.isOpened():
            self.logger.log_error("Cannot start Reader thread: VideoCapture handle is closed or uninitialized")
            return self

        self.logger.log_debug("Spawning background daemon worker thread for video decoding")
        t = Thread(target=self._update, args=())
        t.daemon = True
        t.start()
        self.logger.log_debug("Background worker thread started successfully")
        return self

    def _update(self):
        self.logger.log_debug("Worker thread loop entered execution phase")
        while not self.stopped:
            # --- STATIC IMAGE BRANCH ---
            if self.src_info.src_type == "Image":
                if not self.q.full():
                    self.q.put(self.image_frame.copy())
                else:
                    time.sleep(0.01)  # Keep throttle for static images to save CPU
                continue

            # --- LIVE STREAM / VIDEO BRANCH ---
            # Always read from the backend to clear the hardware/FFmpeg buffer
            success, frame = self.cap.read()
            
            # Get the current frame pointer position and total frames
            current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if not success:
                self.logger.log_warning("Read failed from cap")
                self.stopped = True
                break
            with self.cv:
                # If queue is full, put thread into wait queue until awakened
                while self.q.full() and not self.stopped:
                    self.logger.log_debug("Queue is full. Worker enter the wait queue") 
                    self.cv.wait()
                # Check again if stopped while waiting
                if self.stopped:
                    break
                # Safely push frame to queue after waking up or if queue wasn't full
                self.q.put(frame)
                self.logger.log_debug(f"Frame pushed to queue. Size: {self.q.qsize()}")
        self.logger.log_debug("Worker thread loop exited successfully")

    def read(self):
        # Constantly provide raw copies if processing a static image asset
        if self.src_info.src_type == "Image":
            self.logger.log_debug("Read request received in image mode. Returning cached memory copy")
            self.stopped = True
            return True, self.image_frame.copy()
        with self.cv:
            # Non-blocking pull: Return the oldest element if data exists in the queue
            if not self.q.empty():
                frame = self.q.get()
                self.logger.log_debug(f"Frame successfully retrieved from queue by main thread. Remaining queue size: {self.q.qsize()}")

                # Check if remaining items dropped below 20%
                if self.q.qsize() <= self.low_watermark:
                    self.logger.log_debug("Queue dropped below 20%. Signaling worker")
                    self.cv.notify()
                return True, frame
                
            self.logger.log_warning("Read request received but queue buffer is empty")
        return False, None

    def stop(self):
        # Set volatile flag to terminate the thread loop and free underlying system handles
        self.logger.log_debug("Stop signal received. Terminating Reader pipeline operations")
        self.stopped = True
        if self.cap is not None and self.cap.isOpened():
            self.logger.log_debug("Releasing VideoCapture hardware/file resources")
            self.cap.release()
            self.logger.log_debug("VideoCapture resource released successfully")

    def get_src_info(self):
        return self.src_info
    def is_stop(self):
        return self.stopped
