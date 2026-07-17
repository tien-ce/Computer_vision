import os
import queue
import time
from threading import Thread

import cv2
import numpy as np

from my_dataclass import SRC_INFO
from my_logger    import Logger, LogLevel


class Reader:
    def __init__(self, source: str | int, queue_size: int = 30, time_out: float = 0.5, frame_stride: int = 5, level: LogLevel = LogLevel.DEBUG):
        self.logger = Logger("Reader", level=level)
        self.logger.log_debug(f"Initializing Reader with source: {source} (type: {type(source)}), queue_size: {queue_size}")

        # --- Core state ---
        self.source = source
        self.q = queue.Queue(maxsize=queue_size)
        self.time_out = time_out
        self.frame_stride = frame_stride
        self.stopped = False
        self.cap = None
        self.image_frame = None
        self.thread = None

        # --- Source type detection (no cv2.VideoCapture opened yet) ---
        if isinstance(source, int):
            src_type = "Camera"
        else:
            ext = os.path.splitext(source.lower())[1]
            if ext in ['.mp4', '.avi', '.mkv', '.mov']:
                src_type = "Video"
            elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                src_type = "Image"
            else:
                src_type = "Stream"

        # --- Placeholder src info values, populated once a stream actually connects ---
        width, height, fps, total_frames = -1, -1, None, None

        # --- Static image preload ---
        if src_type == "Image":
            self.logger.log_debug(f"Source matched image extension list. Attempting imread for file: {source}")
            self.image_frame = cv2.imread(source)
            if self.image_frame is None:
                self.logger.log_error(f"FileNotFoundError: Cannot read static image path: {source}")
                raise FileNotFoundError(f"Cannot read image path: {source}")
            height, width = self.image_frame.shape[:2]
            self.logger.log_debug("Static image successfully cached into memory")

        # --- Finalize source info ---
        self.src_info = SRC_INFO(src_type, width, height, fps, total_frames)
        self.logger.log_info("Source information initialized (deferred for streams)")

    def start(self) -> bool:
        # Static images do not require background thread orchestration
        if self.src_info.src_type == "Image":
            self.logger.log_debug("Starting Reader pipeline for static image mode. Seeding queue...")
            for i in range(self.q.maxsize):
                self.q.put(self.image_frame.copy())
            self.logger.log_debug(f"Queue successfully pre-populated with {self.q.maxsize} image frame copies")
            return True

        # Reset stopped flag
        self.stopped = False

        # Clear queue of any stale frames from the previous connection session
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break

        # Connection Logic: Try to connect exactly once
        if self.cap is None or not self.cap.isOpened():
            self.logger.log_info(f"Attempting to connect to source: {self.source}")
            params = [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000
            ]

            self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG, params)
            
            # If not connected, stop the reader immediately and return
            if not self.cap.isOpened():
                self.logger.log_error("Connection failed. Stopping reader.")
                self.stop()
                return False
            
            # Read properties once connected successfully
            width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps    = self.cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.src_info = SRC_INFO(self.src_info.src_type, width, height, fps, total_frames)
            self.logger.log_info(f"Successfully connected! Resolution: {width}x{height}")

        # Spawn worker thread if it doesn't exist or has finished running
        if self.thread is None or not self.thread.is_alive():
            self.logger.log_debug("Spawning background daemon worker thread for video decoding")
            self.thread = Thread(target=self._update, args=())
            self.thread.daemon = True
            self.thread.start()
            self.logger.log_debug("Background worker thread started successfully")
        return True

    def _update(self):
        previous_time = time.time()
        current_stride = 0
        self.logger.log_debug("Worker thread loop entered execution phase")
        current_time = time.time()
        interval = current_time - previous_time
        previous_time = current_time        
        while not self.stopped:
            # --- STATIC IMAGE BRANCH ---
            if self.src_info.src_type == "Image":
                if not self.q.full():
                    self.q.put(self.image_frame.copy())
                else:
                    time.sleep(0.01)
                continue
            # --- LIVE STREAM / VIDEO BRANCH ---
            # Ensure cap is still open
            if self.cap is None or not self.cap.isOpened():
                self.logger.log_error("VideoCapture is not opened. Exiting worker thread.")
                self.stop()
                break
            success, frame = self.cap.read()
            if not success:
                self.logger.log_warning("Read failed from cap")
                if self.src_info.src_type == "Video":
                    if self.cap.get(cv2.CAP_PROP_POS_FRAMES) >= self.src_info.total_frames:
                        self.logger.log_info("End of video file reached. Stopping Reader thread")
                        self.stop()
                else:
                    # Stream read failed. Stop the reader instead of trying to reconnect.
                    self.logger.log_error("Stream read failed.")
                continue
            if current_stride < self.frame_stride:
                current_stride += 1
                self.logger.log_debug(f"Frame skipped due to frame_stride. Current stride: {current_stride}/{self.frame_stride}")
                continue
            # Drop oldest frame if full to keep real-time
            if self.q.full():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)
            current_stride = 1
        self.logger.log_debug("Worker thread loop exited successfully")

    def read(self) -> tuple[bool, np.ndarray]:
        self.logger.log_debug("Read request received from consumer thread")
        
        # Constantly provide raw copies if processing a static image asset
        if self.src_info.src_type == "Image":
            self.logger.log_debug("Read request received in image mode. Returning cached memory copy")
            self.stopped = True
            return True, self.image_frame.copy()
            
        if self.src_info.src_type != "Image" and (self.cap is None or not self.cap.isOpened()):
            return False, None

        try:
            frame = self.q.get(block=True, timeout=self.time_out) 
            self.logger.log_debug(f"Frame successfully retrieved from queue. Remaining queue size: {self.q.qsize()}")
            return True, frame
        except queue.Empty:
            self.logger.log_warning("Queue is empty. No frame available to read")                
            return False, None

    def stop(self):
        self.logger.log_info("Stop signal received. Terminating Reader pipeline operations")
        if self.stopped == True:
            return
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break
        if self.thread is not None and self.thread.is_alive():
            self.logger.log_info("Waiting for background worker thread to finish...")
            self.thread.join(timeout=2.0)
        if self.cap is not None and self.cap.isOpened():
            self.logger.log_info("Releasing VideoCapture hardware/file resources")
            self.cap.release()
            self.cap = None  # Reset to None so a new connection is attempted next start
            self.logger.log_info("VideoCapture resource released successfully")
        self.stopped = True
    def get_src_info(self):
        return self.src_info
        
    def is_stop(self):
        return self.stopped
