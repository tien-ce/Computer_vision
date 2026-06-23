import cv2  # Import OpenCV to handle video capture and frame decoding
from threading import Thread  # Import Thread to run the stream reader in the background
import queue  # Import queue to handle thread-safe, FIFO frame buffering


class StreamReader:

    def __init__(self, source: str, queue_size: int = 2):
        # Initialize the OpenCV VideoCapture object with the stream URL or device index
        self.cap = cv2.VideoCapture(source)

        # Create a thread-safe First-In-First-Out queue with a strict maximum size
        # A small size (like 2) forces the pipeline to drop old frames and prevent latency lag
        self.q = queue.Queue(maxsize=queue_size)

        # Initialize a boolean flag to control the background execution loop safely
        self.stopped = False

    def start(self):
        # Create a new background thread that target executes the hidden _update method
        t = Thread(target=self._update, args=())

        # Set the thread as a daemon so it terminates automatically when the main script exits
        t.daemon = True

        # Start the execution of the thread in the background
        t.start()

        # Return the current instance to allow chained initialization (e.g., reader = StreamReader().start())
        return self

    def _update(self):
        # Keep looping continuously until the stop signal flag is explicitly set to True
        while not self.stopped:
            # Read the next available raw frame sequence from the network stream
            success, frame = self.cap.read()
            # Check if the frame buffer queue still has available slots open
            if not self.q.full():
                # If the stream disconnects or fails to decode a frame, drop out of the loop
                if not success:
                    self.stop()
                    break

                # Place the successfully decoded frame array directly into the memory queue
                self.q.put(frame)

    def read(self):
        # Check if there is at least one frame sitting inside the memory buffer queue
        # If the queue has data, pop and return the oldest available frame array
        # If the queue is currently completely empty, immediately return None instead of blocking
        if not self.q.empty():
            return True, self.q.get()
        return False, None

    def stop(self):
        # Set the execution flag to True to halt the internal while loop running in the thread
        self.stopped = True

        # Release the hardware or network resource hooks held by the OpenCV video capture object
        self.cap.release()