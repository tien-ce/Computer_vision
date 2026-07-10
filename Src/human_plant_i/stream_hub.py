import async_reader
import threading
from my_logger import Logger, LogLevel

class VideoStreamBroadcastHub:
    def __init__(self, level: LogLevel = LogLevel.DEBUG):
        self.logger = Logger("Stream Hub",level=level)
        self.sub_count = 0              # Number of subcriber
        self.read_count = 0
        self.rc_mutex = threading.lock() # Lock for read_count variable
        self.rw_mutex = threading.lock() # Lock for reader and writer
