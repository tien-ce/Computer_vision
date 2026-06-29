import os
import sys
import cv2
import numpy as np
from my_logger import LogLevel, Logger


class Frame2Video:
    def __init__(self, frame_width: int, frame_height: int, fps: float, total_frames: int, out_path: str, log_level: LogLevel = LogLevel.DEBUG):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.logger = Logger("Frame2Video",log_level)
        self.out_path = out_path
        self.video_writer = cv2.VideoWriter(out_path, fourcc, fps, (frame_width,frame_height))
        self.frame_index = 0
        self.logger.log_info(f"Video Config: {frame_width}x{frame_height} @ {fps} FPS | Total Frames: {total_frames}")
    
    def write_frame(self,frame):
        self.video_writer.write(frame)
        self.logger.log_debug(f"Write frame {self.frame_index}")

    def stop(self):
        self.video_writer.release()
        self.logger.log_info(f"Video exported successfully. Target destination: {self.out_path}")
