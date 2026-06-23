# shirt_analyzer.py
from dataclasses import dataclass
import cv2
import numpy as np

from my_logger import Logger, LogLevel

@dataclass
class DetailedColorStats:
    mean_h: int; mean_s: int; mean_v: int
    min_h: int; max_h: int
    min_s: int; max_s: int
    min_v: int; max_v: int


class MaskedShirtAnalyzer:
    def __init__(self, masked_frame: np.ndarray, log_level: LogLevel = LogLevel.INFO):
        """
        Initializes the analyzer with a pre-masked frame and pre-computes stats 
        strictly for the garment ROI. Background pixels are completely ignored.
        """
        self.logger = Logger(name="MaskedShirtAnalyzer")
        self.logger.setLevel(log_level)
        
        if masked_frame is None:
            self.logger.log_error("Received an empty masked frame matrix")
            raise ValueError("Masked frame cannot be None")

        self.src_frame = masked_frame
        self.hsv_src_frame = cv2.cvtColor(self.src_frame, cv2.COLOR_BGR2HSV)
        
        # Binary segmentation mask for the Garment ROI
        self.is_garment = np.any(self.src_frame > 0, axis=2)
        
        # Pre-compute structural statistics strictly for the Garment ROI
        self.garment_stats = self._compute_roi_stats()
        
        # Render window states
        self.display_frame = None
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.window_name = "Masked Shirt Analyzer Display"
        
        # Interactive tracking states
        self.current_mouse_hsv = (0, 0, 0)
        self.is_mouse_pressed = False

    
    def _compute_roi_stats(self) -> DetailedColorStats:
        """Extracts and computes comprehensive HSV statistical boundaries exclusively for the garment ROI."""
        pixels = self.hsv_src_frame[self.is_garment]
        if len(pixels) == 0:
            return DetailedColorStats(0, 0, 0, 0, 0, 0, 0, 0, 0)
            
        means = np.mean(pixels, axis=0)
        mins = np.min(pixels, axis=0)
        maxs = np.max(pixels, axis=0)
        
        return DetailedColorStats(
            mean_h=int(means[0]), mean_s=int(means[1]), mean_v=int(means[2]),
            min_h=int(mins[0]), max_h=int(maxs[0]),
            min_s=int(mins[1]), max_s=int(maxs[1]),
            min_v=int(mins[2]), max_v=int(maxs[2])
        )

    # ==============================================================================
    # UI DRAWING SECTION (EASY TO MODIFY AND EXTEND IN FUTURE)
    # ==============================================================================
    def _draw_ui_overlay(self, target_render_frame: np.ndarray):
        """
        Renders textual analytics dashboards directly onto the display frame window canvas.
        Modify this block to change font styles, positions, or append new evaluation metrics.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        line_type = cv2.LINE_AA
        
        # --- BLOCK 1: ROI STATS OVERLAY ---
        g_stats = self.garment_stats
        
        # Row data definitions for Garment ROI: (Text string, Y coordinate position, Color tuple BGR)
        ui_rows = [
            (f"GARMENT AVG HSV: ({g_stats.mean_h}, {g_stats.mean_s}, {g_stats.mean_v})", 25, (0, 255, 0)),
            (f"GARMENT H RANGE: [{g_stats.min_h} - {g_stats.max_h}]", 45, (0, 255, 0)),
            (f"GARMENT S RANGE: [{g_stats.min_s} - {g_stats.max_s}]", 65, (0, 255, 0)),
            (f"GARMENT V RANGE: [{g_stats.min_v} - {g_stats.max_v}]", 85, (0, 255, 0)),
        ]
        
        # Render static text blocks
        for text, y_pos, color in ui_rows:
            cv2.putText(target_render_frame, text, (15, y_pos), font, font_scale, color, thickness, line_type)
            
        # --- BLOCK 2: DYNAMIC INTERACTIVE MOUSE DATA OVERLAY ---
        # Only renders text when mouse is pressed down inside the Garment ROI
        if self.is_mouse_pressed:
            mh, ms, mv = self.current_mouse_hsv
            status_text = f"ROI TARGET HSV: ({mh}, {ms}, {mv})"
            
            # Draw background box for dynamic text readability
            cv2.rectangle(target_render_frame, (10, target_render_frame.shape[0] - 35), 
                          (target_render_frame.shape[1] - 10, target_render_frame.shape[0] - 10), (50, 50, 50), -1)
            
            cv2.putText(target_render_frame, status_text, (15, target_render_frame.shape[0] - 18), 
                        font, font_scale, (255, 255, 255), thickness, line_type)
    # ==============================================================================

    def _mouse_callback(self, event, x, y, flags, param):
        """Handles mouse press and movement triggers within the isolated Garment ROI."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Map coordinates immediately to verify if the initial click hits the ROI
            orig_x = int(x * self.scale_x)
            orig_y = int(y * self.scale_y)
            src_h, src_w = self.src_frame.shape[:2]
            
            if 0 <= orig_x < src_w and 0 <= orig_y < src_h and self.is_garment[orig_y, orig_x]:
                self.is_mouse_pressed = True
                self.current_mouse_hsv = tuple(self.hsv_src_frame[orig_y, orig_x])
                self._update_display()
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.is_mouse_pressed = False
            self._update_display()
            
        elif event == cv2.EVENT_MOUSEMOVE and self.is_mouse_pressed:
            orig_x = int(x * self.scale_x)
            orig_y = int(y * self.scale_y)
            src_h, src_w = self.src_frame.shape[:2]
            
            # Only update values if the dragged mouse remains inside the valid Garment ROI
            if 0 <= orig_x < src_w and 0 <= orig_y < src_h and self.is_garment[orig_y, orig_x]:
                self.current_mouse_hsv = tuple(self.hsv_src_frame[orig_y, orig_x])
                self._update_display()

    def _update_display(self):
        """Refreshes the active window frame instance with newly calculated text graphics."""
        temp_render = self.display_frame.copy()
        self._draw_ui_overlay(temp_render)
        cv2.imshow(self.window_name, temp_render)


    def start_interactive_display(self, width: int = 400, height: int = 600):
        """Creates the display interface canvas, attaches callbacks, and mounts UI projections."""
        src_h, src_w = self.src_frame.shape[:2]
        self.scale_x = src_w / width
        self.scale_y = src_h / height
        
        self.display_frame = cv2.resize(self.src_frame, (width, height), interpolation=cv2.INTER_LINEAR)
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        # Trigger initial clean render
        self._update_display()