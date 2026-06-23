"""
FRAME DETECTION TREE ARCHITECTURE
=================================

System Concept:
---------------
When processing hierarchical detections (e.g., Full Frame -> Detected Person -> Detected Shirt), 
each sub-image extraction (cropping) shifts the local origin (0,0) to the upper-left corner of the crop. 
Downstream models processing these crops return coordinates relative only to that crop (Local Coordinates).

Hierarchical Tree Structure & Coordinate Projection:
----------------------------------------------------

  +-------------------------------------------------------------+
  | ROOT NODE (Full Frame Canvas)                               |
  | (0, 0)                                                      |
  |      +---------------------------------------+              |
  |      | PERSON NODE (Local x1, y1 relative to Root)          |
  |      | (Global x1, y1)                                      |
  |      |      +-------------------------------+               |
  |      |      | SHIRT NODE (Local relative to |               |
  |      |      |             Person Node)      |               |
  |      |      +-------------------------------+               |
  |      +---------------------------------------+              |
  +-------------------------------------------------------------+

Mathematical Solution:
----------------------
This architecture uses a recursive Tree Data Structure (`FrameNode`) to encapsulate the coordinate system.
Each node retains a reference to its structural parent. When `get_global_coordinates()` is invoked:
1. It requests the global coordinates of its parent node via a recursive call.
2. It extracts the parent's absolute Top-Left origin (parent_x1, parent_y1).
3. It offsets its own local coordinates against that absolute origin:
      global_x = parent_x1 + local_x

This eliminates manual tracking of offsets or mutating global coordinate states in the main execution loop.
"""
class FrameNode:
    """
    Represents a contextual region of interest within a hierarchical computer vision pipeline.
    """
    def __init__(self, x1=0, y1=0, x2=None, y2=None, image=None, parent=None):
        self.parent = parent
        
        if parent is None and image is not None:
            # Nếu là root node
            self.image = image
            self.x1 = 0
            self.y1 = 0
            self.y2, self.x2 = image.shape[:2]
        elif parent is not None:
            # Nếu là child node (cắt từ parent)
            self.x1 = x1
            self.y1 = y1
            self.x2 = x2
            self.y2 = y2
            # Crop ảnh từ ảnh của parent node
            self.image = parent.image[y1:y2, x1:x2]

    def get_global_coordinates(self):
        """
        Recursively walks up the node tree to map local boundaries back to absolute main frame pixels.
        """
        # Base Case: Root node represents the full original image frame (0 offset dependency)
        if self.parent is None:
            return self.local_x1, self.local_y1, self.local_x2, self.local_y2
        
        # Recursive Step: Fetch the absolute global starting origin of the parent node
        parent_x1, parent_y1, _, _ = self.parent.get_global_coordinates()
        
        # Transform current local coordinates into absolute coordinates using parent absolute origin
        global_x1 = parent_x1 + self.local_x1
        global_y1 = parent_y1 + self.local_y1
        global_x2 = parent_x1 + self.local_x2
        global_y2 = parent_y1 + self.local_y2
        
        return global_x1, global_y1, global_x2, global_y2

    def is_valid(self):
        """
        Ensures the structural slice contains actual data array payloads to avoid processing empty crops.
        """
        return self.image is not None and self.image.size > 0