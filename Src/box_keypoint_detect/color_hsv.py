import cv2
import numpy as np

# Global variable to store the HSV image for the mouse callback function
hsv_img = None

def mouse_click_callback(event, x, y, flags, param):
    """
    Callback function to capture mouse clicks and print the exact HSV values
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        # Get HSV values at the clicked pixel coordinate (y, x)
        h, s, v = hsv_img[y, x]
        
        # Calculate standard web/graphics software equivalents for comparison
        web_h = h * 2
        web_s = int((s / 255.0) * 100)
        web_v = int((v / 255.0) * 100)
        
        print(f"[CLICK] Coordinates: (X: {x}, Y: {y})")
        print(f"  └─ OpenCV HSV: H={h}, S={s}, V={v}")
        print(f"  └─ Standard Graphic Equivalent: H={web_h}°, S={web_s}%, V={web_v}%\n")

def main():
    global hsv_img
    
    # 1. Path to your sample uniform image
    image_path = "D:\Van_Tien\Project\Camera\Input\human_detections\Screenshot 2026-06-22 161118.png"
    
    # Load the image in standard BGR format
    bgr_img = cv2.imread(image_path)
    
    if bgr_img is None:
        print(f"[ERROR] Could not open or find the image at: {image_path}")
        return

    # 2. Convert the image to HSV color space
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    height, width, _ = hsv_img.shape
    
    print(f"[INFO] Image Loaded: {width}x{height}")
    print("[INFO] --- OPTION 1: SAMPLING ALL PIXELS VIA PRINT ---")
    
    # To prevent terminal flooding, we sample every 40th pixel (adjust stride as needed)
    stride = 40 
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            h, s, v = hsv_img[y, x]
            print(f"Pixel({x},{y}) -> H:{h} S:{s} V:{v}")
            
    print("\n[INFO] --- OPTION 2: INTERACTIVE MOUSE DEBUGGING ---")
    print("[INFO] Click anywhere on the image window to get the exact OpenCV HSV bounds.")
    print("[INFO] Press 'q' or 'ESC' to exit the program.")
    
    # 3. Setup window and mouse callback registry
    window_name = "Uniform Color Analyzer (Click to Debug)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_click_callback)
    
    # Keep window alive until user exits
    while True:
        cv2.imshow(window_name, bgr_img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 27 is the ESC key code
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()