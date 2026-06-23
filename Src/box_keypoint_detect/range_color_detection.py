import cv2
import numpy as np

def nothing(x):
    # Dummy function required by cv2.createTrackbar
    pass

def main():
    # 1. Path to your fixed sample image
    image_path = "D:\Van_Tien\Project\Camera\Input\human_detections\Screenshot 2026-06-22 153457.png"
    
    # Load the image in standard BGR format
    bgr_img = cv2.imread(image_path)
    
    if bgr_img is None:
        print(f"[ERROR] Could not open or find the image at: {image_path}")
        return

    # Resize image if it's too large for your screen monitor
    # We can use a standard square size or keep its aspect ratio
    bgr_img = cv2.resize(bgr_img, (400, 400))
    
    # Convert the fixed image to HSV color space once
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    total_pixels = hsv_img.shape[0] * hsv_img.shape[1]

    # 2. Create a named window for GUI controllers
    control_window = "Color Range Tuner"
    cv2.namedWindow(control_window, cv2.WINDOW_AUTOSIZE)

    # 3. Create Trackbars for Lower and Upper HSV boundaries
    # Hue max is 179 in OpenCV, Saturation and Value max are 255
    cv2.createTrackbar("Low H", control_window, 0, 179, nothing)
    cv2.createTrackbar("High H", control_window, 179, 179, nothing)
    
    cv2.createTrackbar("Low S", control_window, 0, 255, nothing)
    cv2.createTrackbar("High S", control_window, 255, 255, nothing)
    
    cv2.createTrackbar("Low V", control_window, 0, 255, nothing)
    cv2.createTrackbar("High V", control_window, 255, 255, nothing)

    # Initialize Trackbars to a default wide range
    cv2.setTrackbarPos("Low H", control_window, 90)
    cv2.setTrackbarPos("High H", control_window, 130)
    cv2.setTrackbarPos("Low S", control_window, 50)
    cv2.setTrackbarPos("High S", control_window, 255)
    cv2.setTrackbarPos("Low V", control_window, 50)
    cv2.setTrackbarPos("High V", control_window, 255)

    print("[INFO] GUI Started. Adjust the trackbars to filter the uniform.")
    print("[INFO] Press 'q' or 'ESC' on the image window to close.")

    # 4. Infinite loop to recalculate mask when trackbars change
    while True:
        # Get current positions of all 6 trackbars
        low_h = cv2.getTrackbarPos("Low H", control_window)
        high_h = cv2.getTrackbarPos("High H", control_window)
        low_s = cv2.getTrackbarPos("Low S", control_window)
        high_s = cv2.getTrackbarPos("High S", control_window)
        low_v = cv2.getTrackbarPos("Low V", control_window)
        high_v = cv2.getTrackbarPos("High V", control_window)

        # Structure the boundary arrays dynamically
        lower_bound = np.array([low_h, low_s, low_v], dtype=np.uint8)
        upper_bound = np.array([high_h, high_s, high_v], dtype=np.uint8)

        # Generate the binary mask based on current boundary limits
        color_mask = cv2.inRange(hsv_img, lower_bound, upper_bound)

        # Calculate matching ratio percentage
        matching_pixels = cv2.countNonZero(color_mask)
        matching_percentage = (matching_pixels / total_pixels) * 100

        # Draw the real-time percentage text onto a copy of the mask window
        mask_display = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(
            mask_display, 
            f"Matching: {matching_percentage:.2f}%", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, 
            (0, 255, 0), 
            2
        )

        # Show the original source image and the updated binary mask matrix
        cv2.imshow("Original Image", bgr_img)
        cv2.imshow("Live Color Mask Output", mask_display)

        # Break loop on 'q' or ESC key press
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q') or key == 27:
            # Print the final determined array numbers before closing for easy copy-paste
            print("\n[FINAL RESULTS] Copy these values into your main code:")
            print(f"LOWER_BOUNDARY = np.array([{low_h}, {low_s}, {low_v}], dtype=np.uint8)")
            print(f"UPPER_BOUNDARY = np.array([{high_h}, {high_s}, {high_v}], dtype=np.uint8)")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()