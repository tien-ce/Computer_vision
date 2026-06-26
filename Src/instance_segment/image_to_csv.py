import cv2
import numpy as np
import pandas as pd
import os

def find_color_range_by_mode(hist, threshold_ratio=0.2):
    """
    Find the mode position and expand left/right until the density
    drops below (threshold_ratio * density_at_mode) or a boundary condition is met.
    """
    mode_val = int(np.argmax(hist))
    mode_density = hist[mode_val][0]
    min_density_threshold = mode_density * threshold_ratio
    accepted_values = [mode_val]
    
    # Step left
    left = mode_val - 1
    while left >= 0:
        if hist[left][0] >= min_density_threshold:
            accepted_values.append(left)
            left -= 1
        else:
            break
            
    # Step right
    right = mode_val + 1
    while right < len(hist):
        if hist[right][0] >= min_density_threshold:
            accepted_values.append(right)
            right += 1
        else:
            break
            
    return accepted_values

def extract_hsv_inputs(image_path, threshold_ratio=0.1):
    """
    Extract HSV rows that match the density criteria around the mode.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot read image at {image_path}")
        return None
        
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv_img)
    
    hist_h = cv2.calcHist([hsv_img], [0], None, [180], [0, 180])
    hist_s = cv2.calcHist([hsv_img], [1], None, [256], [0, 256])
    hist_v = cv2.calcHist([hsv_img], [2], None, [256], [0, 256])
    
    valid_h = find_color_range_by_mode(hist_h, threshold_ratio)
    valid_s = find_color_range_by_mode(hist_s, threshold_ratio)
    valid_v = find_color_range_by_mode(hist_v, threshold_ratio)
    
    h_mask = np.isin(h_channel, valid_h)
    s_mask = np.isin(s_channel, valid_s)
    v_mask = np.isin(v_channel, valid_v)
    
    combined_mask = h_mask & s_mask & v_mask
    matching_pixels = hsv_img[combined_mask]
    
    return matching_pixels.reshape(-1, 3)

def save_hsv_to_csv(image_path, label, output_path, threshold_ratio=0.1):
    """
    Extract HSV data, append the specified label, and save to a CSV file.
    """
    # Extract the H, S, V rows
    hsv_data = extract_hsv_inputs(image_path, threshold_ratio)
    
    if hsv_data is None or hsv_data.shape[0] == 0:
        print("No matching data extracted to save.")
        return
        
    # Create a pandas DataFrame
    df = pd.DataFrame(hsv_data, columns=['H', 'S', 'V'])
    
    # Add the label column with the specified value for all rows
    df['Label'] = label
    
    # Check if file already exists to append or write fresh with headers
    file_exists = os.path.isfile(output_path)
    
    # Save to CSV (append mode allows collecting data from multiple images)
    df.to_csv(output_path, mode='a', index=False, header=not file_exists)
    print(f"Successfully saved {df.shape[0]} rows with label '{label}' to: {output_path}")

if __name__ == "__main__":
    # Define your configuration
    INPUT_IMAGE = "C:\Users\IT-TIEN\Pictures\Screenshots\Screenshot_2026-06-25 111425.png"
    TARGET_LABEL = "Green"
    OUTPUT_CSV = "hsv_dataset.csv"
    THRESHOLD = 0.15
    
    # Execute the processing and saving pipeline
    save_hsv_to_csv(
        image_path=INPUT_IMAGE, 
        label=TARGET_LABEL, 
        output_path=OUTPUT_CSV, 
        threshold_ratio=THRESHOLD
    )