import cv2
import glob
import os
import shutil

def sort_directory_images(src_folder_path: str):
    """
    Reads all images from a directory sequentially, displays them, and shifts them
    into subfolders (Red, Blue, etc.) inside that exact same directory based on hotkeys.
    
    Hotkeys:
    [r] -> Red, [b] -> Blue, [l] -> Light_Blue, [g] -> Green, [w] -> White
    [n] -> Skip/Next, [q] -> Quit pipeline
    
    :param src_folder_path: Path to the directory containing unsorted masked images
    """
    if not os.path.exists(src_folder_path):
        print(f"[ERROR] Source directory path does not exist: {src_folder_path}")
        return

    # 1. Map key commands to target subfolder names
    key_mapping = {
        ord('r'): 'Red',
        ord('b'): 'Blue',
        ord('l'): 'Light_Blue',
        ord('g'): 'Green',
        ord('w'): 'White',
        ord('h'): 'Heavy_Blue'
    }

    # Atomically create the subfolders inside the source directory if missing
    for folder_name in key_mapping.values():
        os.makedirs(os.path.join(src_folder_path, folder_name), exist_ok=True)

    # 2. Gather and sort all valid image assets from the directory
    extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.webp')
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(src_folder_path, ext)))
        
    image_paths.sort()

    if not image_paths:
        print(f"[WARNING] No valid image files available for sorting in: {src_folder_path}")
        return

    print(f"[INFO] Initializing sorting pipeline for {len(image_paths)} images.")
    print("Controls: [r]=Red, [b]=Blue, [l]=Light_Blue, [g]=Green, [w]=White | [n]=Skip, [q]=Quit")

    # 3. Main processing loop over directory assets
    for idx, path in enumerate(image_paths):
        filename = os.path.basename(path)
        print(f"\n[INFO] Progress ({idx + 1}/{len(image_paths)}): {filename}")
        file_ext = os.path.splitext(filename)[1] # Keep original extension (.png, .jpg, etc.)
        frame = cv2.imread(path)
        if frame is None:
            print(f"[ERROR] Failed to parse file: {filename}")
            continue

        cv2.imshow("Directory Image Sorter Utility", frame)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            print("[INFO] 'q' pressed. Terminating sorting execution loop.")
            break
        elif key == ord('n'):
            print(f"[INFO] Skipped: {filename}")
            continue
        elif key in key_mapping:
            target_folder_name = key_mapping[key]
            target_dir_path = os.path.join(src_folder_path, target_folder_name)
            # Count existing files in the specific subfolder to safely determine the next index 'i'
            existing_files = [f for f in os.listdir(target_dir_path) if os.path.isfile(os.path.join(target_dir_path, f))]
            next_index = len(existing_files)
            # Format the new filename according to your pattern: picture{i}.ext
            new_filename = f"picture{next_index}{file_ext}"
            destination_path = os.path.join(target_dir_path, new_filename)
            try:
                # Shift filesystem target location
                shutil.move(path, destination_path)
                print(f"[SUCCESS] Moved '{filename}' -> Subfolder: {target_folder_name}")
            except Exception as e:
                print(f"[ERROR] Filesystem shift failed for {filename}: {e}")
        else:
            print("[WARNING] Invalid hotkey. Use r, b, l, g, w or n/q.")
            
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Pass the path to your main target directory containing the pictures
    target_directory = r"D:\Van_Tien\Project\Camera\Input\Picture\shirt_segmentation"
    
    # Execute folder-wide utility sorting structure
    sort_directory_images(target_directory)