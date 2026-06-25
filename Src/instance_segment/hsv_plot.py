import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_and_plot_hsv_3d(folder_path):
    """
    Reads all CSV files in the folder, groups data by text labels, 
    and plots HSV coordinates in 3D with unique colors per label.
    """
    search_pattern = os.path.join(folder_path, "*.csv")
    csv_files = glob.glob(search_pattern)
    
    if not csv_files:
        print(f"No CSV files found in the directory: {folder_path}")
        return

    # Master DataFrame to pool all cleaned data records
    master_df = pd.DataFrame()

    print(f"Found {len(csv_files)} CSV files. Initializing supervised data extraction...")

    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path, header=None, names=['H', 'S', 'V', 'Label'])
            
            # Clean and sanitize coordinate entries to prevent non-numeric crashes
            df['H'] = pd.to_numeric(df['H'], errors='coerce')
            df['S'] = pd.to_numeric(df['S'], errors='coerce')
            df['V'] = pd.to_numeric(df['V'], errors='coerce')
            
            # Remove header rows or broken rows containing NaN strings
            df = df.dropna(subset=['H', 'S', 'V'])
            
            # Ensure the text label column is stripped of whitespaces and standardized
            df['Label'] = df['Label'].astype(str).str.strip().str.lower()
            
            # Concatenate current frame records into the master dataframe pool
            master_df = pd.concat([master_df, df], ignore_index=True)
            
        except Exception as e:
            print(f"Error parsing file {os.path.basename(file_path)}: {e}")

    if master_df.empty:
        print("No valid numeric HSV entries extracted from the CSV file cluster.")
        return

    unique_labels = master_df['Label'].unique()
    print(f"Extraction completed. Total samples: {len(master_df)} | Found unique color classes: {unique_labels}")

    # Set up the Matplotlib window and 3D environment axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Assign dynamic color tokens to each unique label class group using a qualitative colormap
    color_map = plt.colormaps.get_cmap('tab10')
    
    # Iterate over each category group to plot them as distinct isolated point clouds
    for idx, label in enumerate(unique_labels):
        # Isolate rows matching the specific text label string
        group_mask = master_df['Label'] == label
        group_df = master_df[group_mask]
        
        # Pull underlying coordinate matrices
        h_vals = group_df['H'].to_numpy()
        s_vals = group_df['S'].to_numpy()
        v_vals = group_df['V'].to_numpy()
        
        # Choose a static color point for this label array slice from the color map array
        assigned_color = color_map(idx % 10)
        
        # Plot this specific label cluster node group
        ax.scatter(
            h_vals, s_vals, v_vals, 
            label=f"Class: {label.upper()}", 
            color=assigned_color, 
            s=6, 
            alpha=0.7
        )

    # Set axis layout labels and design bounds matching OpenCV topology
    ax.set_xlabel('Hue (H)')
    ax.set_ylabel('Saturation (S)')
    ax.set_zlabel('Value (V)')
    ax.set_title('3D HSV Cloud Distribution Map grouped by Color Dataset Labels')
    
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 255)
    ax.set_zlim(0, 255)

    # Inject descriptive legend map box to identify label classes instantly
    ax.legend(loc='upper left', shadow=True, markerscale=2)

    print("Rendering labeled 3D color cloud view...")
    plt.show()
if __name__ == "__main__":
    # Change this path string to point to your specific CSV data directory folder
    target_folder_directory = r"D:\Van_Tien\Project\Camera\Input\Csv"
    
    load_and_plot_hsv_3d(target_folder_directory)