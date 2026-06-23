import os
import sys
import shutil # shell utility
import my_logger as logger
from my_logger import LogLevel
logger.setLevel(LogLevel.DEBUG)
# Destination path
TARGET_SITE_PACKAGES = "../cpv_env/Lib/site-packages"

def move_to_site_packages(source_folder: str,*filesnames: str):
    """
    Moves specific files from a source folder into the hardcoded site-packages directory.
    
    :param source_folder: Path to the directory containing the files
    :param filenames: Arbitrary number of file names to move
    """

    # 1. Verify the source folder exists
    if not os.path.isdir(source_folder):
        logger.log_error(f"Source folder not found -> {source_folder}")
        return

    # 2. Verify the haredcode destination folder exists
    if not os.path.isdir(TARGET_SITE_PACKAGES):
        logger.log_error(f"Destination site-packages not found -> '{TARGET_SITE_PACKAGES}'")
        return
    
    logger.log_info(f"Starting transfer from: {source_folder}")
    logger.log_info(f"Moving to: {TARGET_SITE_PACKAGES}\n" + "-"*40)

    # 3. Loop through all arbitrary file arguments
    for filename in filesnames:
        # Construct full path to source file
        source_file_path = os.path.join(source_folder,filename)
        logger.log_debug(f"Source file path: {source_file_path}")
        # Construct full path to distination file
        destination_file_path = os.path.join(TARGET_SITE_PACKAGES,filename)

        # Check if file is existing
        if os.path.isfile(source_file_path):
            try:
                shutil.move(source_file_path, destination_file_path)
                logger.log_info(f"Succesfully moved: {source_file_path} --> {destination_file_path}")
            except Exception as e:
                logger.log_error(f"Failed to move {source_file_path} --> {destination_file_path}")
        else:
            logger.log_warning(f"File is not existing at: {source_file_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        logger.log_info(f"Usage: python move_tools.py <source_folder> <file1> [file2] [file3] ...")
    else:
        src_dir = sys.argv[1]
        files_to_move = sys.argv[2:]
        move_to_site_packages(src_dir, *files_to_move)