import shutil
from my_logger      import Logger, LogLevel
from my_shutil      import ShellUtil
from dataclasses    import dataclass
from smbclient      import listdir, open_file, register_session

src_path = r"\\192.168.1.5\vserp_picture\WRK_SIGN"
des_path = r"\\192.168.1.5\vserp_picture\WRK_SIGN\Signature"


def main():
    logger = Logger("Main move file", LogLevel.DEBUG)
    shell_util = ShellUtil(LogLevel.DEBUG) 
    shell_util.set_root_folder(src_path,des_path)
    logger.log_info("Unitilize the shell successfully")

if __name__ == "__main__":
    main()
