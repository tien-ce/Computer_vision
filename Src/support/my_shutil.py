import os
import shutil

from smbclient import listdir, open_file, register_session, reset_connection_cache, rename
from dotenv import load_dotenv

from my_logger import Logger, LogLevel

# Load environmental variables for credentials
load_dotenv()

class ShellUtil:
    def __init__(self, log_level: LogLevel = LogLevel.DEBUG):
        """
        Initialize the ShellUtil instance with logging and state control.
        """
        # --- State ---
        self.logger = Logger("Shell utils", log_level)
        self.root_folder = None
        self.is_network = False
        self.ip = None

    # --- Path helpers ---
    def _parse_unc_path(self, unc_path: str) -> str:
        """
        Extract the host or IP address from a network UNC path.
        """
        parts = [p for p in unc_path.split('\\') if p]
        if len(parts) < 2:
            self.logger.log_error(f"Invalid UNC path structure: {unc_path}")
            return None
        return parts[0]

    def set_root_folder(self, root_folder: str):
        """
        Determine if the root folder is local or a network share, then setup configurations.
        """
        self.root_folder = root_folder

        # Check if the root folder uses a network UNC path format
        if root_folder.startswith(r"\\"):
            self.is_network = True
            self.ip = self._parse_unc_path(root_folder)
            
            if not self.ip:
                raise ValueError(f"Could not parse IP/Host from path: {root_folder}")
            
            user_name = os.getenv("USER_NAME")
            pass_word = os.getenv("PASS_WORD")
            
            # Authenticate session using smbclient
            register_session(self.ip, username=user_name, password=pass_word)
            self.logger.log_info(f"Network path detected. Session registered for IP: {self.ip}")
        else:
            self.is_network = False
            if not os.path.isdir(root_folder):
                self.logger.log_error(f"Path is not a valid directory: {root_folder}")
                raise ValueError("Path is not a directory")
            self.logger.log_info(f"Local path detected: {root_folder}")

    def _get_absolute_path(self, relative_path: str) -> str:
        """
        Combine the established root folder with a relative file or directory path.
        """
        if not self.root_folder:
            raise ValueError("Root folder is not set. Call set_root_folder first.")
        return os.path.join(self.root_folder, relative_path) if relative_path else self.root_folder

    # --- Read operations ---
    def list_directory(self, relative_path: str = "") -> list:
        """
        List all contents of a specific directory path relative to the root folder.
        """
        target_path = self._get_absolute_path(relative_path)
        if self.is_network:
            return listdir(target_path)
        else:
            return os.listdir(target_path)

    def read_file(self, relative_path: str, mode: str = "r"):
        """
        Read the entire content of a file located relative to the root folder.
        """
        target_path = self._get_absolute_path(relative_path)
        if self.is_network:
            with open_file(target_path, mode=mode) as f:
                content = f.read()
            self.logger.log_debug(f"Successfully read from network path: {target_path}")
            return content
        else:
            encoding = "utf-8" if "b" not in mode else None
            with open(target_path, mode=mode, encoding=encoding) as f:
                content = f.read()
            self.logger.log_debug(f"Successfully read from local path: {target_path}")
            return content

    def read_lines(self, relative_path: str, mode: str = 'r'):
        """
        Read the entire content of a file located relative to the root folder 
        and save it into lines.
        """
        target_path = self._get_absolute_path(relative_path)
        if self.is_network:
            with open_file(target_path, mode=mode) as f:
                content = f.readlines()
            self.logger.log_debug(f"Successfully read from network path: {target_path}")
            return content
        else:
            encoding = "utf-8" if "b" not in mode else None
            with open(target_path, mode=mode, encoding=encoding) as f:
                content = f.readlines()
            self.logger.log_debug(f"Successfully read from local path: {target_path}")
            return content

    # --- Write operations ---
    def write_file(self, relative_path: str, message: str, mode: str = "a"):
        """
        Write or append data to a file located relative to the root folder.
        """
        target_path = self._get_absolute_path(relative_path)
        if self.is_network:
            with open_file(target_path, mode=mode) as f:
                f.write(message)
            self.logger.log_debug(f"Successfully written to network path: {target_path}")
        else:
            encoding = "utf-8" if "b" not in mode else None
            with open(target_path, mode=mode, encoding=encoding) as f:
                f.write(message)
            self.logger.log_debug(f"Successfully written to local path: {target_path}")

    def write_file_ln(self, relative_path: str, message: str, mode: str = "a"):
        """
        Write or append data to a file located relative to the root folder
        with end line.
        """
        self.write_file(relative_path,f"{message}\n",mode = mode)
    def move_file(self, src_relative: str, dst_relative: str):
        """
        Move or rename a file within the context of the root directory.
        """
        src_path = self._get_absolute_path(src_relative)
        dst_path = self._get_absolute_path(dst_relative)
        
        if self.is_network:
            rename(src_path, dst_path)
        else:
            shutil.move(src_path, dst_path)
        self.logger.log_debug(f"Moved file from {src_path} -> {dst_path}")

    # --- Lifecycle ---
    def stop(self):
        """
        Clear connection cache and unregister network session if applicable.
        """
        if self.is_network and self.ip:
            self.logger.log_info(f"Unregistering network session for IP: {self.ip}")
            reset_connection_cache(self.ip)
