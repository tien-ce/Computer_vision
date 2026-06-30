import shutil
from my_logger      import Logger, LogLevel
from dataclasses    import dataclass
from smbclient      import listdir, open_file, register_session

       
@dataclass
class Shared_Info:
    server: str
    folder: str
    username = str
    password = str

@dataclass
class FileInfo:
    src_folder: str
    des_folder: str
    shared_info: Shared_Info

class ShellUtil:
    def __init__(self, log_level: LogLevel = LogLevel.DEBUG):
        self.logger = Logger ("Shell utils", log_level)
        self.root_folder = None

    def _parse_unc_path(self,unc_path):
        parts = unc_path.split('\\')
        # parts[0]: Empty, parts[1]: Empty because split on //
        # parts[2] is IP, Part[3] is folder, the rest of subfolders
        if len(parts) < 4:
           self.logger.log_error(f"Part length is less than 4: unc_path: {unc_path}")
           return None, None, None
        ip = parts[2]
        share = parts[3]
        subfolders = ""
        if len(parts) > 4:
           subfolders = parts[4].join(parts[5:])
        return ip, share, subfolders

    def set_root_folder(self,src_folder: str, des_folder: str):
        self.logger.log_debug(f"Set root folder: Source folder {src_folder}, Destination: {des_folder}")
        if src_folder.startswith(r"\\192.168") and des_folder.startswith(r"\\192.168"):
            self.logger.log_debug("Using network shared folder")
            src_ip, src_share, src_subfolders = self._parse_unc_path(src_folder)
            des_ip, des_share, des_subfolders = self._parse_unc_path(des_folder)
            self.logger.log_debug(f"Src [{src_ip}, {src_share}], Des [{des_ip}, {des_share}]")
            if src_ip != des_ip or src_share != des_share:
                self.logger.log_error(f"Srouce and destination is not match: Src {src_folder}, Des {des_folder}")
                raise ValueError("Source and destination is through network but infomation not match")
            self.logger.log_debug(f"ip: {src_ip}, share: {src_share}")
        self.root_folder = FileInfo(src_folder, des_folder, None)

