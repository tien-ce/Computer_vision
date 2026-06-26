from enum import Enum

class LogLevel(Enum):
    DEBUG    = 1
    INFO     = 2
    WARNING  = 3
    ERROR    = 4
    CRITICAL = 5  # Fixed missing attribute from your original code

class Logger:
    def __init__(self, name: str, level: LogLevel = LogLevel.DEBUG):
        self.name = name
        self.current_log_level = level

    def _log(self, level: LogLevel, message: str):
        if not isinstance(level, LogLevel):
            raise TypeError("Parameter 'level' must be an instance of LogLevel")
        
        if level.value >= self.current_log_level.value:
            # Removed the extra space/colon formatting error from the original print
            print(f"[{level.name}][{self.name}] {message}")
        
    def setLevel(self, level: LogLevel):
        if not isinstance(level, LogLevel):
            raise TypeError("Parameter 'level' must be an instance of LogLevel")
        self.current_log_level = level

    def log_debug(self, message: str):
        self._log(LogLevel.DEBUG, message=message)

    def log_info(self, message: str):
        self._log(LogLevel.INFO, message=message)

    def log_warning(self, message: str):
        self._log(LogLevel.WARNING, message=message)

    def log_error(self, message: str):
        self._log(LogLevel.ERROR, message=message)

    def log_critical(self, message: str):
        self._log(LogLevel.CRITICAL, message=message)