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
        if level.value > LogLevel.DEBUG.value:
            self.log_debug = lambda *args, **kwargs: None
        else:
            self.log_debug = self._real_log_debug
        if level.value > LogLevel.INFO.value:
            self.log_info= lambda *args, **kwargs: None
        else:
            self.log_info= self._real_log_info


    def _log(self, level: LogLevel, message: str):
        if not isinstance(level, LogLevel):
            raise TypeError("Parameter 'level' must be an instance of LogLevel")
        print(f"[{level.name}][{self.name}] {message}")

    def _real_log_debug(self, message: str):
        self._log(LogLevel.DEBUG, message=message)

    def _real_log_info(self, message: str):
        self._log(LogLevel.INFO, message=message)

    def log_warning(self, message: str):
        self._log(LogLevel.WARNING, message=message)

    def log_error(self, message: str):
        self._log(LogLevel.ERROR, message=message)

    def log_critical(self, message: str):
        self._log(LogLevel.CRITICAL, message=message)
