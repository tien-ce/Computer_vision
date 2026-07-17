from enum import Enum

# ==============================================================================
# Terminal escape codes used to color log output
# ==============================================================================
class TerminalColor:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"

# ==============================================================================
# Severity level used to gate what a Logger instance prints
# ==============================================================================
class LogLevel(Enum):
    DEBUG    = 1
    INFO     = 2
    WARNING  = 3
    ERROR    = 4
    CRITICAL = 5  # Fixed missing attribute from your original code

# ==============================================================================
# Logger
# ==============================================================================
class Logger:
    def __init__(self, name: str, level: LogLevel = LogLevel.DEBUG):
        # --- Identity / configured level ---
        self.name = name
        self.current_log_level = level

        # --- Gate debug/info methods to no-ops when below the configured level ---
        if level.value > LogLevel.DEBUG.value:
            self.log_debug = lambda *args, **kwargs: None
        else:
            self.log_debug = self._real_log_debug
        if level.value > LogLevel.INFO.value:
            self.log_info = lambda *args, **kwargs: None
        else:
            self.log_info = self._real_log_info

    # --- Core print primitive ---
    def _log(self, level: LogLevel, message: str):
        if not isinstance(level, LogLevel):
            raise TypeError("Parameter 'level' must be an instance of LogLevel")
        print(f"[{level.name}][{self.name}] {message}")

    # --- Gated methods (see __init__) ---
    def _real_log_debug(self, message: str):
        self._log(LogLevel.DEBUG, message=f"{TerminalColor.GREEN}{message}{TerminalColor.RESET}")

    def _real_log_info(self, message: str):
        self._log(LogLevel.INFO, message=message)

    # --- Always-on methods ---
    def log_warning(self, message: str):
        self._log(LogLevel.WARNING, message=f"{TerminalColor.YELLOW}{message}{TerminalColor.RESET}")

    def log_error(self, message: str):
        self._log(LogLevel.ERROR, message=f"{TerminalColor.RED}{message}{TerminalColor.RESET}")

    def log_critical(self, message: str):
        self._log(LogLevel.CRITICAL, message=message)
