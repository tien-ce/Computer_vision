from enum import Enum
class LogLevel(Enum):
    DEBUG   =   1
    INFO    =   2
    WARNING =   3
    ERROR   =   4

CURRENT_LOG_LEVEL = LogLevel.DEBUG

def log(level: LogLevel, message: str):
    if not isinstance(level, LogLevel):
        raise TypeError(f"Parameter 'level' must be an instace of LogLevel")
    # Filtering logic remains the same
    if level.value >= CURRENT_LOG_LEVEL.value:
        # Modified formatting string to match '[LEVEL: ] message' format
        print(f"[{level.name}: ] {message}")
    
def setLevel(level: LogLevel):
    CURRENT_LOG_LEVEL = level