import inspect
from datetime import datetime
from enum import Enum


class LogLevel(Enum):
    DEBUG = 1
    INFO = 2
    WARN = 3


_LOGLEVELCOLORS = {
    LogLevel.DEBUG: "\033[94m",  # Blue
    LogLevel.INFO: "\033[92m",  # Green
    LogLevel.WARN: "\033[93m",  # Yellow
}

_RESETCOLOR = "\033[0m"

_DEBUGENABLED = True

_LOGFILE = None

_LOGINCOLOR = False


def set_log(file=None, debug=False, color=False):
    global _LOGFILE
    global _DEBUGENABLED
    global _LOGINCOLOR
    _LOGINCOLOR = color
    _DEBUGENABLED = debug
    # NOTE We will append to a file that already exists!
    _LOGFILE = file


def myprint(level, timestamp, caller, *args):
    # Get color
    color = _LOGLEVELCOLORS.get(level, _RESETCOLOR)
    # Construct message
    msg = f"[ZRL][{timestamp}][{level.name:5s}][{caller}] {' '.join(str(arg) for arg in args)}"
    if _LOGINCOLOR:
        print(f"{color}{msg}{_RESETCOLOR}")
    else:
        # Color is disabled
        print(msg)
    if _LOGFILE:
        # Log to file
        with open(_LOGFILE, "a") as f:
            f.write(f"{msg}\n")


def debug(*msg):
    if not _DEBUGENABLED:
        return
    # timestamp = datetime.now().strftime("%H:%M:%S")
    # ZD, avoid collision
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    fn = inspect.stack()[1].function
    myprint(LogLevel.DEBUG, timestamp, fn, *msg)


def warn(*msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    fn = inspect.stack()[1].function
    myprint(LogLevel.WARN, timestamp, fn, *msg)


def info(*msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    fn = inspect.stack()[1].function
    myprint(LogLevel.INFO, timestamp, fn, *msg)
