import os
import datetime
import logging
from typing import Optional, List

class MyFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        return datetime.datetime.fromtimestamp(record.created).strftime('%m-%d-%H:%M:%S')

def setup_logger(
    name: str = None,
    log_file: Optional[str] = None,
    file_level: int = logging.DEBUG,
    handlers: Optional[List[logging.Handler]] = None
) -> logging.Logger:
    """
    params:
        name (str): Logger name, default: current module name
        log_file (Optional[str]): log path
        format_str (str): format: ts, logger name, level, message
    """
    if name is None:
        logger = logging.getLogger()
    else:
        logger = logging.getLogger(name)

    logger.setLevel(logging.DEBUG)

    if not handlers:
        handlers = []

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        handlers.append(console_handler)

        if log_file is not None:
            if not os.path.exists(log_file):
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
        else:
            log_file = f"{os.path.dirname(os.path.abspath(__file__))}/log/{name}.log"
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(file_level)
        handlers.append(file_handler)

    format_str= "%(asctime)s %(name)s %(levelname)s] %(message)s"
    formatter = MyFormatter(format_str)
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def init_logger(name: str):
    return logging.getLogger(name)

def get_logger():
    return logging.getLogger("MxMoE")