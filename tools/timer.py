# timer.py

from dataclasses import dataclass, field
import time
from datetime import datetime
from typing import Callable, ClassVar, Dict, Optional


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


@dataclass
class Timer:
    timers: ClassVar[Dict[str, float]] = {}
    name: Optional[str] = None
    text: str = "Elapsed time: {:0.4f} seconds"
    logger: Optional[Callable[[str], None]] = print
    _start_time: Optional[float] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Add timer to dict of timers after initialization"""
        if self.name is not None:
            self.timers.setdefault(self.name, 0)

    def start(self) -> None:
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError("Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError("Timer is not running. Use .start() to start it")

        # Calculate elapsed time
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        # Report elapsed time
        if self.logger:
            self.logger(self.text.format(elapsed_time))
        if self.name:
            self.timers[self.name] += elapsed_time

        return elapsed_time

    @classmethod
    def current_time_str(self):
        # 获取当前时间
        current_time = datetime.now()

        # 将时间格式化为年月日时分秒
        # %Y 表示四位数的年份。
        # %m 表示两位数的月份。
        # %d 表示两位数的日期。
        # %H 表示两位数的小时（24小时制）。
        # %M 表示两位数的分钟。
        # %S 表示两位数的秒。
        current_datetime_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
        return current_datetime_str
    
    @classmethod
    def current_date_str(self):
        # 获取当前时间
        current_time = datetime.now()

        # 将时间格式化为字符串
        current_date_str = current_time.strftime("%Y_%m_%d")
        return current_date_str
