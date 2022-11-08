# timer.py
# credits: https://realpython.com/python-timer/ (Geir Arne Hjelle)

import time
class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer(object):
    def __init__(self):
        self._start_time = None
        self.reset()

    def reset(self):
        self.clear()
        self.start()

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")
        self._start_time = time.perf_counter()

    def clear(self):
        """Clear the timer. Always succeeds"""
        self._start_time = None

    @property
    def elapsed_time(self):
        ret = time.perf_counter() - self._start_time
        return ret
