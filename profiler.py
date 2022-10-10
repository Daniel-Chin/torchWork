from time import perf_counter
from threading import Lock
from contextlib import contextmanager

class Profiler:
    def __init__(self) -> None:
        self.start = perf_counter()
        self.acc_good_time = 0
        self.lock = Lock()

    def goodTime(self):
        @contextmanager
        def f():
            start = perf_counter()
            try:
                yield None
            finally:
                dt = perf_counter() - start
                self.__accGoodTime(dt)
        
        return f()
    
    def __accGoodTime(self, dt):
        with self.lock:
            self.acc_good_time += dt
    
    def report(self):
        t = perf_counter() - self.start
        score = self.acc_good_time / t
        print('Profiler: Good time makes', format(score, '.0%'))
        return score
