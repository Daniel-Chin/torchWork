from time import perf_counter
from threading import Lock
from contextlib import contextmanager

class Profiler:
    def __init__(self) -> None:
        self.start = perf_counter()
        self.acc = {}
        self.current_tags = set()
        self.lock = Lock()

    @contextmanager
    def __call__(self, *tags: str):
        with self.lock:
            intersection = self.current_tags.intersection(tags)
            if intersection:
                raise ValueError(f'Cannot enter twice: {intersection}')
            self.current_tags.update(tags)
        start = perf_counter()
        try:
            yield None
        finally:
            dt = perf_counter() - start
            self.__accTime(tags, dt)
            with self.lock:
                self.current_tags.difference_update(tags)
    
    def __accTime(self, tags, dt):
        with self.lock:
            for tag in tags:
                if tag not in self.acc:
                    self.acc[tag] = 0
                self.acc[tag] += dt
    
    def report(self):
        print('Profiler:')
        with self.lock:
            T = perf_counter() - self.start
            for tag, t in self.acc.items():
                print(' ', tag, '\t', format(t / T, '3.0%'))
        print(flush=True)
