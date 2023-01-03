from time import perf_counter, time, sleep
from threading import Lock, Thread
from contextlib import contextmanager
from typing import Optional

import torch
from tabulate import tabulate

from torchWork import HAS_CUDA, DEVICE

class Profiler:
    def __init__(self) -> None:
        self.start = perf_counter()
        self.acc = {}
        self.current_tags = set()
        self.lock = Lock()

        self.last_report = None
        
        if HAS_CUDA:
            print('Syncing GPU...', flush=True)
            a = torch.zeros((3, ), device=DEVICE, requires_grad=True)
            b = a + 3
            b.sum().backward()
            torch.cuda.synchronize()

    @contextmanager
    def __call__(self, *tags: str):
        with self.lock:
            intersection = self.current_tags.intersection(tags)
            if intersection:
                raise ValueError(f'Cannot enter twice: {intersection}')
            self.current_tags.update(tags)
        if HAS_CUDA:
            torch.cuda.synchronize()
        start = perf_counter()
        try:
            yield None
        finally:
            if HAS_CUDA:
                torch.cuda.synchronize()
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
    
    def report(self, throttle: Optional[int] = 3):
        if throttle is not None:
            if self.last_report is None:
                pass
            elif time() - self.last_report >= throttle:
                pass
            else:
                return
        self.last_report = time()
        print('Profiler:')
        with self.lock:
            T = perf_counter() - self.start
            table = [('', tag, t / T) for tag, t in self.acc.items()]
        print(tabulate(table, floatfmt='3.0%', tablefmt='plain'), flush=True)

class GPUUtilizationReporter(Thread):
    def __init__(self) -> None:
        super().__init__()
        self.go_on = True
    
    def run(self):
        while self.go_on:
            sleep(1/6)
            print(
                'GPU utilization: ', 
                torch.cuda.utilization(), '%', sep='', 
            )
    
    def close(self):
        self.go_on = False
