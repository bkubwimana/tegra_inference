import time
from contextlib import contextmanager

class LatencyResult:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.elapsed = end - start

@contextmanager
def record_latency():
    start_time = time.time()
    result = LatencyResult(start_time, start_time)
    try:
        yield result
    finally:
        result.end = time.time()
        result.elapsed = result.end - result.start