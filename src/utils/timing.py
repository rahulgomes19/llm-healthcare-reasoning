from __future__ import annotations

import time


def timed_call(fn, *args, **kwargs):
    start = time.time()
    result = fn(*args, **kwargs)
    return result, time.time() - start
