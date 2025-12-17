import time
import numpy as np

# define the fastest model
def measure_latency(embed_fn, texts, runs=5):
    times = []

    for _ in range(runs):
        start = time.time()
        embed_fn(texts)
        times.append((time.time() - start) * 1000)

    return {
        "mean_ms": float(np.mean(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99))
    }
