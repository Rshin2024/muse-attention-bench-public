"""CUDA graph measurement primitives. Importable without CUDA/Torch installed."""
from __future__ import annotations

import statistics


def replay_orders(repeats: int) -> list[str]:
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 4 or repeats % 2:
        raise ValueError("timing repeats must be even and at least four")
    return ["AB" if n % 2 == 0 else "BA" for n in range(repeats)]


class ColdL2:
    """Read a buffer larger than H200 L2 immediately before every timed replay."""
    def __init__(self, torch, byte_count: int):
        # H200 has 50 MiB L2. Do not accept a buffer that could fit entirely in L2.
        if byte_count < 64 * 1024 * 1024:
            raise ValueError("cold-L2 buffer must be at least 64 MiB")
        self.torch = torch
        self.buffer = torch.zeros(byte_count, dtype=torch.uint8, device="cuda")
        self.sink = torch.empty((), dtype=torch.int64, device="cuda")

    def read(self):
        self.torch.sum(self.buffer, dim=0, dtype=self.torch.int64, out=self.sink)


def event_replay_us(torch, graph, cold: ColdL2) -> float:
    # Allocation/setup precedes the flush and event region. Exactly one replay.
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    cold.read()
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) * 1000.0


def calibrate_graph_cost(torch, cold: ColdL2, repeats: int = 50) -> dict:
    """Measure an empty captured graph with the same CUDA-event bracket.

    The organizer's exact fixed-cost calibration is unpublished. This local
    surrogate is measured, never replaced with the approximate documented 4 us.
    """
    if repeats < 10:
        raise ValueError("at least ten graph-cost calibration samples are required")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        pass
    graph.replay()
    torch.cuda.synchronize()
    values = [event_replay_us(torch, graph, cold) for _ in range(repeats)]
    median = statistics.median(values)
    if median < 0:
        raise ValueError("negative graph overhead calibration")
    return {"method": "empty CUDA graph, identical cold-L2/event bracket",
            "median_us": median, "samples_us": values,
            "official_calibration_algorithm_known": False}


def capture(torch, function, data, warmup: int = 2):
    if warmup < 1:
        raise ValueError("at least one warmup is required for lazy compilation")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            function(data)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = function(data)
    torch.cuda.synchronize()
    return graph, output
