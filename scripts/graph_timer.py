##############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
##############################################################################

"""Steady-state kernel timing by CUDA-graph replay, shared by every bench.py.

The tutorial's kernels are power-bound on MI355X: their speed depends on what the GPU did in the
last tens of milliseconds to minutes. A timing therefore has to say which state it measures. This
module measures the back-to-back state, the one a kernel reaches when it runs continuously:

* ``d`` dispatches are captured in one CUDA graph and replayed back to back, with at most two
  replays queued, so the GPU never idles and no host launch sits between dispatches. ``d`` is
  chosen so a replay lasts about 20 ms; the CUDA-event pair around each replay then costs a few
  microseconds per ``d`` dispatches instead of per dispatch.
* Dispatch ``i`` reads rotating tensor set ``i % sets``, with enough sets that the inputs in
  rotation exceed the 256 MiB last-level cache (MALL) twice, so every dispatch reads its inputs
  from HBM.
* The replays first run for a warm-up time, then for a measurement window. The result is the
  median over the window's replays of replay time / ``d``, with the 10th and 90th percentiles and
  the drift between the window's two halves.

Two protocols are predefined (``--protocol``):

* ``quick`` (default): 1 s warm-up, 4 s window. Past the start-up transient (about 50 ms), but on
  a power-bound kernel that warms the board it can read up to about 1.5% faster than ``sustained``.
  Use it for A/B comparisons, alternating the two sides.
* ``sustained``: 60 s warm-up, 30 s window, the level a kernel holds indefinitely. Use it for
  published numbers.

The window's average socket power and effective GFX clock are reported when the driver exposes the
firmware metrics table (``/sys/bus/pci/devices/<bdf>/pm_metrics``, MI350 table version 0x14), read
once at each end of the window.

What this deliberately avoids, and why (measured on MI355X, a16w16 v9 4096x4096x8192 fp16):

* ``triton.testing.do_bench``: its 5-run estimate includes a one-time ~8 ms host stall on the first
  call, so its "25 ms" warm-up becomes ~3 ms and its timed runs fall inside the start-up dip; each
  timed run also pays ~5 us of event overhead (25% at 20 us kernels) and follows a 256 MiB cache
  flush that slows the kernel by ~3%. Together: 6 to 9% below the back-to-back time.
* ``AMD_SERIALIZE_KERNEL=3``: a 17 to 26 us host round trip before every dispatch changes the clock
  and the power state the kernel runs in; per-dispatch kernel time moved by -0.8 to +2.1%
  depending on the GPU, and small kernels ran on a GPU idle half the time.
* "last 100 of 1000 dispatches": a window 0.2 s after the start, before the board has warmed.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import struct
import time
from dataclasses import dataclass

TARGET_REPLAY_MS = 20.0
MIN_DISPATCHES = 16
MAX_DISPATCHES = 4096
COLD_INPUT_MIB = 512  # inputs in rotation: twice the 256 MiB MALL of MI300X/MI350
MAX_SETS = 64
IN_FLIGHT = 2  # replays queued ahead of the GPU


@dataclass(frozen=True)
class Protocol:
    name: str
    warmup_s: float
    window_s: float


PROTOCOLS = {
    "quick": Protocol("quick", 1.0, 4.0),
    "sustained": Protocol("sustained", 60.0, 30.0),
}
DEFAULT_PROTOCOL = "quick"


def add_timing_args(parser: argparse.ArgumentParser) -> None:
    """Add --protocol, --warmup-s, --window-s and --result-json to a bench.py parser."""
    g = parser.add_argument_group("timing (graph replay, see scripts/graph_timer.py)")
    g.add_argument(
        "--protocol",
        choices=sorted(PROTOCOLS),
        default=DEFAULT_PROTOCOL,
        help="quick: 1 s warm-up + 4 s window (A/B work); sustained: 60 s + 30 s (published "
        f"numbers). Default: {DEFAULT_PROTOCOL}",
    )
    g.add_argument("--warmup-s", type=float, default=None, help="override the warm-up time")
    g.add_argument("--window-s", type=float, default=None, help="override the window length")
    g.add_argument(
        "--result-json", default=None, help="append each result as one JSON line to this file"
    )


def protocol_from_args(args) -> Protocol:
    p = PROTOCOLS[args.protocol]
    warmup = p.warmup_s if args.warmup_s is None else args.warmup_s
    window = p.window_s if args.window_s is None else args.window_s
    if warmup < 0 or window <= 0:
        raise ValueError("--warmup-s must be >= 0 and --window-s > 0")
    name = p.name if (warmup, window) == (p.warmup_s, p.window_s) else "custom"
    return Protocol(name, warmup, window)


def rotating_sets(input_bytes_per_set: int, cold_input_mib: float = COLD_INPUT_MIB) -> int:
    """Rotating sets needed so the inputs in rotation exceed ``cold_input_mib`` (at least 2)."""
    need = math.ceil(cold_input_mib * 2**20 / max(1, input_bytes_per_set))
    return max(2, min(MAX_SETS, need))


# ----------------------------------------------------------------------------------- GPU state


def gpu_info(device) -> dict:
    import torch

    props = torch.cuda.get_device_properties(device)
    bdf = f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
    info = {
        "name": props.name,
        "arch": getattr(props, "gcnArchName", ""),
        "bdf": bdf,
        "visible_devices": os.environ.get("HIP_VISIBLE_DEVICES")
        or os.environ.get("CUDA_VISIBLE_DEVICES"),
        "power_cap_w": None,
    }
    for cap in glob.glob(f"/sys/bus/pci/devices/{bdf}/hwmon/hwmon*/power1_cap"):
        try:
            info["power_cap_w"] = int(open(cap).read()) / 1e6
        except (OSError, ValueError):
            pass
    return info


class PmMetrics:
    """Window averages of socket power and GFX clock from the firmware's metrics table.

    Only MetricsTable_t version 0x14 (MI350, smu_v13_0_12) is decoded; elsewhere ``ok`` is False.
    The table holds running totals: the energy (2^-16 J units), a clock accumulator per XCD that
    adds the clock (MHz x 1024) once per 1 ms firmware tick, the tick counter and a 10 ns timestamp,
    so two reads give exact averages over the time between them.
    """

    HEADER = struct.Struct(
        "<HHIII"
    )  # structure_size, pad, mp1 version, pmfw version, table version
    OFF = {"acc": 0, "ts": 52, "energy": 60, "clkacc": 168}

    def __init__(self, bdf: str):
        self.path = f"/sys/bus/pci/devices/{bdf}/pm_metrics"
        self.ok = False
        try:
            raw = self._raw()
            size, _, _, _, version = self.HEADER.unpack_from(raw, 0)
            self.ok = version == 0x14 and size >= 1284 + self.HEADER.size
        except (OSError, struct.error):
            pass

    def _raw(self) -> bytes:
        fd = os.open(self.path, os.O_RDONLY)
        try:
            return os.read(fd, 4096)
        finally:
            os.close(fd)

    def read(self):
        if not self.ok:
            return None
        b = self._raw()
        h = self.HEADER.size

        def u64(key, n=1):
            return struct.unpack_from(f"<{n}Q", b, h + self.OFF[key])

        return u64("acc")[0], u64("ts")[0], u64("energy")[0], u64("clkacc", 8)

    @staticmethod
    def average(r0, r1):
        if r0 is None or r1 is None or r1[0] <= r0[0]:
            return None, None
        seconds = (r1[1] - r0[1]) * 1e-8
        power = (r1[2] - r0[2]) * 2.0**-16 / seconds
        ticks = r1[0] - r0[0]
        clock = statistics.fmean((c1 - c0) / ticks / 1024 for c0, c1 in zip(r0[3], r1[3]))
        return power, clock


# ----------------------------------------------------------------------------------- timing


def _capture(launch, n, n_sets, stream):
    import torch

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        for i in range(n):
            launch(i % n_sets)
    return g


def time_dispatches(launch, n_sets: int, protocol: Protocol, *, flops=None, label="", device=None):
    """Time ``launch`` back to back by graph replay. Returns a result dict (see print_result).

    ``launch(slot)`` must enqueue exactly one kernel dispatch that reads rotating set ``slot`` on
    the current stream, without host synchronization (it is captured in a CUDA graph).
    """
    import torch

    device = device if device is not None else torch.cuda.current_device()
    # One dispatch per set outside the graph: compiles the kernel and touches every set.
    for slot in range(n_sets):
        launch(slot)
    torch.cuda.synchronize()

    stream = torch.cuda.Stream()
    # Estimate the dispatch time from a short graph's second replay.
    n0 = n_sets * math.ceil(MIN_DISPATCHES / n_sets)
    g0 = _capture(launch, n0, n_sets, stream)
    with torch.cuda.stream(stream):
        g0.replay()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        g0.replay()
        e.record()
    e.synchronize()
    est_us = s.elapsed_time(e) * 1e3 / n0
    del g0
    d = min(MAX_DISPATCHES, max(MIN_DISPATCHES, math.ceil(TARGET_REPLAY_MS * 1e3 / est_us)))
    d = n_sets * math.ceil(
        d / n_sets
    )  # a multiple of the sets, so one graph continues the rotation
    graph = _capture(launch, d, n_sets, stream)
    torch.cuda.synchronize()

    pm = PmMetrics(gpu_info(device)["bdf"])
    queue = []  # end events of queued replays

    def replay(timed):
        s = torch.cuda.Event(enable_timing=True) if timed else None
        e = torch.cuda.Event(enable_timing=timed)
        if s is not None:
            s.record()
        graph.replay()
        e.record()
        queue.append(e)
        if len(queue) > IN_FLIGHT:
            queue.pop(0).synchronize()  # keep the host at most IN_FLIGHT replays ahead
        return s, e

    rounds = []
    with torch.cuda.stream(stream):
        t0 = time.monotonic()
        while time.monotonic() - t0 < protocol.warmup_s:
            replay(False)
        s, e = replay(True)
        rounds.append((s, e))
        s.synchronize()  # the window starts when its first replay starts on the GPU
        pm0, w0 = pm.read(), time.monotonic_ns()
        while (time.monotonic_ns() - w0) / 1e9 < protocol.window_s:
            rounds.append(replay(True))
    rounds[-1][1].synchronize()
    pm1, w1 = pm.read(), time.monotonic_ns()
    torch.cuda.synchronize()

    us = [s.elapsed_time(e) * 1e3 / d for s, e in rounds]
    half = len(us) // 2
    q = statistics.quantiles(us, n=10) if len(us) >= 2 else [us[0]] * 9
    med = statistics.median(us)
    drift = (statistics.median(us[half:]) / statistics.median(us[:half]) - 1) * 100 if half else 0.0
    power, clock = PmMetrics.average(pm0, pm1)
    result = {
        "label": label,
        "protocol": protocol.name,
        "warmup_s": protocol.warmup_s,
        "window_s": protocol.window_s,
        "dispatches_per_replay": d,
        "replays": len(us),
        "sets": n_sets,
        "median_us": med,
        "p10_us": q[0],
        "p90_us": q[-1],
        "drift_pct": drift,
        "estimate_us": est_us,
        "tflops": (flops / (med * 1e-6) / 1e12) if flops else None,
        "flops": flops,
        "power_w": power,
        "clock_mhz": clock,
        "window_monotonic_ns": [w0, w1],
        "gpu": gpu_info(device),
    }
    return result


def print_result(result: dict, result_json: str | None = None) -> None:
    """One human line, then ``RESULT <json>`` for scripts (run_perf_table.py, collect_perf.py)."""
    r = result
    tf = f" ({r['tflops']:.0f} TFLOPS)" if r.get("tflops") else ""
    pw = f"; {r['power_w']:.0f} W, {r['clock_mhz']:.0f} MHz" if r.get("power_w") is not None else ""
    print(
        f"[{r['label']}] {r['median_us']:.2f} us/dispatch{tf}, p10-p90 {r['p10_us']:.2f}-"
        f"{r['p90_us']:.2f} us, drift {r['drift_pct']:+.2f}%: {r['replays']} replays x "
        f"{r['dispatches_per_replay']} dispatches over {r['window_s']:g} s after {r['warmup_s']:g} s "
        f"warm-up ({r['protocol']}), {r['sets']} rotating sets, GPU {r['gpu']['bdf']}{pw}",
        flush=True,
    )
    print("RESULT " + json.dumps(r), flush=True)
    if result_json:
        with open(result_json, "a") as f:
            f.write(json.dumps(r) + "\n")


def parse_results(text: str) -> list[dict]:
    """The RESULT lines of a bench.py run's output."""
    return [json.loads(line[7:]) for line in text.splitlines() if line.startswith("RESULT ")]


# ----------------------------------------------------------------------------------- profilers


def profile_loop(launch, n_sets: int, warmup_dispatches: int, iters: int) -> None:
    """Eager dispatches for an external profiler (rocprofv3 --pmc / --att / --kernel-trace).

    Issues ``warmup_dispatches`` and then ``iters`` dispatches back to back, set ``i % n_sets`` for
    dispatch ``i``; the first warm-up dispatch compiles the kernel. With a bench.py correctness
    check in front, the first dispatch after the warm-up is that kernel's dispatch number
    ``checks + warmup_dispatches + 1`` (1-based), which is what an ATT ``kernel_iteration_range``
    has to name to trace a dispatch in the back-to-back state.
    """
    import torch

    for i in range(warmup_dispatches + iters):
        launch(i % n_sets)
    torch.cuda.synchronize()


def warmup_dispatches_for(seconds: float, dispatch_us: float) -> int:
    """Dispatches that keep the GPU busy for ``seconds`` at ``dispatch_us`` each."""
    return max(0, math.ceil(seconds * 1e6 / max(dispatch_us, 1e-3)))


def trace_window_median(csv_path: str, kernel: str, t0_ns: int, t1_ns: int):
    """Median kernel-trace duration (us) of ``kernel`` dispatches that start inside [t0, t1].

    rocprofv3 timestamps share time.monotonic_ns's clock, so a RESULT's ``window_monotonic_ns``
    selects the dispatches of the measurement window. Returns (median_us, count).
    """
    import csv

    durations = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if kernel in row["Kernel_Name"]:
                start, end = int(row["Start_Timestamp"]), int(row["End_Timestamp"])
                if t0_ns <= start <= t1_ns:
                    durations.append((end - start) / 1e3)
    if not durations:
        return None, 0
    return statistics.median(durations), len(durations)
