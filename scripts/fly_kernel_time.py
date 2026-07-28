#!/usr/bin/env python3
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

"""Time ROCm/FlyDSL's flash attention with the protocol `fa_kernel_time.py` uses.

The comparison in `kernels/attention/README.md` §8 is only meaningful if both sides are
measured the same way, and the default FlyDSL benchmark is not: a shallower averaging
window leaves the kernel in its thermal transient, so the reported number depends on how
hot the die was when the run started. Six consecutive runs of one config measured 1236.9,
1242.8, 1165.9, 1167.7, 1159.3 and 1157.5 TFLOPS. This script instead does what
`fa_kernel_time.py` does -- `rocprofv3 --kernel-trace` with `AMD_SERIALIZE_KERNEL=3`, and
the mean of the **last 100 of 1000** dispatches -- so the window sits in steady state.

Needs a FlyDSL checkout; pass `--flydsl-root` or set `FLYDSL_ROOT`. Numbers in the README
were taken at ROCm/FlyDSL `63eb891` (v0.2.4-26-g63eb891).

    python scripts/fly_kernel_time.py --seqlen 16384
    python scripts/fly_kernel_time.py --seqlen 16384 --causal 1     # ~half the FLOPs

`--eager-rescale` selects the fav3-equivalent path; the remaining flags expose the
builder's other performance knobs, all of which default to FlyDSL's own tuned values (see
`FLASH_ATTN_FUNC_KERNEL_CONFIG` in its `tests/kernels/test_flash_attn_fwd.py`).
"""

import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KERNEL = "flash_attn_dualwave_swp"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--flydsl-root", default=os.environ.get("FLYDSL_ROOT", "/root/FlyDSL"))
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--hq", type=int, default=64)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--seqlen", type=int, default=16384)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--last-n", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--rotating-buffer-size", type=int, default=512)
    p.add_argument("--causal", type=int, default=0)
    p.add_argument("--eager-rescale", action="store_true",
                   help="dualwave_swp_lazy_rescale=False, the fav3-equivalent path")
    p.add_argument("--setprio", type=int, default=1)
    p.add_argument("--stagger", type=int, default=1)
    p.add_argument("--waves-per-eu", type=int, default=2)
    p.add_argument("--daz", type=int, default=1)
    p.add_argument("--_inner", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def dispatch_loop(a):
    """The part that runs under rocprofv3."""
    import torch

    sys.path.insert(0, a.flydsl_root)
    # `FLY_SCHED_VALU` re-schedules the in-loop VALU of FlyDSL's own assembly and runs the
    # kernel from the result -- see scripts/fly_sched.py. Installed before the module is
    # built, since it replaces the compile step.
    if os.environ.get("FLY_SCHED_VALU"):
        sys.path.insert(0, HERE)
        import fly_sched  # noqa: E402

        fly_sched.patch()
    from kernels.attention.flash_attn_gfx950 import build_flash_attn_dualwave_swp_module

    dt = torch.float16 if a.dtype == "fp16" else torch.bfloat16
    launch = build_flash_attn_dualwave_swp_module(
        num_heads=a.hq, head_dim=a.d, causal=bool(a.causal),
        dtype_str="f16" if a.dtype == "fp16" else "bf16",
        waves_per_eu=a.waves_per_eu, daz=bool(a.daz),
        dualwave_swp_lazy_rescale=not a.eager_rescale,
        dualwave_swp_setprio=bool(a.setprio),
        dualwave_swp_enable_stagger=bool(a.stagger))

    # Match bench.py: cycle enough (q,k,v,o) sets that the loop's footprint exceeds cache.
    elem = torch.empty(0, dtype=dt).element_size()
    per_set = 4 * a.batch * a.hq * a.seqlen * a.d * elem
    n_sets = max(1, -(-(a.rotating_buffer_size * 1024 * 1024) // per_set))
    sets = []
    for _ in range(n_sets):
        mk = lambda: torch.randn(a.batch, a.seqlen, a.hq, a.d, dtype=dt, device="cuda")
        q, k, v = mk(), mk(), mk()
        sets.append((q, k, v, torch.empty_like(q)))

    # Check correctness once, so a fast-but-wrong build cannot be reported as a result.
    q, k, v, o = sets[0]
    launch(q, k, v, o, a.batch, a.seqlen)
    torch.cuda.synchronize()
    qr, kr, vr = (t.transpose(1, 2) for t in (q, k, v))
    ref = torch.nn.functional.scaled_dot_product_attention(
        qr.float(), kr.float(), vr.float(), is_causal=bool(a.causal), scale=a.d ** -0.5)
    err = (o.transpose(1, 2).float() - ref).abs().max().item()
    print(f"[FlyDSL] causal={bool(a.causal)} lazy={not a.eager_rescale} "
          f"max_err={err:.2e} {'OK' if err < 1e-2 else 'MISMATCH'}", flush=True)
    del ref, qr, kr, vr
    torch.cuda.empty_cache()

    for i in range(a.warmup):
        q, k, v, o = sets[i % n_sets]
        launch(q, k, v, o, a.batch, a.seqlen)
    torch.cuda.synchronize()
    for i in range(a.iters):
        q, k, v, o = sets[i % n_sets]
        launch(q, k, v, o, a.batch, a.seqlen)
    torch.cuda.synchronize()


def main():
    a = parse_args()
    if a._inner:
        dispatch_loop(a)
        return

    sys.path.insert(0, HERE)
    from run_perf_table import avg_kernel_time_ns, find_kernel_trace_csv

    trace = os.path.join(HERE, f".fly_kt_{os.getpid()}")
    shutil.rmtree(trace, ignore_errors=True)
    cmd = ["rocprofv3", "--kernel-trace", "-f", "csv", "--kernel-include-regex", KERNEL,
           "-d", trace, "--", sys.executable, os.path.abspath(__file__), "--_inner"]
    cmd += [x for x in sys.argv[1:] if x != "--_inner"]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          env={**os.environ, "AMD_SERIALIZE_KERNEL": "3"})
    for line in proc.stdout.splitlines():
        if "[FlyDSL]" in line:
            print("   ", line.strip())

    csv = find_kernel_trace_csv(trace)
    if not csv:
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-5:]
        sys.exit("fly_kernel_time: no kernel trace produced:\n  " + "\n  ".join(tail))
    avg_ns, count = avg_kernel_time_ns(csv, KERNEL, last_n=a.last_n)
    shutil.rmtree(trace, ignore_errors=True)

    # Causal skips roughly half the tiles, so charge it half the FLOPs.
    flops = 2 * (2.0 * a.batch * a.hq * a.seqlen * a.seqlen * a.d)
    if a.causal:
        flops *= 0.5
    print(f"    {count} dispatches, final-{a.last_n} avg={avg_ns / 1e3:.2f} us "
          f"-> {flops / avg_ns * 1e-3:.1f} TFLOPS")


if __name__ == "__main__":
    main()
