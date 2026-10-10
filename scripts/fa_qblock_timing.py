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
"""Per-q-block time breakdown of the persistent FA kernel (fmha_v5), from in-kernel clocks.

With FA_WG_TIMING=1 the kernel stores s_memrealtime (100 MHz) at fixed points of every
q-block. This runs a few launches, then fits each phase as ``a + b * tiles`` over the
q-blocks (a causal launch has blocks of 4 .. S/64 tiles), which separates a q-block's fixed
cost from its per-tile cost. The stamps are global stores, so they perturb the kernel a
little -- read the output as a breakdown and time with fa_kernel_time.py.

    FA_MODULE=fmha_v5 python scripts/fa_qblock_timing.py --causal

Same environment as bench.py (LLVM_PASS_PLUGIN_PATH, DISABLE_LLVM_OPT).
"""

import argparse
import importlib
import os
import sys

ATTENTION_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kernels", "attention"
)
sys.path.insert(0, ATTENTION_DIR)
if os.environ.get("LLVM_PASS_PLUGIN_PATH"):
    sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from common import input_helper  # noqa: E402

# fmha_v5's stamp points (NSLOT = 8, the last unused)
PHASES = ["prologue", "plain loop", "masked loop", "drain + prefetch", "last PV", "epilogue"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--hq", type=int, default=8)
    ap.add_argument("--seqlen", type=int, default=8192)
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    os.environ["FA_WG_TIMING"] = "1"
    mod = importlib.import_module(os.environ.get("FA_MODULE", "fmha_v5"))
    q, k, v, md = input_helper(a.batch, a.hq, a.hq, a.seqlen, a.seqlen, 128, torch.bfloat16, "bhsd")
    md.causal = a.causal
    o = torch.empty_like(q)
    for _ in range(a.reps):
        mod.run_gluon_attention(q, k, v, o, md)
    torch.cuda.synchronize()

    T = mod.LAST_TIMING.cpu().numpy()[:, :, 0].astype(np.float64)
    T = T[T[:, 0] > 0]  # q-blocks this launch ran (the buffer is sized per workgroup)
    T = (T - T[:, 0].min()) / 100.0  # 100 MHz ticks -> us
    nwg = mod.num_cus(q.device)
    jobs = a.batch * a.hq * (a.seqlen // (512 if a.causal else 256))
    nwg = min(nwg, jobs) - min(nwg, jobs) % 8
    per = T.shape[0] // nwg
    nm = a.seqlen // 256
    wg = np.arange(nwg)
    wgx = (wg % 8) * (nwg // 8) + wg // 8
    tiles = np.zeros((nwg, per))
    for t in range(per):
        if a.causal:
            jj = ((t // 2) * nwg + wgx) % (nm // 2)
            sm = np.where(t % 2 == 0, nm - 1 - jj, jj)
            tiles[:, t] = (sm + 1) * 4
        else:
            tiles[:, t] = a.seqlen // 64
    tiles = tiles.ravel()

    dur = T[:, 6] - T[:, 0]
    print(f"kernel span {T[:, 6].max():.1f} us over {T.shape[0]} q-blocks")

    def fit(y):
        if len(set(tiles.tolist())) < 2:
            return y.mean(), 0.0
        c, *_ = np.linalg.lstsq(np.vstack([np.ones_like(tiles), tiles]).T, y, rcond=None)
        return c

    c = fit(dur)
    print(f"q-block: {c[0]:7.2f} us + {c[1]:.4f} us/tile (mean {dur.mean():.2f} us)")
    for i, name in enumerate(PHASES):
        d = T[:, i + 1] - T[:, i]
        c = fit(d)
        print(f"  {name:<18} mean {d.mean():8.2f} us   fit {c[0]:7.2f} + {c[1]:.4f}/tile")


if __name__ == "__main__":
    main()
