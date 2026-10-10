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
"""Check an FA kernel's O and LSE against fp32 references, causal and not, and that
repeated launches are bit-identical (a race between the async K/V copies and the LDS
reads shows up as run-to-run drift long before it fails a tolerance).

    FA_MODULE=fmha_v5 python scripts/fa_check.py --seqlens 1024,2048,4096,8192

The module must expose LAST_LSE (the log-sum-exp of its last launch). O is compared over
the whole batch, LSE on three (batch, head) slices (it needs the full S x S scores).
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

import torch  # noqa: E402
from common import _check_output, input_helper  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqlens", default="1024,2048,8192")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--hq", type=int, default=8)
    ap.add_argument("--modes", default="causal,noncausal")
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    mod = importlib.import_module(os.environ.get("FA_MODULE", "fmha_v5"))
    ok_all = True
    for S in [int(x) for x in a.seqlens.split(",")]:
        for mode in a.modes.split(","):
            causal = mode == "causal"
            q, k, v, md = input_helper(a.batch, a.hq, a.hq, S, S, 128, torch.bfloat16, "bhsd")
            md.causal = causal
            runs = []
            for _ in range(a.reps):
                o = torch.empty_like(q)
                mod.run_gluon_attention(q, k, v, o, md)
                torch.cuda.synchronize()
                runs.append((o.clone(), mod.LAST_LSE.clone()))
            o, lse = runs[0]
            det = all(torch.equal(o, r[0]) and torch.equal(lse, r[1]) for r in runs[1:])
            ref = torch.nn.functional.scaled_dot_product_attention(
                q.float(), k.float(), v.float(), is_causal=causal, scale=md.sm_scale
            )
            ok_o, max_o, _ = _check_output(o, ref)
            lse_err = 0.0
            for b, h in [(0, 0), (a.batch - 1, a.hq - 1), (0, a.hq // 2)]:
                s = (q[b, h].float() @ k[b, h].float().T) * md.sm_scale
                if causal:
                    keep = torch.tril(torch.ones(S, S, dtype=torch.bool, device=s.device))
                    s = s.masked_fill(~keep, float("-inf"))
                lse_err = max(lse_err, (lse[b, h] - torch.logsumexp(s, dim=-1)).abs().max().item())
            ok = ok_o and lse_err < 1e-2 and det
            ok_all &= ok
            print(
                f"S={S:6d} {mode:9s} O {'ok' if ok_o else 'BAD'} (max {max_o:.2e})  "
                f"LSE {'ok' if lse_err < 1e-2 else 'BAD'} (max {lse_err:.2e})  "
                f"bit-identical across {a.reps} runs: {det}"
            )
    print("ALL OK" if ok_all else "FAILURES")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
