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

"""Correctness + benchmark driver for both gfx942 a16w16 GEMM kernels.

    python bench.py                                  # both kernels, fp16+bf16, vs torch
    python bench.py --kernel inter_wave --K 4160 --dtype fp16 --show-clock
    python bench.py --sweep-gm --K 4160              # GROUP_SIZE_M sweep
    python bench.py --kernel intra_wave --K 8256 --rocprof
    rocprofv3 --kernel-trace -f csv -d out -- python bench.py -k inter_wave --K 4160 --rocprof

Correctness is checked against an **fp32** reference: the kernels accumulate in
fp32 and round once at the end, so they are strictly more accurate than a
half-precision reference, and the tolerance is 4 ulps of the output type
relative to the peak output magnitude.

MI300X caveat: sustained MFMA drives the part into its 750 W cap and the clock
drops from ~2.03 GHz to ~1.52 GHz, so absolute TFLOPS depend heavily on how long
the benchmark has been running.  `--show-clock` samples rocm-smi so the number
can be read against the clock that produced it, and the torch/hipBLASLt column
is measured in the same loop as a same-conditions reference.
"""

import argparse
import importlib.util
import math
import os
import statistics
import subprocess
import sys

# The 4-wave kernel needs the force-agpr RA hint (amdgpu-agpr-alloc=256) before
# it compiles; harmless for the 8-wave kernel, which pins agpr-alloc=0,0 itself.
os.environ.setdefault("TRITON_FORCE_MFMA_AGPR", "1")

import torch  # noqa: E402
import triton  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
# get_pids (XCD remap + GROUP_SIZE_M swizzle) is shared with the gfx950 kernels
sys.path.insert(0, os.path.join(HERE, "..", "utils"))

DEVICE = triton.runtime.driver.active.get_active_torch_device()
KERNELS = ("intra_wave", "inter_wave")
NAME_TO_TORCH_TYPE = {"fp16": torch.float16, "bf16": torch.bfloat16}
# 4 ulps of the output type, relative to the largest output element
TOL = {"fp16": 4 * 2**-10, "bf16": 4 * 2**-7}


def get_x_vals():
    """gfx942-native shapes.

    N=4864 makes the grid 16 x 19 = 304 workgroups for a 256x256 tile, which is
    exactly the MI300X CU count -- one workgroup per CU, no tail wave.  (4096 x
    4096 gives 256 workgroups and leaves 48 of 304 CUs idle; LDS is full at
    64 KB, so occupancy is 1 workgroup/CU and the grid must match the machine.)

    K is deliberately not a power of two: with K=4096 the row stride of A is
    8192 B, so every row of a tile maps to the same L1 set and the loads
    hotspot.  4160 = 65 x 64 and 8256 = 129 x 64 are both multiples of BLOCK_K
    but make the stride an odd multiple of the 128 B line.
    """
    return [
        (4096, 4864, 2112),
        (4096, 4864, 4160),
        (4096, 4864, 8256),
        (4096, 4864, 16448),
    ]


def load_kernel(name):
    """Import kernels/gemm/gfx942/<name>/matmul_kernel.py under a unique module
    name -- both kernel files are called `matmul_kernel`."""
    spec = importlib.util.spec_from_file_location(
        f"gfx942_{name}", os.path.join(HERE, name, "matmul_kernel.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def gpu_clock_mhz():
    try:
        out = subprocess.run(
            ["rocm-smi", "--showgpuclocks", "--csv"], capture_output=True, text=True, timeout=5
        ).stdout
        for line in out.splitlines():
            if line.startswith("card"):
                for tok in line.split(","):
                    if "Mhz" in tok:
                        return int(tok.strip().strip("()").split("Mhz")[0].split("(")[-1])
    except Exception:
        pass
    return None


def make_inputs(M, N, K, torch_dtype):
    """A is (M, K) contiguous; b_t is B stored transposed (N, K) contiguous, so
    the kernel's logical (K, N) operand has K contiguous, as both kernels need."""
    a = torch.randn((M, K), device=DEVICE, dtype=torch_dtype) * 0.1
    b_t = torch.randn((N, K), device=DEVICE, dtype=torch_dtype) * 0.1
    c = torch.empty((M, N), device=DEVICE, dtype=torch_dtype)
    return a, b_t, c


def check(mod, a, b_t, c, dtype):
    """Return (ok, relative error vs an fp32 reference)."""
    out = mod.matmul_kernel_only(a, b_t, c)
    ref = a.float() @ b_t.t().float()
    torch.cuda.synchronize()
    rel = (out.float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-30)
    return rel <= TOL[dtype], rel


def gen_rotating_tensors(M, N, K, torch_dtype, mb=512):
    """Several copies of (a, b_t, c) so consecutive iterations read different
    addresses and the caches stay cold, as in the gfx950 bench.py."""
    elem = torch.tensor([], dtype=torch_dtype).element_size()
    n = max(1, mb * 1024 * 1024 // ((M * K + K * N + M * N) * elem))
    return (
        [torch.randn((M, K), device=DEVICE, dtype=torch_dtype) for _ in range(n)],
        [torch.randn((N, K), device=DEVICE, dtype=torch_dtype) for _ in range(n)],
        [torch.empty((M, N), device=DEVICE, dtype=torch_dtype) for _ in range(n)],
        n,
    )


def run_rocprof(mods, dtypes, sizes, n_iters=1000, mb=512):
    for name, mod in mods.items():
        for dtype in dtypes:
            td = NAME_TO_TORCH_TYPE[dtype]
            for M, N, K in sizes:
                if K < mod.MIN_K:
                    continue
                a_l, b_l, c_l, n = gen_rotating_tensors(M, N, K, td, mb)
                print(f"[{name}] {M}x{N}x{K} {dtype}: {n} rotating copies")
                mod.matmul_kernel_only(a_l[0], b_l[0], c_l[0])
                torch.cuda.synchronize()
                for i in range(n_iters):
                    j = i % n
                    mod.matmul_kernel_only(a_l[j], b_l[j], c_l[j])
                torch.cuda.synchronize()
                print(f"[{name}] {M}x{N}x{K} {dtype}: {n_iters} iterations done")


def run_bench(mods, dtypes, sizes, reps, with_torch, show_clock):
    names = list(mods)
    hdr = f"{'M':>6} {'N':>6} {'K':>7} {'dtype':>6}"
    for n in names:
        hdr += f" {n + ' TF':>18}"
    if with_torch:
        hdr += f" {'torch TF':>10}"
    if show_clock:
        hdr += f" {'MHz':>6}"
    print(f"\nTFLOPS, do_bench median of {reps}   (correctness vs fp32 shown as ok/BAD)")
    print(hdr)

    ok_all = True
    for dtype in dtypes:
        td = NAME_TO_TORCH_TYPE[dtype]
        for M, N, K in sizes:
            a, b_t, c = make_inputs(M, N, K, td)
            flops = 2 * M * N * K
            row = f"{M:>6} {N:>6} {K:>7} {dtype:>6}"
            for n in names:
                mod = mods[n]
                if K < mod.MIN_K:
                    row += f" {'skip (K<MIN_K)':>18}"
                    continue
                ok, _ = check(mod, a, b_t, c, dtype)
                ok_all &= ok
                tf = [
                    flops
                    * 1e-12
                    / (triton.testing.do_bench(lambda: mod.matmul_kernel_only(a, b_t, c)) * 1e-3)
                    for _ in range(reps)
                ]
                row += f" {statistics.median(tf):>14.1f} {'ok' if ok else 'BAD':>3}"
            if with_torch:
                bt = b_t.t()
                tf = [
                    flops * 1e-12 / (triton.testing.do_bench(lambda: torch.matmul(a, bt)) * 1e-3)
                    for _ in range(reps)
                ]
                row += f" {statistics.median(tf):>10.1f}"
            if show_clock:
                clk = gpu_clock_mhz()
                row += f" {clk if clk else '?':>6}"
            print(row)
    return ok_all


def run_sweep_gm(mods, dtype, sizes, gms, reps):
    """Sweep GROUP_SIZE_M against the v9_beyond_hotloop model: each XCD gets
    P = GRID_MN / NUM_XCDS workgroups laid out GM x ceil(P/GM), so it reads GM
    row-strips of A and ceil(P/GM) column-strips of B -- minimise
    f(GM) = GM + ceil(P/GM)."""
    td = NAME_TO_TORCH_TYPE[dtype]
    names = list(mods)
    for M, N, K in sizes:
        a, b_t, c = make_inputs(M, N, K, td)
        flops = 2 * M * N * K
        any_mod = mods[names[0]]
        grid = triton.cdiv(M, any_mod.BLOCK_M) * triton.cdiv(N, any_mod.BLOCK_N)
        per_xcd = grid / any_mod.NUM_XCDS
        print(
            f"\n{M}x{N}x{K} {dtype}  grid={grid} workgroups, "
            f"{per_xcd:.1f} per XCD over {any_mod.NUM_XCDS} XCDs  "
            f"(median of {reps})"
        )
        hdr = f"{'GM':>4} {'f(GM)':>6}"
        for n in names:
            hdr += f" {n + ' TF':>18}"
        print(hdr)
        for gm in gms:
            row = f"{gm:>4} {gm + math.ceil(per_xcd / gm):>6}"
            for n in names:
                mod = mods[n]
                mod.GROUP_SIZE_M = gm
                ok, _ = check(mod, a, b_t, c, dtype)
                tf = [
                    flops
                    * 1e-12
                    / (triton.testing.do_bench(lambda: mod.matmul_kernel_only(a, b_t, c)) * 1e-3)
                    for _ in range(reps)
                ]
                row += f" {statistics.median(tf):>14.1f} {'ok' if ok else 'BAD':>3}"
            print(row)
        for n in names:  # restore the shipped default
            mods[n].GROUP_SIZE_M = 4


def main():
    p = argparse.ArgumentParser(description="gfx942 a16w16 GEMM correctness + benchmark")
    p.add_argument("-k", "--kernel", nargs="+", choices=[*KERNELS, "both"], default=["both"])
    p.add_argument("--K", type=int, default=None, help="only the size with this K")
    p.add_argument("--dtype", nargs="+", choices=["fp16", "bf16"], default=["fp16", "bf16"])
    p.add_argument("--reps", type=int, default=1, help="do_bench repeats, reported as median")
    p.add_argument("--no-torch", action="store_true", help="skip the torch/hipBLASLt column")
    p.add_argument(
        "--show-clock", action="store_true", help="sample rocm-smi per row (MI300X power-throttles)"
    )
    p.add_argument("--sweep-gm", action="store_true", help="sweep GROUP_SIZE_M instead")
    p.add_argument("--gm", type=int, nargs="+", default=[1, 2, 4, 5, 6, 8, 16])
    p.add_argument(
        "--rocprof",
        action="store_true",
        help="1000 iterations with rotating buffers, no do_bench, for "
        "external `rocprofv3 --kernel-trace` timing",
    )
    p.add_argument(
        "--rotating-buffer-size",
        type=int,
        default=512,
        help="total MB of rotating copies in --rocprof mode",
    )
    args = p.parse_args()

    which = KERNELS if "both" in args.kernel else tuple(args.kernel)
    mods = {n: load_kernel(n) for n in which}

    sizes = get_x_vals()
    if args.K is not None:
        sizes = [s for s in sizes if s[2] == args.K]
        if not sizes:
            p.error(f"no shape with K={args.K}; have {sorted({k for _, _, k in get_x_vals()})}")

    if args.rocprof:
        run_rocprof(mods, args.dtype, sizes, mb=args.rotating_buffer_size)
        return 0
    if args.sweep_gm:
        run_sweep_gm(mods, args.dtype[0], sizes, args.gm, args.reps)
        return 0

    ok = run_bench(mods, args.dtype, sizes, args.reps, not args.no_torch, args.show_clock)
    print("\nall correctness checks passed" if ok else "\nSOME CORRECTNESS CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
