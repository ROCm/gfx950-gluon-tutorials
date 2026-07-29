#!/usr/bin/env python3
##############################################################################
# MIT License
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
##############################################################################
"""Prepared-launch, rocprofv3-timed benchmark for the gfx942 GEMM kernels.

The gfx950 tutorial's `scripts/benchmark_prepared.py` is hardwired to the
`kernels/gemm/{intra,inter}_wave/<kernel>` layout, so this is the gfx942
equivalent. It reuses the tutorial's generic `scripts/prepared_kernel.py`
unchanged: compile once via `warmup` (no dispatch), pre-bind the full argument
list for every rotating tensor set, then enter the compiled launch stub
directly. That removes the Python binding and specialization lookup from the
gap between dispatches, so `rocprofv3 --kernel-trace` sees back-to-back
kernels.

Timing is taken from the *last* `--avg-last` of `--iters` dispatches, by which
point the clock has settled to its sustained value under the 750 W cap. Two
identical `--sets` rotate so no dispatch reads its predecessor's cache-resident
operands.

    python bench_prepared.py --gpus all          # sweep every GPU, pick fastest
    python bench_prepared.py --gpus 3            # one GPU

Re-execs itself as `--_driver` under rocprofv3, like `run_att.py`.
"""

import argparse
import collections
import csv
import glob
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
REPO = os.path.join(ROOT, "..", "..", "..")

KERNEL_REGEX = {
    "intra_wave": "a16w16_intra_wave_gfx942",
    "inter_wave": "a16w16_inter_wave_gfx942",
    # "torch" runs torch.matmul (hipBLASLt) through the identical harness so the
    # reference number is comparable: same GPU, same rotating sets, same
    # rocprofv3 --kernel-trace, same mean-of-last-N. hipBLASLt picks its kernel
    # by shape so the name is not knowable up front; the parent falls back to the
    # most frequent kernel name in the trace.
    "torch": None,
}


def _load(name):
    import importlib.util

    sys.path.insert(0, os.path.join(ROOT, "..", "utils"))
    spec = importlib.util.spec_from_file_location(
        f"gfx942_{name}", os.path.join(ROOT, name, "matmul_kernel.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _driver_main(argv):
    p = argparse.ArgumentParser(prog="bench_prepared.py --_driver")
    p.add_argument("kernel", choices=sorted(KERNEL_REGEX))
    p.add_argument("--M", type=int, default=4096)
    p.add_argument("--N", type=int, default=4864)
    p.add_argument("--K", type=int, default=8256)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    p.add_argument("--sets", type=int, default=3)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=1000)
    args = p.parse_args(argv)

    os.environ.setdefault("TRITON_FORCE_MFMA_AGPR", "1")
    if os.environ.get("LLVM_PASS_PLUGIN_PATH"):
        sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)

    import torch
    import triton

    if args.kernel == "torch":
        return _driver_torch(args, torch, triton)

    sys.path.insert(0, os.path.join(REPO, "scripts"))
    from prepared_kernel import PreparedKernel

    mod = _load(args.kernel)
    dt = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    device = triton.runtime.driver.active.get_active_torch_device()
    M, N, K = args.M, args.N, args.K
    torch.manual_seed(0)

    runtime_sets, per_set = [], 0
    for _ in range(args.sets):
        a = torch.randn((M, K), device=device, dtype=dt) * 0.1
        b_t = torch.randn((N, K), device=device, dtype=dt) * 0.1
        c = torch.empty((M, N), device=device, dtype=dt)
        per_set = sum(t.numel() * t.element_size() for t in (a, b_t, c))
        runtime_sets.append(
            (
                a,
                b_t,
                c,
                M,
                N,
                K,
                a.stride(0),
                a.stride(1),
                b_t.stride(1),
                b_t.stride(0),
                c.stride(0),
                c.stride(1),
            )
        )

    jit_kernel = getattr(mod, KERNEL_REGEX[args.kernel])
    grid_mn = triton.cdiv(M, mod.BLOCK_M) * triton.cdiv(N, mod.BLOCK_N)

    # Derive the constexprs from the kernel signature rather than hardcoding one
    # kernel's shape: take every constexpr argument the module defines at module
    # level (BLOCK_*, WARPS_*, SPLIT_K, NUM_XCDS, GROUP_SIZE_M ...). GRID_MN is
    # the only one computed from the problem size.
    constexprs = {"GRID_MN": grid_mn}
    for name in jit_kernel.arg_names:
        if name in constexprs or not hasattr(mod, name):
            continue
        value = getattr(mod, name)
        constexprs[name] = getattr(value, "value", value)  # unwrap gl.constexpr

    compiler_options = {"num_warps": mod.NUM_WARPS}
    if hasattr(mod, "_agpr_attrs"):
        attrs = mod._agpr_attrs()  # intra_wave: env-gated agpr-alloc=256
    else:
        attrs = (("amdgpu-agpr-alloc", "0,0"),)  # inter_wave pins no-AGPR itself
    if attrs:
        compiler_options["llvm_fn_attrs"] = attrs
    prepared = PreparedKernel.create(
        jit_kernel,
        (grid_mn, 1),
        runtime_sets,
        constexprs=constexprs,
        compiler_options=compiler_options,
    )

    for i in range(args.warmup):
        prepared(i % args.sets)
    torch.cuda.synchronize()

    # Guard against a stale/overridden kernel in the Triton cache. The cache key
    # does not include TRITON_KERNEL_OVERRIDE, so a previous ablation run can
    # leave a hand-edited hsaco behind and every number below would silently
    # describe the wrong kernel. Verify before timing, not after.
    a0, b0, c0 = runtime_sets[0][0], runtime_sets[0][1], runtime_sets[0][2]
    ref = (a0.float() @ b0.float().T).to(dt)
    if not torch.allclose(c0, ref, atol=2e-2, rtol=2e-2):
        raise SystemExit(
            "FATAL: kernel output is numerically wrong -- refusing to report timings.\n"
            "       Most likely a stale TRITON_KERNEL_OVERRIDE build in the Triton\n"
            "       cache. Clear it (rm -rf ~/.triton/cache) and re-run."
        )

    for i in range(args.iters):
        prepared(i % args.sets)
    torch.cuda.synchronize()

    print(
        f"[driver] {args.kernel} {M}x{N}x{K} {args.dtype} sets={args.sets} "
        f"({per_set / 1024 ** 2:.1f} MiB/set) warmup={args.warmup} iters={args.iters} "
        f"prepared_launch=True"
    )
    return 0


def _driver_torch(args, torch, triton):
    """hipBLASLt reference through the same rotating-set / dispatch-count harness."""
    dt = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    device = triton.runtime.driver.active.get_active_torch_device()
    M, N, K = args.M, args.N, args.K
    torch.manual_seed(0)
    sets = []
    for _ in range(args.sets):
        a = torch.randn((M, K), device=device, dtype=dt) * 0.1
        b = torch.randn((K, N), device=device, dtype=dt) * 0.1
        c = torch.empty((M, N), device=device, dtype=dt)
        sets.append((a, b, c))
    for i in range(args.warmup):
        a, b, c = sets[i % args.sets]
        torch.matmul(a, b, out=c)
    torch.cuda.synchronize()
    for i in range(args.iters):
        a, b, c = sets[i % args.sets]
        torch.matmul(a, b, out=c)
    torch.cuda.synchronize()
    print(f"[driver] torch {M}x{N}x{K} {args.dtype} sets={args.sets} iters={args.iters}")
    return 0


def time_one_gpu(args, gpu, out):
    """Run the driver once under rocprofv3 --kernel-trace; return (tflops, us, n)."""
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out, exist_ok=True)
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("TRITON_FORCE_MFMA_AGPR", "1")
    cmd = [
        "rocprofv3",
        "--kernel-trace",
        "--output-format",
        "csv",
        "-d",
        out,
        "--",
        sys.executable,
        os.path.abspath(__file__),
        "--_driver",
        args.kernel,
        "--M",
        str(args.M),
        "--N",
        str(args.N),
        "--K",
        str(args.K),
        "--dtype",
        args.dtype,
        "--sets",
        str(args.sets),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
    ]
    subprocess.run(
        cmd, env=env, check=True, cwd=out, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    csvs = glob.glob(os.path.join(out, "**", "*kernel_trace.csv"), recursive=True)
    if not csvs:
        raise RuntimeError(f"no kernel trace csv under {out}")
    rows = []
    if args.kernel == "torch":
        # hipBLASLt's kernel name depends on the shape; take the modal name.
        names = collections.Counter()
        with open(csvs[0], newline="") as f:
            for row in csv.DictReader(f):
                names[row["Kernel_Name"]] += 1
        match = names.most_common(1)[0][0]
    else:
        match = KERNEL_REGEX[args.kernel]
    with open(csvs[0], newline="") as f:
        for row in csv.DictReader(f):
            if match in row["Kernel_Name"]:
                rows.append(int(row["End_Timestamp"]) - int(row["Start_Timestamp"]))
    if len(rows) < args.avg_last:
        raise RuntimeError(f"only {len(rows)} dispatches, need {args.avg_last}")
    tail = rows[-args.avg_last :]
    ns = sum(tail) / len(tail)
    tflops = 2 * args.M * args.N * args.K / (ns * 1e-9) / 1e12
    return tflops, ns / 1e3, len(rows)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--_driver":
        return _driver_main(sys.argv[2:])

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--kernel", choices=sorted(KERNEL_REGEX), default="intra_wave")
    p.add_argument("--M", type=int, default=4096)
    p.add_argument("--N", type=int, default=4864)
    p.add_argument("--K", type=int, default=8256)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    p.add_argument("--sets", type=int, default=3, help="rotating tensor sets")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--avg-last", type=int, default=100, help="average the final N dispatches")
    p.add_argument("--gpus", default="all", help="'all', or a comma list like 0,3,5")
    p.add_argument("--out", default="/tmp/bench_prepared_gfx942")
    args = p.parse_args()

    if args.gpus == "all":
        import torch

        gpus = list(range(torch.cuda.device_count()))
    else:
        gpus = [int(g) for g in args.gpus.split(",")]

    print(
        f"{args.kernel} {args.M}x{args.N}x{args.K} {args.dtype} | prepared launch, "
        f"{args.sets} rotating sets | {args.iters} dispatches, mean of last {args.avg_last}"
    )
    print(f"{'GPU':>4} {'TFLOPS':>10} {'us/dispatch':>13} {'dispatches':>11}")
    results = {}
    for g in gpus:
        try:
            tf, us, n = time_one_gpu(args, g, os.path.join(args.out, f"gpu{g}"))
            results[g] = tf
            print(f"{g:>4} {tf:>10.1f} {us:>13.2f} {n:>11}")
        except Exception as e:  # a busy or unhealthy GPU should not abort the sweep
            print(f"{g:>4} {'FAILED':>10}   {e}")
    if results:
        best = max(results, key=results.get)
        print(
            f"\nfastest: GPU {best} at {results[best]:.1f} TFLOPS "
            f"(slowest {min(results.values()):.1f}, spread "
            f"{100 * (results[best] / min(results.values()) - 1):.1f}%)"
        )
        print(json.dumps({"best_gpu": best, "tflops": results}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
