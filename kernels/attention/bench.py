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
"""Baseline benchmark for the FAV3 (rotated 4-cluster) Gluon flash-attention
kernel, ported from AMD-Triton/gluon-kernels (kernels/cdna4/fa).

Mirrors the GEMM tutorial bench.py: correctness vs a torch reference, do_bench
TFLOPS, a --rocprof iteration mode, and the out-of-tree plugin hooks so the same
harness can later measure base / llir / llir+plugins configs.
"""

import argparse
import os
import sys

import torch

# The out-of-tree LLIR scheduler ships as an LLVM pass plugin. Loaded via
# LLVM_PASS_PLUGIN_PATH, it resolves LLVM symbols from libtriton at dlopen time,
# which requires libtriton in the *global* symbol scope. CPython loads
# C-extensions RTLD_LOCAL by default, so opt into RTLD_GLOBAL before the first
# `import triton`. Only takes effect when the plugin is in use.
if os.environ.get("LLVM_PASS_PLUGIN_PATH"):
    sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)

import triton  # noqa: E402

# Out-of-tree amdgcnas peephole (post-assembly): install the amdgcn-stage hook
# when TRITON_AMDGCNAS_PLUGIN is set. Pure-Python text transform, no rebuild.
if os.environ.get("TRITON_AMDGCNAS_PLUGIN"):
    sys.path.insert(
        0,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "plugins", "amdgcnas"),
    )
    import amdgcnas_plugin  # noqa: E402
    from triton import knobs  # noqa: E402

    knobs.runtime.add_stages_inspection_hook = amdgcnas_plugin.inspect_stages_hook

# VALU ablation (scripts/ablate_valu.py): delete the loop's vector ALU from the final
# assembly to time the MFMA-only ceiling. The kernel then computes garbage -- this is a
# timing probe, so the correctness check below is expected to fail and must be ignored.
if os.environ.get("FA_ABLATE_VALU"):
    sys.path.insert(
        0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts")
    )
    import ablate_valu  # noqa: E402
    from triton import knobs  # noqa: E402

    knobs.runtime.add_stages_inspection_hook = ablate_valu.inspect_stages_hook

# Hoist a loop-invariant LDS base address out of the loop (scripts/hoist_lds_addr.py).
if os.environ.get("FA_HOIST_LDS_ADDR"):
    sys.path.insert(
        0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts")
    )
    import hoist_lds_addr  # noqa: E402
    from triton import knobs  # noqa: E402

    knobs.runtime.add_stages_inspection_hook = hoist_lds_addr.inspect_stages_hook

# VALU rescheduling (scripts/sched_valu.py): re-balance the loop's vector ALU across the
# MFMA shadows without adding or removing any, to separate the scheduling effect on
# throughput from the power effect of changing the instruction count. Dependency-checked,
# so the correctness check below still applies and is the control for the experiment.
if os.environ.get("FA_SCHED_VALU"):
    sys.path.insert(
        0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts")
    )
    import sched_valu  # noqa: E402
    from triton import knobs  # noqa: E402

    knobs.runtime.add_stages_inspection_hook = sched_valu.inspect_stages_hook

# Ported FAV3 kernel + shared helpers live alongside this file.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Select the kernel implementation: FA_MODULE=fav3 (default, eager rescale) or
# fav4 (lazy-rescale variant). Both expose run_gluon_attention.
import importlib  # noqa: E402

from f16_fa_gfx950_common import (
    _check_output,
    compute_flops,
    get_shape_from_layout,
    get_strides_from_layout,
    input_helper,
    sdpa_reference,
)  # noqa: E402

_FA_MODULE = os.environ.get("FA_MODULE", "fav3")
run_gluon_attention = importlib.import_module(_FA_MODULE).run_gluon_attention  # noqa: E402

# Where the softmax scale is applied (SCALE_ON_Q: pre-scale Q outside the loop, so
# VEC1's per-element fma becomes a sub). Both kernels take it, but forward it only
# when the selected kernel's launcher actually accepts the parameter.
import inspect  # noqa: E402

_SCALE_ON_Q_ARG = "scale_on_q" in inspect.signature(run_gluon_attention).parameters
_SCALE_ON_Q = True


def launch_attention(q, k, v, o, md):
    if _SCALE_ON_Q_ARG:
        run_gluon_attention(q, k, v, o, md, scale_on_q=_SCALE_ON_Q)
    else:
        run_gluon_attention(q, k, v, o, md)


DEVICE = triton.runtime.driver.active.get_active_torch_device()

name_to_torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}


# Sequence-length sweep (Q == K), the classic FA scaling axis.
SEQLENS = [1024, 2048, 4096, 8192, 16384]


def parse_args():
    p = argparse.ArgumentParser(description="FAV3 rotated-4cluster attention benchmark (gfx950)")
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--layout", choices=["bhsd", "bshd"], default="bhsd")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--hq", type=int, default=64, help="number of query heads")
    p.add_argument(
        "--hk", type=int, default=64, help="number of kv heads (hq%%hk==0; hq==hk is MHA)"
    )
    p.add_argument("--d", type=int, default=128, help="head dimension")
    p.add_argument("--seqlen", type=int, default=8192, help="sequence length (default: 8192)")
    p.add_argument("--sweep", action="store_true", help="sweep all seqlens instead of a single one")
    p.add_argument(
        "--rocprof",
        action="store_true",
        help="run the kernel n_iters times (no do_bench); wrap with rocprofv3 --kernel-trace",
    )
    p.add_argument("--n-iters", type=int, default=1000)
    p.add_argument(
        "--prepared",
        action="store_true",
        help="rocprof mode: launch through a pre-bound PreparedKernel (scripts/prepared_kernel.py) "
        "instead of the JIT wrapper, removing per-dispatch Python argument binding",
    )
    p.add_argument(
        "--n-warmup", type=int, default=10, help="prepared mode: unmeasured warmup dispatches"
    )
    p.add_argument(
        "--rotating-sets",
        type=int,
        default=0,
        help="rocprof mode: number of complete (q,k,v,o) sets cycled across dispatches so "
        "consecutive dispatches read different memory. 0 (default) derives the count from "
        "--rotating-buffer-size",
    )
    p.add_argument(
        "--scale-on-q",
        type=int,
        default=1,
        choices=[0, 1],
        help="1 (default) pre-scales Q before the loop so VEC1 does a sub; "
        "0 applies qk_scale per element inside VEC1 as an fma",
    )
    p.add_argument(
        "--rotating-buffer-size",
        type=int,
        default=512,
        help="rocprof mode: total working set (MB) to spread across rotating sets, so it exceeds "
        "the GPU cache (L2 + 256MB MALL) and dispatches see cold data. Matches the GEMM bench "
        "default. Large FA shapes already exceed it with one set (default: 512)",
    )
    return p.parse_args()


def make_inputs(B, HQ, HK, N_CTX, D, dtype, layout):
    q, k, v, md = input_helper(B, HQ, HK, N_CTX, N_CTX, D, dtype, layout)
    o = torch.empty_like(q)
    return q, k, v, o, md


def test_correctness(B, HQ, HK, N_CTX, D, dtype, layout):
    q, k, v, o, md = make_inputs(B, HQ, HK, N_CTX, D, dtype, layout)
    launch_attention(q, k, v, o, md)
    # sdpa_reference computes attention over the last two axes, i.e. it assumes a
    # bhsd [B, H, S, D] layout. For bshd [B, S, H, D] we must swap the H and S axes
    # before building the reference (and compare the kernel output in the same
    # frame), otherwise the reference attends over the wrong dims and every bshd
    # run spuriously reports a mismatch even though the kernel is correct.
    if layout == "bshd":
        q_ref, k_ref, v_ref = (t.transpose(1, 2) for t in (q, k, v))
        o_cmp = o.transpose(1, 2)
    else:
        q_ref, k_ref, v_ref, o_cmp = q, k, v, o
    o_ref = sdpa_reference(q_ref, k_ref, v_ref, causal=False, sm_scale=md.sm_scale)
    ok, max_diff, mean_diff = _check_output(o_cmp, o_ref)
    tag = "✅ match" if ok else "❌ MISMATCH"
    print(
        f"[FAV3] B={B} HQ={HQ} HK={HK} N={N_CTX} D={D} non-causal {layout} "
        f"{q.dtype}: {tag}  (max={max_diff:.2e} mean={mean_diff:.2e})"
    )
    return ok


def rotating_set_count(args, torch_dtype, N_CTX):
    """How many (q,k,v,o) sets to cycle so the loop's footprint exceeds the GPU cache.

    A single FA dispatch touches (2*HQ + 2*HK) * B * N_CTX * D elements (q,k,v,o). When
    that already exceeds --rotating-buffer-size, one set is enough and rotation would only
    waste memory; small shapes fit inside the 256MB MALL and stay warm across dispatches,
    reporting a kernel time that no cold-start workload would ever see.
    """
    if args.rotating_sets > 0:
        return args.rotating_sets
    itemsize = torch.empty(0, dtype=torch_dtype).element_size()
    per_set = (2 * args.hq + 2 * args.hk) * args.batch * N_CTX * args.d * itemsize
    target = args.rotating_buffer_size * 1024 * 1024
    return max(1, -(-target // max(1, per_set)))


def run_rocprof_iterations(args, torch_dtype, seqlens):
    for N_CTX in seqlens:
        n_sets = rotating_set_count(args, torch_dtype, N_CTX)
        sets = [
            make_inputs(args.batch, args.hq, args.hk, N_CTX, args.d, torch_dtype, args.layout)
            for _ in range(n_sets)
        ]
        q, k, v, o, md = sets[0]
        launch_attention(q, k, v, o, md)  # warmup + autotune
        torch.cuda.synchronize()
        for i in range(args.n_iters):
            q, k, v, o, md = sets[i % len(sets)]
            launch_attention(q, k, v, o, md)
        torch.cuda.synchronize()
        print(f"[FAV3] N={N_CTX}: {args.n_iters} iterations done ({n_sets} rotating set(s))")


def make_prepared_kernel(N_CTX, args, torch_dtype, n_sets):
    """Bind the FA specialization once, for ``n_sets`` rotating tensor sets.

    Mirrors ``run_gluon_attention``'s launch, but hoists the argument binding and
    specialization lookup out of the dispatch loop (see ``scripts/prepared_kernel.py``).
    The single autotune config is read off the kernel module, so the prepared launch
    compiles the same specialization the ordinary path would have picked.
    """
    sys.path.insert(
        0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts")
    )
    from prepared_kernel import PreparedKernel  # noqa: E402

    fa = sys.modules[_FA_MODULE]
    cfg = fa.get_gluon_cdna_autotune_configs()[0]
    BLOCK_M, BLOCK_N = cfg.kwargs["BLOCK_M"], cfg.kwargs["BLOCK_N"]

    sets = [
        make_inputs(args.batch, args.hq, args.hk, N_CTX, args.d, torch_dtype, args.layout)
        for _ in range(n_sets)
    ]
    q0, k0, _, o0, md0 = sets[0]
    batch, nheads_q, nheads_k, head_size = get_shape_from_layout(q0, k0, md0)

    runtime_argument_sets = []
    for q, k, v, o, md in sets:
        strides = get_strides_from_layout(q, k, v, o, md)
        L = torch.empty((batch, nheads_q, N_CTX), device=q.device, dtype=torch.float32)
        runtime_argument_sets.append(
            (q, k, v, md.sm_scale, L, o, *[s for grp in strides for s in grp])
        )

    prepared = PreparedKernel.create(
        fa.gluon_attn_fwd.fn,
        (nheads_q, triton.cdiv(N_CTX, BLOCK_M), batch),
        runtime_argument_sets,
        constexprs=dict(
            **({"SCALE_ON_Q": _SCALE_ON_Q} if _SCALE_ON_Q_ARG else {}),
            HQ=nheads_q,
            HK=nheads_k,
            N_CTX=N_CTX,
            IS_CAUSAL=False,
            BLOCK_M=BLOCK_M,
            BLOCK_DMODEL=head_size,
            BLOCK_N=BLOCK_N,
        ),
        compiler_options=dict(
            num_warps=cfg.num_warps,
            waves_per_eu=cfg.kwargs["waves_per_eu"],
            llvm_fn_attrs=cfg.kwargs["llvm_fn_attrs"],
        ),
    )
    return prepared, sets


def run_prepared_iterations(args, torch_dtype, seqlens):
    """Prepared-launch rocprof mode: n_warmup unmeasured + n_iters measured dispatches."""
    for N_CTX in seqlens:
        prepared, sets = make_prepared_kernel(
            N_CTX, args, torch_dtype, rotating_set_count(args, torch_dtype, N_CTX)
        )
        # Validate the pre-bound launch against the ordinary launcher before timing it,
        # so a mis-bound argument list can never be reported as a fast kernel.
        q, k, v, o, md = sets[0]
        o_ref = torch.empty_like(o)
        launch_attention(q, k, v, o_ref, md)
        o.zero_()
        prepared(0)
        torch.cuda.synchronize()
        ok, max_diff, _ = _check_output(o, o_ref)
        print(
            f"[FAV3] N={N_CTX}: prepared vs ordinary launch "
            f"{'✅ match' if ok else '❌ MISMATCH'} (max={max_diff:.2e})"
        )
        if not ok:
            # Under FA_ABLATE_VALU the kernel computes garbage by construction, so this
            # launcher-equivalence check compares NaN against NaN and cannot pass. The
            # ablation is a timing probe; skip the check rather than the measurement.
            if os.environ.get("FA_ABLATE_VALU"):
                print(
                    "[FAV3] ablation active: launcher-equivalence check skipped "
                    "(output is garbage by design)"
                )
            else:
                sys.exit(1)

        for i in range(args.n_warmup):
            prepared(i)
        torch.cuda.synchronize()
        for i in range(args.n_iters):
            prepared(i)
        torch.cuda.synchronize()
        print(
            f"[FAV3] N={N_CTX}: {args.n_warmup} warmup + {args.n_iters} prepared "
            f"dispatches done ({prepared.slots} rotating set(s))"
        )


def main():
    args = parse_args()
    global _SCALE_ON_Q
    _SCALE_ON_Q = bool(args.scale_on_q)
    if not _SCALE_ON_Q and not _SCALE_ON_Q_ARG:
        print(f"Error: {_FA_MODULE} has no scale_on_q parameter")
        sys.exit(2)
    torch_dtype = name_to_torch_dtype[args.dtype]
    seqlens = SEQLENS if args.sweep else [args.seqlen]

    # 1) Correctness vs the memory-efficient torch SDPA reference (non-causal).
    for N_CTX in seqlens:
        test_correctness(args.batch, args.hq, args.hk, N_CTX, args.d, torch_dtype, args.layout)

    # 2) Rocprof mode: cold, external timing.
    if args.rocprof:
        if args.prepared:
            run_prepared_iterations(args, torch_dtype, seqlens)
        else:
            run_rocprof_iterations(args, torch_dtype, seqlens)
        return

    # 3) do_bench TFLOPS sweep (non-causal).
    configs = [
        triton.testing.Benchmark(
            x_names=["N_CTX"],
            x_vals=seqlens,
            line_arg="provider",
            line_vals=["gluon"],
            line_names=["non-causal"],
            styles=[("blue", "-")],
            ylabel="TFLOPS",
            plot_name=(
                f"fav3-attn-B{args.batch}-HQ{args.hq}-HK{args.hk}"
                f"-D{args.d}-{args.layout}-{args.dtype}"
            ),
            args={},
        )
    ]

    @triton.testing.perf_report(configs)
    def benchmark(N_CTX, provider):
        q, k, v, o, md = make_inputs(
            args.batch, args.hq, args.hk, N_CTX, args.d, torch_dtype, args.layout
        )
        fn = lambda: launch_attention(q, k, v, o, md)  # noqa: E731
        fn()  # trigger autotune + warm compile before timing
        torch.cuda.synchronize()
        ms = triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")
        return compute_flops(args.batch, args.hq, N_CTX, N_CTX, args.d, False) / ms * 1e-9

    print(
        f"\nFAV3 rotated-4cluster attention | B={args.batch} HQ={args.hq} HK={args.hk} "
        f"D={args.d} layout={args.layout} dtype={args.dtype} non-causal"
    )
    benchmark.run(show_plots=False, print_data=True)


if __name__ == "__main__":
    main()
