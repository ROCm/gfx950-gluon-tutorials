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

"""Prepared-launch driver for kernel-only rocprofv3 timing.

This driver supports the final intra- and inter-wave a16w16, a8w8, and a4w4
GEMMs.  It allocates complete rotating tensor sets, compiles once without a
dispatch, and then enters the cached compiled launcher directly.  Use it under
``rocprofv3 --kernel-trace`` and compute time from the final matching records.
"""

from __future__ import annotations

import argparse
import importlib
import os
import pathlib
import sys

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
UTILS_DIR = REPO_ROOT / "kernels" / "gemm" / "utils"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=("intra", "inter"), required=True)
    parser.add_argument("--kernel", choices=("a16w16", "a8w8", "a4w4"), required=True)
    parser.add_argument(
        "--version",
        type=int,
        default=None,
        help="Kernel version. Required for versioned a16w16/a4w4 families.",
    )
    parser.add_argument(
        "--dtype",
        choices=("fp16", "bf16"),
        default="bf16",
        help="Input/output type for a16w16 (default: bf16).",
    )
    parser.add_argument("--M", type=int, default=4096)
    parser.add_argument("--N", type=int, default=4096)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--sets", type=int, default=3, help="Complete rotating tensor sets.")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=1000)
    args = parser.parse_args()
    if min(args.M, args.N, args.sets, args.warmup, args.iters) <= 0:
        parser.error("M, N, sets, warmup, and iters must be positive")
    if args.K is None:
        args.K = {"a16w16": 8192, "a8w8": 16384, "a4w4": 32768}[args.kernel]
    if args.K <= 0:
        parser.error("K must be positive")
    if args.kernel == "a16w16" and args.route == "intra" and args.version is None:
        args.version = 9
    if args.kernel == "a4w4" and args.version is None:
        args.version = 1
    return args


def load_kernel(args):
    kernel_dir = REPO_ROOT / "kernels" / "gemm" / f"{args.route}_wave" / args.kernel
    sys.path.insert(0, str(UTILS_DIR))
    sys.path.insert(0, str(kernel_dir))

    # The intra-wave bench module installs the optional LLIR/amdgcnas hooks
    # before importing a kernel. Inter-wave runs require no compiler plugins.
    bench = importlib.import_module("bench")
    if args.route == "inter":
        forbidden = (
            "LLVM_PASS_PLUGIN_PATH",
            "LLVM_PASS_PLUGIN_KEEP_TARGET_MACHINE",
            "TRITON_FORCE_MFMA_AGPR",
            "TRITON_AMDGCNAS_PLUGIN",
        )
        active = [name for name in forbidden if os.environ.get(name)]
        if active:
            raise RuntimeError(
                "inter-wave kernels require compiler plugins to be unset; active: "
                + ", ".join(active)
            )

    if args.kernel == "a16w16" and args.route == "intra":
        if args.version not in bench.VERSION_MAP:
            raise ValueError(f"invalid intra a16w16 version {args.version}")
        version_dir = bench.VERSION_MAP[args.version]
        module = importlib.import_module(f"{version_dir}.matmul_kernel")
        # v3's directory label covers two LDS-layout experiments. Match its
        # public matmul() wrapper, which selects the padding variant by default.
        kernel_name = "v3_lds_padding" if args.version == 3 else version_dir
        return module, getattr(module, kernel_name), kernel_name
    if args.kernel == "a4w4":
        if args.version not in bench.VERSION_MAP:
            raise ValueError(f"invalid {args.route} a4w4 version {args.version}")
        version_dir = bench.VERSION_MAP[args.version]
        module = importlib.import_module(f"{version_dir}.matmul_kernel")
        return module, getattr(module, version_dir), version_dir

    module = importlib.import_module("matmul_kernel")
    kernel_name = f"{args.kernel}_kernel"
    return module, getattr(module, kernel_name), kernel_name


def make_fp4_set(m, n, k, device):
    a = torch.randint(0, 256, (m, k // 2), dtype=torch.uint8, device=device)
    b = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=device)
    m_pad = (m + 255) // 256 * 256
    a_scales = torch.randint(124, 128, (k // 32, m_pad), dtype=torch.uint8, device=device).T[:m]
    b_scales = torch.randint(124, 128, (k // 32, n), dtype=torch.uint8, device=device).T
    c = torch.empty((m, n), dtype=torch.bfloat16, device=device)
    return a, b, a_scales, b_scales, c


def make_runtime_arguments(args, device):
    m, n, k = args.M, args.N, args.K
    runtime_sets = []
    bytes_per_set = 0
    for _ in range(args.sets):
        if args.kernel == "a16w16":
            dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
            a = torch.randn((m, k), device=device, dtype=dtype)
            b_t = torch.randn((n, k), device=device, dtype=dtype)
            c = torch.empty((m, n), device=device, dtype=dtype)
            b_strides = (
                (b_t.stride(1), b_t.stride(0))
                if args.route == "inter"
                else (b_t.T.stride(0), b_t.T.stride(1))
            )
            runtime_sets.append(
                (
                    a,
                    b_t,
                    c,
                    m,
                    n,
                    k,
                    a.stride(0),
                    a.stride(1),
                    *b_strides,
                    c.stride(0),
                    c.stride(1),
                )
            )
            bytes_per_set = (m * k + n * k + m * n) * a.element_size()
        elif args.kernel == "a8w8":
            a = torch.randn((m, k), device=device, dtype=torch.float16).to(torch.float8_e5m2)
            b_t = torch.randn((n, k), device=device, dtype=torch.float16).to(torch.float8_e5m2)
            c = torch.empty((m, n), device=device, dtype=torch.float16)
            runtime_sets.append(
                (
                    a,
                    b_t,
                    c,
                    m,
                    n,
                    k,
                    a.stride(0),
                    a.stride(1),
                    b_t.stride(1),
                    b_t.stride(0),
                    c.stride(0),
                    c.stride(1),
                )
            )
            bytes_per_set = m * k + n * k + 2 * m * n
        else:
            a, b, a_scales, b_scales, c = make_fp4_set(m, n, k, device)
            runtime_sets.append(
                (
                    a,
                    b,
                    c,
                    a_scales,
                    b_scales,
                    m,
                    n,
                    k,
                    a.stride(0),
                    a.stride(1),
                    b.stride(0),
                    b.stride(1),
                    c.stride(0),
                    c.stride(1),
                    a_scales.stride(0),
                    a_scales.stride(1),
                    b_scales.stride(0),
                    b_scales.stride(1),
                )
            )
            bytes_per_set = (m * k + n * k) // 2 + (m + n) * (k // 32) + 2 * m * n
    return runtime_sets, bytes_per_set


def launch_configuration(args, jit_kernel):
    import triton

    block_k = {"a16w16": 64, "a8w8": 128, "a4w4": 256}[args.kernel]
    grid_mn = triton.cdiv(args.M, 256) * triton.cdiv(args.N, 256)
    constexprs = {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": block_k}
    optional = {"GRID_MN": grid_mn, "NUM_XCDS": 8, "GROUP_SIZE_M": 4}
    constexprs.update(
        {name: value for name, value in optional.items() if name in jit_kernel.arg_names}
    )

    compiler_options = {"num_warps": 4}
    if args.route == "inter":
        constexprs.update({"WARPS_M": 2, "WARPS_N": 4})
        compiler_options = {
            "num_warps": 8,
            "llvm_fn_attrs": (("amdgpu-agpr-alloc", "0,0"),),
        }
    elif os.environ.get("TRITON_FORCE_MFMA_AGPR"):
        compiler_options["llvm_fn_attrs"] = "amdgpu-agpr-alloc=256"
    return (grid_mn, 1), constexprs, compiler_options


def main():
    args = parse_args()
    _module, jit_kernel, kernel_label = load_kernel(args)

    import triton  # Imported after the bench module installs optional hooks.
    from prepared_kernel import PreparedKernel

    device = triton.runtime.driver.active.get_active_torch_device()
    torch.manual_seed(0)
    runtime_sets, bytes_per_set = make_runtime_arguments(args, device)
    grid, constexprs, compiler_options = launch_configuration(args, jit_kernel)
    prepared = PreparedKernel.create(
        jit_kernel,
        grid,
        runtime_sets,
        constexprs=constexprs,
        compiler_options=compiler_options,
    )

    for iteration in range(args.warmup):
        prepared(iteration)
    torch.cuda.synchronize()
    for iteration in range(args.iters):
        prepared(iteration)
    torch.cuda.synchronize()

    datatype_contract = {
        "a16w16": args.dtype,
        "a8w8": "BF8/e5m2 x BF8/e5m2 -> fp16",
        "a4w4": "MXFP4/e2m1 x MXFP4/e2m1, per-32 E8M0 scales -> bf16",
    }[args.kernel]
    print(f"route={args.route} kernel={args.kernel} label={kernel_label}")
    print(f"shape={args.M}x{args.N}x{args.K} datatype_contract={datatype_contract}")
    print(f"warmup={args.warmup} iters={args.iters} rotating_sets={args.sets}")
    print(f"bytes_per_set_mib={bytes_per_set / 1024**2:.1f}")
    print("prepared_launch=True")


if __name__ == "__main__":
    main()
