"""Check and time the AGPR-pinned v9 kernel (see README.md).

Needs Triton built from the branch listed in README.md, which adds `cd_regclass`
to `gl.amd.cdna4.mfma`. To regenerate ir_dumps/:
    TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/tmp/dump python run.py
To run with the llirSched plugin, set LLVM_PASS_PLUGIN_PATH and
LLVM_PASS_PLUGIN_KEEP_TARGET_MACHINE=1, as for the tutorial's bench.py.
"""

import argparse
import importlib
import os
import sys

import torch

# The llirSched plugin resolves LLVM symbols from libtriton, so libtriton has to
# be in the global symbol scope before the first `import triton` (as in bench.py).
if os.environ.get("LLVM_PASS_PLUGIN_PATH"):
    sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)

import triton  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "kernels", "gemm", "utils"))
sys.path.insert(0, HERE)


def main():
    parser = argparse.ArgumentParser(description="Check and time the AGPR-pinned v9 kernel")
    parser.add_argument("--K", type=int, default=8192, help="GEMM K (M = N = 4096)")
    args = parser.parse_args()
    matmul = importlib.import_module("matmul_kernel_cd_regclass").matmul

    torch.manual_seed(0)
    M = N = 4096
    K = args.K
    a = torch.rand((M, K), device="cuda", dtype=torch.float16) - 0.5
    b = torch.rand((N, K), device="cuda", dtype=torch.float16).T - 0.5
    c = matmul(a, b)
    if torch.allclose(c, torch.matmul(a, b), atol=1e-1, rtol=0):
        print(f"{M=} {N=} {K=} fp16: ✅ Triton and Torch match")
    else:
        print(f"{M=} {N=} {K=} fp16: ❌ Triton and Torch differ")
    ms = triton.testing.do_bench(lambda: matmul(a, b, c), warmup=50, rep=300)
    print(f"{ms:.4f} ms, {2 * M * N * K / ms / 1e9:.1f} TFLOPS (do_bench, warm cache)")


if __name__ == "__main__":
    main()
