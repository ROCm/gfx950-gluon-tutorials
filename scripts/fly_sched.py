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

"""Run ROCm/FlyDSL's attention kernel from **modified assembly**.

`sched_valu.py` sweeps in-loop MFMA efficiency by re-scheduling our own kernels' final
assembly. FlyDSL is the one point that sits off our cycle law -- +3.7% above it -- so the
obvious question is what its own curve looks like, and that needs the same rewrite applied
to its assembly. FlyDSL has no assembly stage to hook: it goes MLIR -> LLVM IR -> code
object entirely inside `mlir-opt`'s `gpu-module-to-binary`, with no file in between.

The way in is that the pass accepts `format=isa`, which stops at the ISA text, and that the
`gpu.binary` op it otherwise produces holds a **bare HSA code object** (`\\7FELF...`) as an
escaped string attribute. So:

    pre-binary pipeline  ->  clone, format=isa  ->  ISA text
                                                     |  sched_valu rewrite
                                                     v
    gpu.binary(format=fatbin)  <- blob substitution <- clang -x assembler

`patch()` installs this in place of `MlirCompiler.compile`, so any FlyDSL entry point picks
it up with no changes to FlyDSL itself. `FLY_SCHED_VALU=<f>` is the knob, with the same
meaning as `FA_SCHED_VALU`: negative clumps the in-loop VALU, positive spreads it, 0 leaves
the assembly alone but still round-trips it through the assembler so that the control and
the treatment are built the same way.

    FLY_SCHED_VALU=-0.5 python scripts/fly_kernel_time.py --batch 32 --seqlen 8192 --hq 8

`--selftest` checks the round trip on its own: build at f=0, run, compare against torch.
`FLY_SCHED_DUMP=<prefix>` keeps the before/after ISA and the assembled object.
"""

import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CLANG = os.environ.get("FLY_SCHED_CLANG", "/opt/rocm/llvm/bin/clang")
_ISA_RE = re.compile(r'assembly = "((?:[^"\\]|\\.)*)"')
_BIN_RE = re.compile(r'bin = "((?:[^"\\]|\\.)*)"')


def _unescape(s):
    """MLIR string-attribute escapes -> bytes."""
    out, i = bytearray(), 0
    while i < len(s):
        c = s[i]
        if c != "\\":
            out += c.encode("utf-8")
            i += 1
            continue
        n = s[i + 1]
        if n == "n":
            out.append(0x0A); i += 2
        elif n == "t":
            out.append(0x09); i += 2
        elif n in ('"', "\\"):
            out.append(ord(n)); i += 2
        else:
            out.append(int(s[i + 1:i + 3], 16)); i += 3
    return bytes(out)


def _escape(b):
    """bytes -> MLIR string-attribute escapes. Hex for everything that is not plainly
    printable, which is always accepted and avoids having to reason about which characters
    the parser would take literally."""
    safe = set(range(0x20, 0x7F)) - {ord('"'), ord("\\")}
    return "".join(chr(c) if c in safe else "\\%02X" % c for c in b)


def loop_bounds(lines):
    """(first, last) of the innermost hot loop body.

    `sched_valu` finds ours from the `This Inner Loop Header` comment that LLVM's AsmPrinter
    emits. MLIR's ISA has no comments at all, so fall back to the CFG: take the backward
    branch spanning the most instructions. On this kernel that is unambiguous -- the winner
    holds all 64 MFMA and all 8 barriers of an iteration, and the runners-up are its own
    suffix.
    """
    lab = {}
    for i, l in enumerate(lines):
        m = re.match(r"^(\.LBB\S+):", l)
        if m:
            lab[m.group(1)] = i
    best = None
    for i, l in enumerate(lines):
        m = re.match(r"^\s*s_(?:cbranch\S*|branch)\s+(\.LBB\S+)\s*$", l.split(";")[0].rstrip())
        if not m:
            continue
        t = lab.get(m.group(1))
        if t is None or t >= i:
            continue
        if best is None or i - t > best[1] - best[0]:
            best = (t, i)
    return best


def rewrite_isa(isa, frac, strip=()):
    """Apply the sched_valu rewrite to one ISA text.

    `strip` names scalar pacing opcodes to delete from the loop (`s_nop`, `s_setprio`).
    FlyDSL's VALU already fits inside its MFMA shadows, so its cycles above the ideal are
    pacing, and stripping is the only lever in this file that can raise its efficiency.
    """
    import sched_valu as S

    lines = isa.split("\n")
    b = loop_bounds(lines)
    if b is None:
        raise RuntimeError("fly_sched: could not find the inner loop in FlyDSL's ISA")
    moved = nops = 0
    if strip:
        lines, (removed, readded) = S.strip_pacing(lines, b[0], b[1], drop=tuple(strip))
        moved, nops = -removed, readded
        b = loop_bounds(lines)
    if frac:
        S._NOPS[0] = 0
        lines, moved = S.schedule(lines, b[0], b[1], frac)
        nops += S._NOPS[0]
    return "\n".join(lines), (moved, nops)


def assemble(isa, arch="gfx950"):
    """ISA text -> bare HSA code object bytes."""
    with tempfile.TemporaryDirectory() as d:
        src, obj = os.path.join(d, "k.s"), os.path.join(d, "k.hsaco")
        with open(src, "w") as f:
            f.write(isa)
        r = subprocess.run([CLANG, "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
                            f"-mcpu={arch}", "-nogpulib", "-Wl,--no-undefined",
                            "-o", obj, src], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("fly_sched: assembling the rewritten ISA failed:\n"
                               + (r.stderr or r.stdout)[-2000:])
        with open(obj, "rb") as f:
            return f.read()


def patch():
    """Install the assembly-rewriting compile step in place of FlyDSL's."""
    frac = float(os.environ.get("FLY_SCHED_VALU", "0") or 0)
    strip = tuple(x.strip() for x in os.environ.get("FLY_SCHED_STRIP", "").split(",") if x.strip())
    dump = os.environ.get("FLY_SCHED_DUMP")

    # FlyDSL caches the finished code object on disk, and nothing about the rewrite is in its
    # cache key -- so a second run at a different fraction silently replays the first one's
    # binary. That is not a hypothetical: the first sweep taken with this harness reported
    # f=0 and f=1.0 as 82.9%/1316 and 83.0%/1318, which was one kernel measured twice. Give
    # each fraction its own cache namespace.
    base = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR") or os.path.expanduser("~/.flydsl/cache")
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = os.path.join(
        base, f"fly_sched_{frac}_{'-'.join(strip) or 'none'}")

    from flydsl._mlir import ir
    from flydsl._mlir.passmanager import PassManager
    from flydsl.compiler import jit_function as J

    def compile_patched(cls, module, *, arch="", func_name="", link_libs=None):
        module.operation.verify()
        backend = J.get_backend(arch=arch)
        hints = J.CompilationContext.get_compile_hints()
        module = ir.Module.parse(module.operation.get_asm(enable_debug_info=False))
        backend.lower_compile_hints(module, compile_hints=hints)
        cfg = J._pipeline_fragments_for_mode(backend, compile_hints=hints)
        if cfg.external:
            raise RuntimeError("fly_sched: FLYDSL_COMPILE_LLVM_DIR external codegen is not "
                               "supported; unset it")
        pre, binfrag = cfg.fragments[:-1], cfg.fragments[-1]
        if not binfrag.strip().startswith("gpu-module-to-binary"):
            raise RuntimeError(f"fly_sched: unexpected final pipeline fragment {binfrag!r}")

        J._run_pipeline(module, pre, verifier=False, print_after_all=False)
        pre_asm = module.operation.get_asm(enable_debug_info=False)

        # 1. the ISA, from a clone so the real module is untouched
        clone = ir.Module.parse(pre_asm, context=module.context)
        pm = PassManager.parse(
            "builtin.module(gpu-module-to-binary{format=isa opts=\"\" section= toolkit=})",
            context=module.context)
        pm.run(clone.operation)
        m = _ISA_RE.search(clone.operation.get_asm(enable_debug_info=False))
        if not m:
            raise RuntimeError("fly_sched: no `assembly = ...` in the format=isa output")
        isa = _unescape(m.group(1)).decode("utf-8", "replace")

        # 2. rewrite, 3. assemble
        new_isa, (moved, nops) = rewrite_isa(isa, frac, strip=strip)
        obj = assemble(new_isa, arch=backend.target.arch)
        if dump:
            open(dump + ".orig.s", "w").write(isa)
            open(dump + ".sched.s", "w").write(new_isa)
            open(dump + ".hsaco", "wb").write(obj)

        # 4. the real binary pass, then substitute the code object inside it
        J._run_pipeline(module, [binfrag], verifier=False, print_after_all=False)
        asm = module.operation.get_asm(enable_debug_info=False)
        mb = _BIN_RE.search(asm)
        if not mb:
            raise RuntimeError("fly_sched: no `bin = ...` in the gpu.binary output")
        old = _unescape(mb.group(1))
        if not old.startswith(b"\x7fELF"):
            raise RuntimeError("fly_sched: gpu.binary holds something other than a bare ELF; "
                               "substitution would need unbundling")
        if dump:
            open(dump + ".stock.hsaco", "wb").write(old)
        asm = asm[:mb.start(1)] + _escape(obj) + asm[mb.end(1):]
        out = ir.Module.parse(asm, context=module.context)
        same = " (byte-identical to FlyDSL's own)" if obj == old else ""
        print(f"[fly_sched] frac={frac} strip={strip or '()'} moved {moved} VALU, {nops} s_nop, "
              f"code object {len(old)} -> {len(obj)} bytes{same}",
              file=sys.stderr, flush=True)
        return out

    J.MlirCompiler.compile = classmethod(compile_patched)


def _selftest():
    import torch
    frac = float(os.environ.get("FLY_SCHED_VALU", "0") or 0)
    sys.path.insert(0, os.environ.get("FLYDSL_ROOT", "/root/FlyDSL"))
    patch()
    from kernels.attention.flash_attn_gfx950 import build_flash_attn_dualwave_swp_module
    launch = build_flash_attn_dualwave_swp_module(
        num_heads=8, head_dim=128, causal=False, dtype_str="bf16", waves_per_eu=2,
        daz=True, dualwave_swp_lazy_rescale=True, dualwave_swp_setprio=True,
        dualwave_swp_enable_stagger=True)
    B, S, H, D = 2, 2048, 8, 128
    mk = lambda: torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")
    q, k, v = mk(), mk(), mk()
    o = torch.empty_like(q)
    launch(q, k, v, o, B, S)
    torch.cuda.synchronize()
    qr, kr, vr = (t.transpose(1, 2) for t in (q, k, v))
    ref = torch.nn.functional.scaled_dot_product_attention(
        qr.float(), kr.float(), vr.float(), is_causal=False, scale=D ** -0.5)
    err = (o.transpose(1, 2).float() - ref).abs().max().item()
    print(f"[fly_sched selftest] frac={frac} max_err={err:.3e} "
          f"{'OK' if err < 1e-2 else 'MISMATCH'}")
    return 0 if err < 1e-2 else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    if "--report" in sys.argv:
        lines = open(sys.argv[1]).read().split("\n")
        b = loop_bounds(lines)
        print(f"inner loop lines {b[0]}..{b[1]}")
        import sched_valu as S
        S.analyse(lines, b[0], b[1], verbose=True)
    else:
        sys.exit("usage: fly_sched.py --selftest | <isa.s> --report")
