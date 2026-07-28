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

"""Delete the vector ALU from the hot loop's assembly, to measure the MFMA-only ceiling.

The whole of §5-§7 in `kernels/attention/README.md` is about fitting softmax VALU into
the shadow of the MFMAs. This asks the complementary question: **what would the loop cost
if that VALU were free?** Strip every non-MFMA vector instruction out of the loop body and
the remaining time is the memory traffic, the barriers, and the MFMA issue rate alone.

**The resulting kernel computes garbage.** It is a timing probe and nothing else -- an
upper bound to compare a real schedule against, in the same spirit as the roofline
ceilings in §5. Correctness checks will fail and should be ignored.

Enable from `bench.py` with `FA_ABLATE_VALU`:

    FA_ABLATE_VALU=dot    drop VALU only from sub-regions that contain an MFMA, so the
                          mem clusters keep theirs. Isolates the co-execution question.
    FA_ABLATE_VALU=loop   drop every non-MFMA VALU anywhere in the loop body.
    FA_ABLATE_VALU=all    `loop`, plus force every lazy-rescale block to be skipped.

`all` is the mode you want for a clean MFMA-only bound. Ablating the dot clusters destroys
`alpha`, so `warp_predicate(alpha < 1.0)` goes true and the rescale -- which the real kernel
skips on **every** iteration -- starts firing on every one instead. Measured: 0 rescale
`v_pk_mul` per iteration in the real kernel against 63 in the `dot`-ablated one. That single
artifact accounted for the whole of the apparent slowdown. `loop` does not fix it either,
because `warp_predicate(..., unlikely=True)` out-lines the cold block, so it sits outside the
loop's contiguous line range and line-based stripping never reaches it.

`all` therefore edits the guard rather than the body: each `s_and_saveexec` whose guarded
region contains the rescale becomes `s_mov exec-save` + `s_mov exec, 0`, so the following
`s_cbranch_execz` always branches and the restore still runs.

`FA_ABLATE_FRAC=<0..1>` removes that *fraction* of the dot clusters' VALU, evenly spaced, and
forces the rescale to skip. This is the practical way to sweep MFMA efficiency: start from a
build whose efficiency is low (stock LLVM, no scheduler) and remove vector work until the
shadow is empty and efficiency reaches 100%. Sweeping by *addition* does not work -- the
kernel allocates all 256 VGPRs, so filler has no safe destination and corrupting a
drain-live register faults the GPU.

`FA_FILL_VALU=<n>` then adds `n` filler VALU per dot cluster back into the MFMA shadows,
distributed evenly after the MFMAs. With `all` that turns the kernel into a dial for MFMA
efficiency: 0 gives the MFMA-only floor, and past ~6 per MFMA the shadow overflows and each
extra op costs issue cycles. It exists to measure TFLOPS as a function of achieved MFMA
efficiency, since efficiency is read back from a trace rather than assumed. The filler is
real arithmetic (`v_fma_f32`) on a VGPR the loop never references, so it draws representative
power -- a `v_nop` would understate it, and power is the thing being probed.

`FA_ABLATE_DUMP=<path>` writes the before/after assembly next to each other for
inspection. The transform reports how many instructions it removed on stderr.
"""

import hashlib
import os
import re

_MODE = os.environ.get("FA_ABLATE_VALU", "")


def _loop_bounds(lines):
    """(first, last) line indices of the inner loop body, or None.

    The back edge either targets the loop header itself or a latch block laid out just
    before it, so accept both -- guessing only one form silently swallows the whole
    function on kernels that use the other.
    """
    head = next((i for i, l in enumerate(lines) if "This Inner Loop Header" in l), None)
    if head is None:
        return None
    m = re.match(r"^(\.LBB\d+_\d+):", lines[head])
    if not m:
        return None
    label = m.group(1)
    pre, num = label.rsplit("_", 1)
    targets = {label, "%s_%d" % (pre, int(num) - 1)}
    for i in range(head + 1, len(lines)):
        t = lines[i].split(";")[0].strip()
        b = re.match(r"^s_(?:cbranch\S*|branch)\s+(\.LBB\d+_\d+)\s*$", t)
        if b and b.group(1) in targets:
            return head + 1, i
    return None


def _is_valu(op):
    return op.startswith("v_") and not op.startswith("v_mfma")


def _thin(lines, lo, hi, frac):
    """Remove `frac` of the VALU in mfma-bearing regions, evenly spaced.

    Even spacing matters: dropping a contiguous run would empty some MFMA windows while
    leaving others full, which is a different experiment from scaling the load uniformly.
    """
    def opcode(line):
        t = line.split(";")[0].strip()
        return t.split()[0] if t and not t.startswith((".", "/")) else ""

    regions, cur = [], []
    for line in lines[lo:hi]:
        cur.append(line)
        if opcode(line) == "s_barrier":
            regions.append(cur); cur = []
    regions.append(cur)

    out, removed = [], 0
    for rl in regions:
        if not any(opcode(l).startswith("v_mfma") for l in rl):
            out += rl
            continue
        vidx = [i for i, l in enumerate(rl) if _is_valu(opcode(l))]
        n_rm = int(round(len(vidx) * frac))
        # evenly spaced picks over the region's VALU
        kill = set()
        if n_rm > 0:
            for k in range(n_rm):
                kill.add(vidx[min(len(vidx) - 1, int((k + 0.5) * len(vidx) / n_rm))])
        # if rounding collided, top up from the front
        for i in vidx:
            if len(kill) >= n_rm: break
            kill.add(i)
        for i, l in enumerate(rl):
            if i in kill:
                removed += 1
                continue
            out.append(l)
    return lines[:lo] + out + lines[hi:], removed


def _filler_dest(lines, lo, hi):
    """A VGPR the loop body never mentions -- the least unsafe destination for filler.

    There is no strictly safe choice: the kernel allocates all 256 VGPRs, so every register
    is live somewhere. Registers untouched by the loop body are the ones whose corruption
    cannot break the loop's own addressing; picking one live in the drain only garbles an
    already-garbled result. Do NOT also read the destination -- a filler chain that reads
    its own result serialises on VALU latency and costs several times an issue slot
    (measured: 83% efficiency where 100% was expected).
    """
    used = set()
    for line in lines[lo:hi]:
        t = line.split(";")[0]
        for m in re.finditer(r"\bv\[(\d+):(\d+)\]", t):
            used.update(range(int(m.group(1)), int(m.group(2)) + 1))
        for m in re.finditer(r"\bv(\d+)\b", t):
            used.add(int(m.group(1)))
    free = [r for r in range(256) if r not in used]
    return (free[0] if free else None), used


def _fill(lines, lo, hi, total):
    """Insert `total` filler VALU per mfma-bearing sub-region, spread over its MFMAs."""
    dest, used = _filler_dest(lines, lo, hi)
    if dest is None or total <= 0:
        return lines, 0
    # One destination (WAW only, which does not stall) and rotating read-only sources drawn
    # from registers the loop already reads, so successive fillers are independent.
    srcs = sorted(used)[:8] or [0]
    # Rotate the destination and keep the sources on one read-only register, so successive
    # fillers are mutually INDEPENDENT. A single-register chain
    # (v_fma vX, vX, vX, vX) serialises on its own result and costs far more than an
    # issue slot -- measured 83% efficiency where 100% was expected.
    fillers = [f"\tv_fma_f32 v{dest}, v{a}, v{b}, v{c}"
               for a, b, c in zip(srcs, srcs[1:] + srcs[:1], srcs[2:] + srcs[:2])]

    def opcode(line):
        t = line.split(";")[0].strip()
        return t.split()[0] if t and not t.startswith((".", "/")) else ""

    # Split the body into barrier-delimited regions, then rewrite the mfma-bearing ones.
    regions, cur = [], []
    for line in lines[lo:hi]:
        cur.append(line)
        if opcode(line) == "s_barrier":
            regions.append(cur)
            cur = []
    regions.append(cur)

    out, added = [], 0
    for reg_lines in regions:
        idx = [i for i, l in enumerate(reg_lines) if opcode(l).startswith("v_mfma")]
        if not idx:
            out += reg_lines
            continue
        # k-th mfma carries floor((k+1)*total/n) - floor(k*total/n) fillers: as even as
        # integer division allows, and exactly `total` in sum.
        n = len(idx)
        share = [ (k + 1) * total // n - k * total // n for k in range(n) ]
        pos = {i: share[k] for k, i in enumerate(idx)}
        for i, l in enumerate(reg_lines):
            out.append(l)
            for _ in range(pos.get(i, 0)):
                out.append(fillers[added % len(fillers)])
                added += 1
    return lines[:lo] + out + lines[hi:], added


def _force_skip_rescale(lines):
    """Make every lazy-rescale predicated region unreachable, wherever it was laid out.

    A region is the rescale's if its guarded body holds the accumulator multiply. Rewriting
    the guard (not the body) is what makes this robust to the cold block being out-lined.
    """
    out, killed = [], 0
    for i, line in enumerate(lines):
        m = re.match(r"^(\s*)s_and_saveexec_(b64|b32)\s+(\S+),\s*(\S+)\s*$",
                     line.split(";")[0].rstrip())
        if m:
            body = []
            for j in range(i + 1, min(i + 120, len(lines))):
                t = lines[j].split(";")[0].strip()
                if t.startswith("s_or_b64 exec") or t.startswith("s_or_b32 exec"):
                    break
                if t and not t.startswith((".", "/")):
                    body.append(t.split()[0])
            if any(o.startswith("v_pk_mul") or o.startswith("v_mul") for o in body):
                ind, width, save = m.group(1), m.group(2), m.group(3)
                out.append(f"{ind}s_mov_{width} {save}, exec")
                out.append(f"{ind}s_mov_{width} exec, 0")
                killed += 1
                continue
        out.append(line)
    return out, killed


def ablate(text, mode=None):
    mode = mode or _MODE or "dot"
    lines = text.split("\n")
    rescale_killed = 0
    frac = float(os.environ.get("FA_ABLATE_FRAC", "-1"))
    if mode == "frac":
        lines, rescale_killed = _force_skip_rescale(lines)
        b = _loop_bounds(lines)
        if b is None:
            return text, (0, rescale_killed, 0)
        lines, nrm = _thin(lines, b[0], b[1], max(0.0, min(1.0, frac)))
        return "\n".join(lines), (nrm, rescale_killed, 0)
    if mode == "all":
        lines, rescale_killed = _force_skip_rescale(lines)
    bounds = _loop_bounds(lines)
    if bounds is None:
        return text, 0
    lo, hi = bounds

    def opcode(line):
        s = line.split(";")[0].strip()
        return s.split()[0] if s and not s.startswith((".", "/")) else ""

    keep, dropped = lines[:lo], 0
    if mode in ("loop", "all"):
        for line in lines[lo:hi]:
            if _is_valu(opcode(line)):
                dropped += 1
                continue
            keep.append(line)
    else:
        # Sub-region granularity: buffer up to each barrier, and only strip the buffer
        # if it contained an MFMA. A mem cluster keeps its VALU.
        buf, has_mfma = [], False

        def flush():
            nonlocal buf, has_mfma, dropped
            for line in buf:
                if has_mfma and _is_valu(opcode(line)):
                    dropped += 1
                    continue
                keep.append(line)
            buf, has_mfma = [], False

        for line in lines[lo:hi]:
            op = opcode(line)
            if op == "s_barrier":
                flush()
                keep.append(line)
                continue
            buf.append(line)
            if op.startswith("v_mfma"):
                has_mfma = True
        flush()
    keep += lines[hi:]
    fill = int(os.environ.get("FA_FILL_VALU", "0") or 0)
    added = 0
    if fill > 0:
        b2 = _loop_bounds(keep)
        if b2:
            keep, added = _fill(keep, b2[0], b2[1], fill)
    return "\n".join(keep), (dropped, rescale_killed, added)


def get_key():
    return (f"ablate_valu:{_MODE}:fill{os.environ.get('FA_FILL_VALU', '0')}"
            f":frac{os.environ.get('FA_ABLATE_FRAC', '')}")


def get_hash():
    return hashlib.sha256(get_key().encode("utf-8")).hexdigest()


def inspect_stages_hook(self=None, stages=None, options=None, language=None, capability=None):
    if all(a is None for a in (stages, options, language, capability)):
        return get_key(), get_hash()
    if stages is None or "amdgcn" not in stages:
        return get_key(), get_hash()

    orig = self.make_amdgcn

    def wrapper(src, metadata):
        asm = orig(src, metadata, options)
        out, n = ablate(asm)
        n, nresc, nfill = n
        dump = os.environ.get("FA_ABLATE_DUMP")
        if dump:
            with open(dump + ".orig.amdgcn", "w") as f:
                f.write(asm)
            with open(dump + ".ablated.amdgcn", "w") as f:
                f.write(out)
        import sys
        print(f"[ablate_valu] mode={_MODE or 'dot'} removed {n} VALU from the loop"
              + (f", forced {nresc} rescale block(s) to skip" if nresc else "")
              + (f", inserted {nfill} filler VALU" if nfill else ""),
              file=sys.stderr, flush=True)
        return out

    stages["amdgcn"] = wrapper
    return get_key(), get_hash()


if __name__ == "__main__":
    import sys

    src = open(sys.argv[1]).read()
    out, (n, nresc, nfill) = ablate(src, sys.argv[2] if len(sys.argv) > 2 else None)
    sys.stderr.write(f"removed {n} VALU, forced {nresc} skip, inserted {nfill} filler\n")
    sys.stdout.write(out)
