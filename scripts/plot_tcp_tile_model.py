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

"""
Regenerate the seven figures of docs/tcp_bound_tile_model.md (docs/images/tcp_*.svg):

    tcp_budget.svg          — one CU: waves, VMEM queue, TCP, and the round trip a load holds its slot for
    tcp_saturation.svg      — 256 full TCPs against 8 TB/s: why TCP-bound is bandwidth-bound
    tcp_test_timelines.svg  — the test on 256x256 (passes) and 128x128 (fails, 70% ceiling)
    tcp_tiles_vs_L.svg      — the eight a16w16 tiles' MFMA cycles per 44 KB against L = 1000
    tcp_who_waits.svg       — stretched buffer_load (bandwidth-bound) vs long s_waitcnt (latency-bound)
    tcp_prefetch_depth.svg  — 128x128 with 2 and 3 LDS buffers against the round trip
    tcp_decision_flow.svg   — the checklist as a flow

Hand-laid SVG with the vocabulary of kernels/attention/images/*.svg: 760 px wide, system-ui,
green = MFMA, orange = buffer_load, yellow = ds_read, red = waiting, grey dashed = idle. The numbers
are the ones in the page (C = 44 KB, L = 1000 cycles, fp16 16x16x32 MFMA, 4 waves).

    python scripts/plot_tcp_tile_model.py            # writes docs/images/
    python scripts/plot_tcp_tile_model.py <outdir>
"""

import os
import sys

OUT = (
    sys.argv[1]
    if len(sys.argv) > 1
    else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "images")
)
FONT = "system-ui, -apple-system, Segoe UI, Roboto, sans-serif"

# palette (identical to the attention figures)
GREEN = ("#cde8d3", "#4c9a5a")  # MFMA
ORANGE = ("#f7e0c6", "#d9822b")  # buffer_load
YELLOW = ("#f7eec6", "#d4a72c")  # ds_read
RED = ("#f7d2cf", "#c8443c")  # waiting
BLUE = ("#cfe0f7", "#3b7dd8")
GREY_T = "#767676"
INK = "#1a1a1a"
IDLE = ("#ececec", "#b9b9b9")

STYLE = f"""
    .h    {{ font-size: 12.5px; font-weight: 600; fill: {GREY_T}; }}
    .lbl  {{ font-size: 11.5px; fill: {GREY_T}; }}
    .lblb {{ font-size: 11.5px; font-weight: 600; fill: {GREY_T}; }}
    .in   {{ font-size: 11px; fill: {INK}; }}
    .ins  {{ font-size: 9.5px; fill: {INK}; }}
    .inb  {{ font-size: 11.5px; font-weight: 600; fill: {INK}; }}
    .note {{ font-size: 11.5px; fill: {GREY_T}; }}
    .bad  {{ font-size: 11.5px; fill: {RED[1]}; }}
    .badb {{ font-size: 11.5px; font-weight: 600; fill: {RED[1]}; }}
    .ok   {{ font-size: 11.5px; fill: {GREEN[1]}; }}
    .okb  {{ font-size: 11.5px; font-weight: 600; fill: {GREEN[1]}; }}
    .warn {{ font-size: 11.5px; fill: {YELLOW[1]}; }}
    .warnb{{ font-size: 11.5px; font-weight: 600; fill: {YELLOW[1]}; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 10.5px; fill: {INK}; }}
    .monog{{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 10.5px; fill: {GREY_T}; }}
    .idle {{ fill: {IDLE[0]}; stroke: {IDLE[1]}; stroke-dasharray: 3 3; }}
    .wire {{ fill: none; stroke: {GREY_T}; stroke-width: 1.2; }}
    .dash {{ fill: none; stroke: {GREY_T}; stroke-width: 1; stroke-dasharray: 3 3; }}
    .brk  {{ fill: none; stroke: {INK}; stroke-width: 1; }}
    .wireO{{ fill: none; stroke: {ORANGE[1]}; stroke-width: 1.4; }}
"""


def svg_open(w, h, title):
    title = (
        title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'font-family="{FONT}" role="img" aria-label="{title}">\n'
        f"  <style>{STYLE}  </style>\n"
        f"  <defs>\n"
        f'    <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{GREY_T}"/></marker>\n'
        f'    <marker id="arrR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{RED[1]}"/></marker>\n'
        f'    <marker id="arrO" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{ORANGE[1]}"/></marker>\n'
        f"  </defs>\n"
        f'  <rect x="0" y="0" width="{w}" height="{h}" fill="#ffffff"/>\n'
    )


def text(x, y, s, cls="lbl", anchor=None, extra=""):
    a = f' text-anchor="{anchor}"' if anchor else ""
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f'  <text class="{cls}" x="{x}" y="{y}"{a}{extra}>{s}</text>\n'


def rect(x, y, w, h, col, extra="", rx=0):
    r = f' rx="{rx}"' if rx else ""
    return f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{col[0]}" stroke="{col[1]}"{r}{extra}/>\n'


def box(x, y, w, h, col, label, cls="in", rx=0, dy=0):
    return rect(x, y, w, h, col, rx=rx) + text(x + w / 2, y + h / 2 + 4 + dy, label, cls, "middle")


def line(x1, y1, x2, y2, cls="wire", extra=""):
    return f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="{cls}"{extra}/>\n'


def arrow(x1, y1, x2, y2, cls="wire", marker="arr"):
    return line(x1, y1, x2, y2, cls, f' marker-end="url(#{marker})"')


def path(d, cls="wire", extra=""):
    return f'  <path d="{d}" class="{cls}"{extra}/>\n'


def bracket(x1, x2, y, label, cls="lbl", up=True, tick=5):
    """horizontal bracket with a centred label above (up) or below"""
    s = path(
        f"M {x1} {y + (tick if up else -tick)} V {y} H {x2} V {y + (tick if up else -tick)}", "brk"
    )
    s += text((x1 + x2) / 2, y - 5 if up else y + 14, label, cls, "middle")
    return s


def write(name, body):
    p = os.path.join(OUT, name)
    with open(p, "w") as f:
        f.write(body)
    print("wrote", p)


# ---------------------------------------------------------------------------
# Figure 1: the per-CU budget — what a buffer_load passes through, where 44 KB comes from
# ---------------------------------------------------------------------------
def fig_budget():
    W, H = 760, 304
    s = svg_open(
        W,
        H,
        "One CU: four waves issue buffer_loads into a 12-entry VMEM queue and a 32 KB TCP; "
        "a load holds its slot for the whole round trip L, so at most 44 KB is in flight per CU.",
    )
    s += text(
        8,
        16,
        "One CU. A buffer_load holds a 1 KB slot from issue until its data is back: the whole round trip L.",
        "h",
    )

    wx, wy, ww, wh = 8, 46, 70, 26
    for i in range(4):
        y = wy + i * 34
        s += box(wx, y, ww, wh, GREEN, f"wave {i}")
        s += arrow(wx + ww, y + wh / 2, 150, y + wh / 2)
    s += text(wx, wy + 4 * 34 + 8, "one per SIMD", "lbl")
    s += text(wx, wy + 4 * 34 + 22, "dwordx4 = 1 KB", "lbl")

    qx, qy, cw, ch = 152, 46, 24, 20
    for i in range(12):
        c, r = divmod(i, 6)
        s += rect(qx + c * cw, qy + r * ch, cw, ch, ORANGE)
    s += text(qx + cw, qy - 8, "VMEM queue", "lblb", "middle")
    s += text(qx + cw, qy + 6 * ch + 14, "12 x 1 KB", "lbl", "middle")
    s += text(qx + cw, qy + 6 * ch + 28, "issued,", "lbl", "middle")
    s += text(qx + cw, qy + 6 * ch + 42, "not started", "lbl", "middle")

    tx, ty = 236, 46
    for i in range(32):
        c, r = divmod(i, 8)
        s += rect(tx + c * cw, ty + r * 15, cw, 15, ORANGE)
    s += text(tx + 2 * cw, ty - 8, "TCP (L1)", "lblb", "middle")
    s += text(tx + 2 * cw, ty + 8 * 15 + 14, "32 x 1 KB in flight", "lbl", "middle")
    s += arrow(qx + 2 * cw, qy + 6 * ch / 2, tx, qy + 6 * ch / 2)

    mx, my = 420, 46
    s += box(mx, my, 90, 26, BLUE, "L2")
    s += box(mx + 110, my, 90, 26, BLUE, "HBM / MALL")
    s += arrow(tx + 4 * cw, my + 13, mx, my + 13)
    s += text((tx + 4 * cw + mx) / 2, my + 8, "request", "lbl", "middle")
    s += arrow(mx + 90, my + 13, mx + 110, my + 13)
    ry = 150
    s += path(
        f"M {mx + 155} {my + 26} V {ry} H {tx + 4 * cw + 12}", "wire", ' marker-end="url(#arr)"'
    )
    s += text(mx + 155 + 6, (my + 26 + ry) / 2 + 4, "data back", "lbl")
    s += text(
        tx + 4 * cw + 12, ry + 18, "round trip L ≈ 1000 cycles at K = 8192 (L2-friendly),", "lbl"
    )
    s += text(
        tx + 4 * cw + 12,
        ry + 33,
        "longer as L2 misses rise. The slot retires only when the data is back.",
        "lbl",
    )

    s += bracket(
        qx, tx + 4 * cw, 222, "C = 12 KB + 32 KB = 44 KB in flight per CU", "inb", up=False
    )
    s += text(
        8,
        268,
        "Both full → the next buffer_load of any wave waits until the oldest slot retires.",
        "badb",
    )
    s += text(
        8,
        286,
        "Little's law: the CU cannot move more than C / L bytes per cycle, whatever the kernel does.",
        "note",
    )
    s += "</svg>\n"
    write("tcp_budget.svg", s)


# ---------------------------------------------------------------------------
# Figure 2: 256 full TCPs against 8 TB/s — a full TCP is a saturated memory system
# ---------------------------------------------------------------------------
def fig_saturation():
    W, H = 760, 250
    s = svg_open(
        W,
        H,
        "256 CUs each with 44 KB in flight demand about 27 TB/s at a 1000-cycle round trip, "
        "more than three times what HBM delivers, so the round trip stretches until in-flight bytes over L "
        "equals the delivered bandwidth.",
    )
    s += text(
        8,
        16,
        "Every TCP full at once is more than the memory system can serve: L stretches to match.",
        "h",
    )

    cx, cy = 8, 40
    for i in range(5):
        s += rect(cx + i * 4, cy + i * 4, 120, 70, GREEN if i == 4 else ("#ffffff", GREEN[1]))
    s += text(cx + 16 + 60, cy + 16 + 28, "CU", "inb", "middle")
    s += text(cx + 16 + 60, cy + 16 + 46, "44 KB in flight", "in", "middle")
    s += text(cx + 16 + 60, cy + 16 + 62, "per L cycles", "in", "middle")
    s += text(cx + 76, cy + 110, "x 256", "inb", "middle")

    ax1, ax2, ay = 160, 400, 91
    s += arrow(ax1, ay, ax2, ay, "wire")
    s += text((ax1 + ax2) / 2, ay - 22, "demand at L = 1000 cycles", "lbl", "middle")
    s += text((ax1 + ax2) / 2, ay - 8, "256 x 44 KB per 1000 cycles", "lbl", "middle")
    s += text((ax1 + ax2) / 2, ay + 18, "≈ 27 TB/s at 2.4 GHz", "inb", "middle")

    hx, hy, hw, hh = 406, 66, 150, 50
    s += rect(hx, hy, hw, hh, BLUE, rx=4)
    s += text(hx + hw / 2, hy + 22, "HBM", "inb", "middle")
    s += text(hx + hw / 2, hy + 39, "delivers 8 TB/s", "in", "middle")

    s += arrow(hx + hw, ay, 600, ay, "wire")
    s += rect(602, 66, 150, 50, RED, rx=4)
    s += text(677, 88, "requests queue up:", "in", "middle")
    s += text(677, 105, "L stretches", "inb", "middle")

    s += text(
        8,
        172,
        "L  =  bytes in flight / delivered bandwidth  =  256 x 44 KB / 8 TB/s  ≈  3400 cycles if nothing hit L2",
        "mono",
    )
    s += text(
        8,
        196,
        "So TCP-bound and bandwidth-bound are one condition: every CU issues as fast as the hardware allows and the",
        "note",
    )
    s += text(
        8,
        212,
        "memory system cannot absorb more. The cap on the CU side is C; the L the CU sees is set by the other side.",
        "note",
    )
    s += text(
        8,
        236,
        "L2 hits shorten the queue: at K = 8192 enough requests are served from L2 that L stays near 1000 cycles.",
        "note",
    )
    s += "</svg>\n"
    write("tcp_saturation.svg", s)


# ---------------------------------------------------------------------------
# Figure 3: the test on two tiles — compute-bound vs bandwidth-bound timelines
# ---------------------------------------------------------------------------
def fig_test():
    W, H = 760, 392
    L, C = 1000, 44
    scale = 0.30
    x0 = 118
    s = svg_open(
        W,
        H,
        "The test on two tiles. 256x256 issues 64 KB over 2048 MFMA cycles, so one round trip holds 31 KB, "
        "under the 44 KB cap: no stall. 128x128 issues 32 KB over 512 cycles, 62 KB per round trip, over the cap: "
        "the loads issue at the cap rate, the K tile stretches from 512 to 727 cycles and the MFMA pipe idles 30%.",
    )
    s += text(
        8,
        16,
        "One K tile, loads spread evenly over its MFMAs. Does one round trip hold more than C = 44 KB of them?",
        "h",
    )

    def cycles_axis(y, n, step=500):
        s_ = ""
        for c in range(0, n + 1, step):
            x = x0 + c * scale
            s_ += line(x, y, x, y + 4, "wire")
            s_ += text(x, y + 15, str(c), "lbl", "middle")
        s_ += line(x0, y, x0 + n * scale, y, "wire")
        return s_

    def ticks(y, n, span):
        s_ = ""
        for i in range(n):
            s_ += rect(x0 + i * (span / n) * scale, y, 2.2, 14, ORANGE, ' stroke-width="0.6"')
        return s_

    # ---- A: 256x256
    yA = 40
    T, B = 2048, 64
    s += text(8, yA + 10, "256 x 256", "inb")
    s += text(8, yA + 25, "T = 2048 cycles", "lbl")
    s += text(8, yA + 40, "B = 64 KB", "lbl")
    s += rect(x0, yA + 4, T * scale, 18, GREEN)
    s += text(
        x0 + T * scale / 2,
        yA + 17,
        "128 MFMAs per wave x 16 cycles = 2048 cycles, no gaps",
        "in",
        "middle",
    )
    ly = yA + 44
    s += text(8, ly + 12, "loads (CU)", "lbl")
    s += ticks(ly, B, T)
    s += rect(x0, ly - 3, L * scale, 20, ("none", INK), ' stroke-dasharray="4 2"')
    s += text(
        x0 + L * scale + 6,
        yA + 36,
        f"← one round trip (1000 cycles) holds {int(B * L / T)} KB  <  44 KB",
        "okb",
    )
    s += cycles_axis(ly + 22, T)
    s += text(
        x0,
        ly + 50,
        "compute-bound (T x C / B = 1408 ≥ L): a slot is always free when the next load issues; the K tile takes T.",
        "ok",
    )

    # ---- B: 128x128 as scheduled
    yB = 174
    T, B = 512, 32
    s += text(8, yB + 10, "128 x 128", "inb")
    s += text(8, yB + 25, "T = 512 cycles", "lbl")
    s += text(8, yB + 40, "B = 32 KB", "lbl")
    s += rect(x0, yB + 4, T * scale, 18, GREEN)
    s += text(x0 + T * scale / 2, yB + 17, "32 MFMAs = 512 cycles", "in", "middle")
    s += text(x0 + T * scale + 8, yB + 17, "as scheduled", "lblb")
    ly = yB + 44
    s += text(8, ly + 12, "loads (CU)", "lbl")
    s += ticks(ly, B, T)
    s += rect(x0, ly - 3, L * scale, 20, ("none", INK), ' stroke-dasharray="4 2"')
    s += text(
        x0 + L * scale + 6,
        ly + 12,
        f"one round trip would hold {int(B * L / T)} KB  >  44 KB",
        "badb",
    )

    # ---- B as executed
    yC = 268
    Tex = B * L / C
    s += text(8, yC + 10, "as executed", "lblb")
    s += text(8, yC + 25, "44 KB per", "lbl")
    s += text(8, yC + 40, "1000 cycles", "lbl")
    groups = 8
    busy = T / groups * scale
    idle = (Tex - T) / groups * scale
    x = x0
    for g in range(groups):
        s += rect(x, yC + 4, busy, 18, GREEN)
        x += busy
        s += rect(x, yC + 4, idle, 18, IDLE, ' stroke-dasharray="3 3"')
        x += idle
    s += text(x0 + Tex * scale + 8, yC + 17, "K tile = B x L / C = 727 cycles", "badb")
    ly = yC + 44
    s += text(8, ly + 12, "loads (CU)", "lbl")
    s += ticks(ly, B, Tex)
    s += rect(x0, ly - 3, L * scale, 20, ("none", INK), ' stroke-dasharray="4 2"')
    s += text(x0 + L * scale + 6, ly + 12, "holds exactly 44 KB: each load waits for a slot", "bad")
    s += cycles_axis(ly + 22, 1000)
    s += text(
        x0,
        ly + 50,
        "bandwidth-bound (T x C / B = 704 < L): paced by C / L. MFMA efficiency ≤ T x C / (B x L) = 512 / 727 = 70%.",
        "bad",
    )
    s += text(
        x0,
        ly + 66,
        "(where the idle cycles fall depends on the schedule; their total is what the cap fixes)",
        "note",
    )
    s += "</svg>\n"
    write("tcp_test_timelines.svg", s)


# ---------------------------------------------------------------------------
# Figure 4: the eight tiles — MFMA cycles per 44 KB against L = 1000
# ---------------------------------------------------------------------------
def fig_tiles():
    rows = [
        ("256 x 256", 1408, 1536, 100, 96, "compute"),
        ("256 x 128", 939, 1024, 94, 83, "boundary"),
        ("128 x 256", 939, 1024, 94, 84, "boundary"),
        ("128 x 128", 704, 768, 70, 69, "bw"),
        ("256 x 64", 563, 614, 56, 55, "bw"),
        ("64 x 256", 563, 614, 56, 57, "bw"),
        ("128 x 64", 469, 512, 47, 47, "bw"),
        ("64 x 128", 469, 512, 47, 47, "bw"),
    ]
    W, H = 760, 380
    s = svg_open(
        W,
        H,
        "Bar chart: MFMA cycles per 44 KB of loads for the eight a16w16 tiles against the 1000-cycle round trip. "
        "256x256 clears it at 1408; 256x128 and 128x256 reach 939 (1024 with a 48 KB budget), on the line; "
        "128x128 and smaller fall short and their predicted efficiency ceilings 70, 56 and 47 percent match the measured "
        "69, 55 to 57 and 47 percent.",
    )
    s += text(
        8,
        16,
        "The eight a16w16 tiles: MFMA cycles per 44 KB of loads (= 11 x BM x BN / (BM + BN)) against L = 1000.",
        "h",
    )
    x0, xmax = 92, 1600
    scale = 340 / xmax
    y0, rh, gap = 52, 24, 10
    ybot = y0 + len(rows) * (rh + gap)
    for c in (0, 500, 1000, 1500):
        x = x0 + c * scale
        s += line(x, y0 - 6, x, ybot, "dash" if c else "wire")
        s += text(x, ybot + 14, str(c), "lbl", "middle")
    s += text(x0 + 1000 * scale, y0 - 12, "L = 1000 cycles", "inb", "middle")
    s += line(
        x0 + 1000 * scale,
        y0 - 6,
        x0 + 1000 * scale,
        ybot,
        "wire",
        f' stroke="{INK}" stroke-width="1.4"',
    )
    col = {"compute": GREEN, "boundary": YELLOW, "bw": RED}
    verdict = {
        "compute": ("compute-bound", "okb"),
        "boundary": ("on the line", "warnb"),
        "bw": ("bandwidth-bound", "badb"),
    }
    vx, ex = 452, 580
    for i, (tile, c44, c48, ceil, meas, v) in enumerate(rows):
        y = y0 + i * (rh + gap)
        s += text(x0 - 8, y + rh / 2 + 4, tile, "in", "end")
        s += rect(x0, y, c44 * scale, rh, col[v])
        s += rect(
            x0 + c44 * scale,
            y,
            (c48 - c44) * scale,
            rh,
            ("none", col[v][1]),
            ' stroke-dasharray="2 2"',
        )
        s += text(x0 + c44 * scale - 4, y + rh / 2 + 4, str(c44), "ins", "end")
        s += text(x0 + c48 * scale + 5, y + rh / 2 + 4, f"{c48}", "ins")
        s += text(vx, y + rh / 2 + 4, verdict[v][0], verdict[v][1])
        s += text(ex, y + rh / 2 + 4, f"{ceil}%", "in")
        s += text(ex + 60, y + rh / 2 + 4, f"{meas}%", "in")
    s += text(vx, y0 - 26, "verdict", "lblb")
    s += text(vx, y0 - 12, "at L = 1000", "lblb")
    s += text(ex, y0 - 26, "in-loop MFMA efficiency", "lblb")
    s += text(ex, y0 - 12, "ceiling", "lblb")
    s += text(ex + 60, y0 - 12, "measured", "lblb")
    ly = ybot + 34
    s += rect(x0, ly, 18, 12, ("#ffffff", GREY_T))
    s += text(x0 + 24, ly + 10, "solid: C = 44 KB", "lbl")
    s += rect(x0 + 130, ly, 18, 12, ("none", GREY_T), ' stroke-dasharray="2 2"')
    s += text(x0 + 154, ly + 10, "dashed extension: C = 48 KB (queue counted per wave)", "lbl")
    s += text(
        8,
        ly + 32,
        "ceiling = T x C / (B x L), capped at 100%; measured = tile-parametric v9 under the LLIR scheduler, K = 8192, 256 workgroups.",
        "note",
    )
    s += text(
        8,
        ly + 48,
        "Only the two boundary tiles change side with the choice of C; there the loop schedule decides. Below them no",
        "note",
    )
    s += text(8, ly + 64, "schedule can lift the ceiling: it cannot shrink B / T.", "note")
    s += "</svg>\n"
    write("tcp_tiles_vs_L.svg", s)


# ---------------------------------------------------------------------------
# Figure 5: who waits on vmcnt — bandwidth-bound vs latency-bound traces
# ---------------------------------------------------------------------------
def fig_who_waits():
    W, H = 760, 352
    s = svg_open(
        W,
        H,
        "Two instruction streams of one wave. Bandwidth-bound: the buffer_load itself stretches because the TCP and "
        "queue are full, the s_waitcnt before the ds_reads is short. Latency-bound: the buffer_loads issue promptly, "
        "the s_waitcnt before the ds_reads is long because the load it depends on has not landed.",
    )
    s += text(
        8,
        16,
        "Same instructions, one wave, two stalls. Which instruction carries the wait names the bound.",
        "h",
    )
    h = 26

    def stream(y, label, segs):
        s_ = text(8, y + h / 2 + 4, label, "inb")
        x = 118
        xs = []
        for w, col, txt, cls in segs:
            s_ += rect(x, y, w, h, col)
            if txt:
                s_ += text(x + w / 2, y + h / 2 + 4, txt, cls, "middle")
            xs.append((x, w))
            x += w + 2
        return s_, xs

    yA = 48
    s += text(118, yA - 8, "A. bandwidth-bound — the TCP and the queue are full", "badb")
    segs = [
        (40, GREEN, "MFMA", "ins"),
        (40, GREEN, "MFMA", "ins"),
        (250, ORANGE, "buffer_load — waits for a free slot", "ins"),
        (40, GREEN, "MFMA", "ins"),
        (40, GREEN, "MFMA", "ins"),
        (34, RED, "wait", "ins"),
        (46, YELLOW, "ds_read", "ins"),
        (46, YELLOW, "ds_read", "ins"),
    ]
    t, xs = stream(yA, "wave 0", segs)
    s += t
    bx, bw = xs[2]
    s += bracket(bx, bx + bw, yA + h + 6, "stretched buffer_load: no free slot", "bad", up=False)
    wx, ww = xs[5]
    s += text(wx + ww / 2, yA + h + 20, "short s_waitcnt", "note", "middle")

    yB = 160
    s += text(118, yB - 8, "B. latency-bound — the loads issue promptly, but too late", "badb")
    segs = [
        (40, GREEN, "MFMA", "ins"),
        (40, GREEN, "MFMA", "ins"),
        (36, ORANGE, "load", "ins"),
        (40, GREEN, "MFMA", "ins"),
        (40, GREEN, "MFMA", "ins"),
        (250, RED, "s_waitcnt vmcnt — data not landed yet", "ins"),
        (46, YELLOW, "ds_read", "ins"),
        (46, YELLOW, "ds_read", "ins"),
    ]
    t, xs = stream(yB, "wave 0", segs)
    s += t
    bx, bw = xs[5]
    s += bracket(
        bx, bx + bw, yB + h + 6, "long s_waitcnt: the round trip is exposed", "bad", up=False
    )
    lx, lw = xs[2]
    s += text(lx + lw / 2, yB + h + 20, "issues on time", "note", "middle")

    y = 240
    s += text(
        8, y, "Both are 'waiting on vmcnt' in a stall summary. The fixes are opposite:", "lblb"
    )
    s += text(8, y + 22, "A", "badb")
    s += text(
        26,
        y + 22,
        "spread the loads evenly first: a burst stalls even when the average rate passes. Still stalling?",
        "in",
    )
    s += text(
        26,
        y + 38,
        "Then the loop is at the cap C / L, and only the tile shape (BM x BN / (BM + BN)), the element type",
        "in",
    )
    s += text(
        26, y + 54, "or L itself (L2 locality) move it. More LDS buffers do nothing here.", "in"
    )
    s += text(8, y + 78, "B", "badb")
    s += text(
        26,
        y + 78,
        "deepen the pipeline: issue the loads (num_stages − 1) tiles ahead so that (num_stages − 1) x T ≥ L.",
        "in",
    )
    s += text(
        26,
        y + 94,
        "That moves the wait from the s_waitcnt to the TCP cap, and the loop lands on the test's verdict.",
        "in",
    )
    s += "</svg>\n"
    write("tcp_who_waits.svg", s)


# ---------------------------------------------------------------------------
# Figure 6: prefetch depth — (num_stages - 1) x T against L, 128x128 example
# ---------------------------------------------------------------------------
def fig_depth():
    W, H = 760, 340
    T, L = 512, 1000
    scale = 0.2
    x0 = 118
    s = svg_open(
        W,
        H,
        "Prefetch depth on a 128x128 tile, T = 512 cycles. With 2 LDS buffers the loads for tile n+1 are issued one tile "
        "ahead, 512 cycles before they are read, so each tile waits 488 cycles for the 1000-cycle round trip. With 3 "
        "buffers they are issued two tiles ahead, 1024 cycles, and arrive in time; the loop then meets the TCP cap instead.",
    )
    s += text(
        8,
        16,
        "128 x 128 (T = 512): how far ahead the loads issue decides whether the ds_reads wait.",
        "h",
    )

    def tiles(y, n, label, stretched=0):
        s_ = text(8, y + 15, label, "inb")
        x = x0
        for i in range(n):
            s_ += rect(x, y, T * scale, 22, GREEN)
            s_ += text(x + T * scale / 2, y + 15, f"tile {i}", "ins", "middle")
            x += T * scale
            if stretched:
                s_ += rect(x, y, stretched * scale, 22, RED)
                s_ += text(x + stretched * scale / 2, y + 15, "wait", "ins", "middle")
                x += stretched * scale
        return s_

    yA = 46
    s += text(
        x0, yA - 6, "2 buffers: loads issue one tile ahead → (num_stages − 1) x T = 512 < L", "badb"
    )
    s += tiles(yA + 4, 3, "num_stages = 2", stretched=L - T)
    ya = yA + 40
    s += arrow(x0, ya, x0 + L * scale, ya, "wireO", "arrO")
    s += text(
        x0, ya + 16, "round trip L = 1000 for tile 1's loads, issued at the start of tile 0", "lbl"
    )
    s += line(x0 + T * scale, yA + 4, x0 + T * scale, ya + 22, "dash")
    s += text(
        x0 + T * scale + 4,
        ya + 34,
        "tile 1 wants its ds_reads here (T = 512): it waits 488 cycles",
        "bad",
    )

    yB = 164
    s += text(
        x0,
        yB - 6,
        "3 buffers: loads issue two tiles ahead → (num_stages − 1) x T = 1024 ≥ L",
        "okb",
    )
    s += tiles(yB + 4, 4, "num_stages = 3")
    ya = yB + 40
    s += arrow(x0, ya, x0 + L * scale, ya, "wireO", "arrO")
    s += text(
        x0, ya + 16, "round trip L = 1000 for tile 2's loads, issued at the start of tile 0", "lbl"
    )
    s += line(x0 + 2 * T * scale, yB + 4, x0 + 2 * T * scale, ya + 22, "dash")
    s += text(
        x0 + 2 * T * scale + 4,
        ya + 34,
        "tile 2 reads here (2T = 1024): the data landed 24 cycles earlier",
        "ok",
    )

    y = 262
    s += text(
        8,
        y,
        "Deepening does not make the loop compute-bound; it moves the bound. With two tiles of loads in flight,",
        "note",
    )
    s += text(
        8,
        y + 16,
        "the CU asks for 62 KB per round trip against the 44 KB cap, and 128 x 128 lands on the §2 verdict: a 70% ceiling.",
        "note",
    )
    s += text(8, y + 40, "LDS decides how deep you can go (160 KB, unpadded tiles):", "note")
    s += text(
        8,
        y + 56,
        "256 x 256 → 2 buffers,  256 x 128 → 3,  128 x 128 → 5,  256 x 64 → 4,  128 x 64 → 6.",
        "note",
    )
    s += "</svg>\n"
    write("tcp_prefetch_depth.svg", s)


# ---------------------------------------------------------------------------
# Figure 7: the checklist as a decision flow
# ---------------------------------------------------------------------------
def fig_flow():
    W, H = 760, 384
    s = svg_open(
        W,
        H,
        "Decision flow for a new tile: compute B and T; if T times C over B is under L the loop is bandwidth-bound with "
        "ceiling T C over B L; else if num_stages minus one times T is under L it is latency-bound, deepen the pipeline and "
        "re-test; else if it still waits read which instruction waits; else the gap is a scheduling problem.",
    )
    s += text(
        8,
        16,
        "The checklist as a flow. Two tests, then the trace tells you which one you failed.",
        "h",
    )

    def node(x, y, w, h, lines, col, cls="ins", rx=4):
        s_ = rect(x, y, w, h, col, rx=rx)
        n = len(lines)
        for i, ln in enumerate(lines):
            s_ += text(x + w / 2, y + h / 2 + 3.5 + (i - (n - 1) / 2) * 13, ln, cls, "middle")
        return s_

    bw, gp = 160, 34
    xs = [8 + i * (bw + gp) for i in range(4)]
    y1, h1 = 40, 44
    s += node(
        xs[0],
        y1,
        bw,
        h1,
        ["1.  B = A+B bytes per K tile", "T = 16 x MFMAs per wave"],
        ("#ffffff", GREY_T),
    )
    s += node(xs[1], y1, bw, h1, ["2.  T x C / B  ≥  L ?", "fp16: 11 BM BN / (BM + BN)"], YELLOW)
    s += node(
        xs[2], y1, bw, h1, ["3.  (stages − 1) x T  ≥  L ?", "buffers = what LDS allows"], YELLOW
    )
    s += node(xs[3], y1, bw, h1, ["4.  still waits on vmcnt?", "which instruction waits?"], YELLOW)
    for i in range(3):
        s += arrow(xs[i] + bw, y1 + h1 / 2, xs[i + 1] - 2, y1 + h1 / 2)
        if i:
            s += text(xs[i] + bw + gp / 2, y1 + h1 / 2 - 6, "yes", "lbl", "middle")
    y2 = 124
    for i in (1, 2):
        s += text(xs[i] + bw / 2 + 10, y2 - 18, "no", "lbl")
        s += arrow(xs[i] + bw / 2, y1 + h1, xs[i] + bw / 2, y2 - 2)
    s += node(
        xs[1],
        y2,
        bw,
        74,
        [
            "bandwidth-bound",
            "spread the loads; then the",
            "ceiling is T x C / (B x L).",
            "levers: tile shape, dtype, L",
        ],
        RED,
    )
    s += node(
        xs[2],
        y2,
        bw,
        74,
        ["latency-bound", "deepen the pipeline, then", "back to test 2's verdict"],
        RED,
    )
    # node 4: no -> scheduling problem (row 2, narrower); yes -> read the trace (row 3)
    s += text(xs[3] + (bw - 30) / 2 + 10, y2 - 18, "no", "lbl")
    s += arrow(xs[3] + (bw - 30) / 2, y1 + h1, xs[3] + (bw - 30) / 2, y2 - 2)
    s += node(
        xs[3],
        y2,
        bw - 30,
        74,
        ["5. scheduling", "problem: LLIR", "scheduler, the", "v7 → v9 steps"],
        GREEN,
    )
    y3 = 224
    s += text(xs[3] + bw - 16, y2 + 30, "yes", "lbl", "end")
    s += path(f"M {xs[3] + bw - 10} {y1 + h1} V {y3 - 2}", "wire", ' marker-end="url(#arr)"')
    s += node(
        xs[3] - 60,
        y3,
        bw + 60,
        60,
        [
            "a stretched buffer_load",
            "→ the cap of test 2",
            "a long s_waitcnt before the",
            "ds_reads → the depth of test 3",
        ],
        RED,
    )
    s = s.replace(f'y="{y3 + 30 + 3.5 - 19.5}"', f'y="{y3 + 30 + 3.5 - 19.5}"')
    y = 310
    s += text(
        8,
        y,
        "C = 44 KB per CU (one workgroup per CU). L ≈ 1000 cycles at L2-friendly K; it grows with the L2 miss rate, so the",
        "note",
    )
    s += text(
        8,
        y + 16,
        "verdicts are optimistic at large K. An 8-bit element type halves B at the same T, so it doubles a tile's budget.",
        "note",
    )
    s += text(
        8,
        y + 40,
        "Only when both tests pass is the remaining gap something a schedule can close.",
        "lblb",
    )
    s += "</svg>\n"
    write("tcp_decision_flow.svg", s)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_budget()
    fig_saturation()
    fig_test()
    fig_tiles()
    fig_who_waits()
    fig_depth()
    fig_flow()
