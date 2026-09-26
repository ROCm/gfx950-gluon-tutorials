"""Scalar bodies for fmha_v4's map_elementwise rescale (RESCALE_MODE=4).

Generated: map_elementwise hands a pack of each thread's elements to a scalar
function as separate positional arguments, so a per-thread branch covering all
of a thread's accumulator elements needs one parameter per element. With
BLOCK_M=256, BLOCK_DMODEL=128 and 8 warps each thread holds 64 accumulator
elements (a 32x128 tile per wave over 64 lanes) and one row of l_i.
"""

from triton.experimental import gluon


@gluon.jit
def rescale_acc_pack64(
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    x8,
    x9,
    x10,
    x11,
    x12,
    x13,
    x14,
    x15,
    x16,
    x17,
    x18,
    x19,
    x20,
    x21,
    x22,
    x23,
    x24,
    x25,
    x26,
    x27,
    x28,
    x29,
    x30,
    x31,
    x32,
    x33,
    x34,
    x35,
    x36,
    x37,
    x38,
    x39,
    x40,
    x41,
    x42,
    x43,
    x44,
    x45,
    x46,
    x47,
    x48,
    x49,
    x50,
    x51,
    x52,
    x53,
    x54,
    x55,
    x56,
    x57,
    x58,
    x59,
    x60,
    x61,
    x62,
    x63,
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    a7,
    a8,
    a9,
    a10,
    a11,
    a12,
    a13,
    a14,
    a15,
    a16,
    a17,
    a18,
    a19,
    a20,
    a21,
    a22,
    a23,
    a24,
    a25,
    a26,
    a27,
    a28,
    a29,
    a30,
    a31,
    a32,
    a33,
    a34,
    a35,
    a36,
    a37,
    a38,
    a39,
    a40,
    a41,
    a42,
    a43,
    a44,
    a45,
    a46,
    a47,
    a48,
    a49,
    a50,
    a51,
    a52,
    a53,
    a54,
    a55,
    a56,
    a57,
    a58,
    a59,
    a60,
    a61,
    a62,
    a63,
):
    # One branch per thread: skipped by the whole wave (s_cbranch_execz) when
    # no lane holds a row with alpha != 1.
    if (
        (a0 != 1.0)
        | (a1 != 1.0)
        | (a2 != 1.0)
        | (a3 != 1.0)
        | (a4 != 1.0)
        | (a5 != 1.0)
        | (a6 != 1.0)
        | (a7 != 1.0)
        | (a8 != 1.0)
        | (a9 != 1.0)
        | (a10 != 1.0)
        | (a11 != 1.0)
        | (a12 != 1.0)
        | (a13 != 1.0)
        | (a14 != 1.0)
        | (a15 != 1.0)
        | (a16 != 1.0)
        | (a17 != 1.0)
        | (a18 != 1.0)
        | (a19 != 1.0)
        | (a20 != 1.0)
        | (a21 != 1.0)
        | (a22 != 1.0)
        | (a23 != 1.0)
        | (a24 != 1.0)
        | (a25 != 1.0)
        | (a26 != 1.0)
        | (a27 != 1.0)
        | (a28 != 1.0)
        | (a29 != 1.0)
        | (a30 != 1.0)
        | (a31 != 1.0)
        | (a32 != 1.0)
        | (a33 != 1.0)
        | (a34 != 1.0)
        | (a35 != 1.0)
        | (a36 != 1.0)
        | (a37 != 1.0)
        | (a38 != 1.0)
        | (a39 != 1.0)
        | (a40 != 1.0)
        | (a41 != 1.0)
        | (a42 != 1.0)
        | (a43 != 1.0)
        | (a44 != 1.0)
        | (a45 != 1.0)
        | (a46 != 1.0)
        | (a47 != 1.0)
        | (a48 != 1.0)
        | (a49 != 1.0)
        | (a50 != 1.0)
        | (a51 != 1.0)
        | (a52 != 1.0)
        | (a53 != 1.0)
        | (a54 != 1.0)
        | (a55 != 1.0)
        | (a56 != 1.0)
        | (a57 != 1.0)
        | (a58 != 1.0)
        | (a59 != 1.0)
        | (a60 != 1.0)
        | (a61 != 1.0)
        | (a62 != 1.0)
        | (a63 != 1.0)
    ):
        x0 = x0 * a0
        x1 = x1 * a1
        x2 = x2 * a2
        x3 = x3 * a3
        x4 = x4 * a4
        x5 = x5 * a5
        x6 = x6 * a6
        x7 = x7 * a7
        x8 = x8 * a8
        x9 = x9 * a9
        x10 = x10 * a10
        x11 = x11 * a11
        x12 = x12 * a12
        x13 = x13 * a13
        x14 = x14 * a14
        x15 = x15 * a15
        x16 = x16 * a16
        x17 = x17 * a17
        x18 = x18 * a18
        x19 = x19 * a19
        x20 = x20 * a20
        x21 = x21 * a21
        x22 = x22 * a22
        x23 = x23 * a23
        x24 = x24 * a24
        x25 = x25 * a25
        x26 = x26 * a26
        x27 = x27 * a27
        x28 = x28 * a28
        x29 = x29 * a29
        x30 = x30 * a30
        x31 = x31 * a31
        x32 = x32 * a32
        x33 = x33 * a33
        x34 = x34 * a34
        x35 = x35 * a35
        x36 = x36 * a36
        x37 = x37 * a37
        x38 = x38 * a38
        x39 = x39 * a39
        x40 = x40 * a40
        x41 = x41 * a41
        x42 = x42 * a42
        x43 = x43 * a43
        x44 = x44 * a44
        x45 = x45 * a45
        x46 = x46 * a46
        x47 = x47 * a47
        x48 = x48 * a48
        x49 = x49 * a49
        x50 = x50 * a50
        x51 = x51 * a51
        x52 = x52 * a52
        x53 = x53 * a53
        x54 = x54 * a54
        x55 = x55 * a55
        x56 = x56 * a56
        x57 = x57 * a57
        x58 = x58 * a58
        x59 = x59 * a59
        x60 = x60 * a60
        x61 = x61 * a61
        x62 = x62 * a62
        x63 = x63 * a63
    return (
        x0,
        x1,
        x2,
        x3,
        x4,
        x5,
        x6,
        x7,
        x8,
        x9,
        x10,
        x11,
        x12,
        x13,
        x14,
        x15,
        x16,
        x17,
        x18,
        x19,
        x20,
        x21,
        x22,
        x23,
        x24,
        x25,
        x26,
        x27,
        x28,
        x29,
        x30,
        x31,
        x32,
        x33,
        x34,
        x35,
        x36,
        x37,
        x38,
        x39,
        x40,
        x41,
        x42,
        x43,
        x44,
        x45,
        x46,
        x47,
        x48,
        x49,
        x50,
        x51,
        x52,
        x53,
        x54,
        x55,
        x56,
        x57,
        x58,
        x59,
        x60,
        x61,
        x62,
        x63,
    )


@gluon.jit
def rescale_row(l_i, alpha):
    if alpha != 1.0:
        l_i = l_i * alpha
    return l_i
