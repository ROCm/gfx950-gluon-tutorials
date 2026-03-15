# Plot script for triton layouts (latex version)

This script is used to draw Triton layouts in the context of matmul.
Here is the help info from the script.

```bash
>$ python3 plot_layout.py -h
usage: Draw triton layouts [-h] [--output OUTPUT] [--keep] [--force] PLOT_TYPE ...

options:
  -h, --help            show this help message and exit
  --output OUTPUT       output pdf file name (without suffix)
  --keep                If set, keep the generated .tex file
  --force               If set, overwrite the pdf file with the same name

subcommands:
  Choose to plot blocked, lds, or dot

  PLOT_TYPE        Choose one of the three plot modes
    blocked        plot blocked layout for global memory access
    dot            plot dot layout for MFMA/WMMA
    lds            plot LDS (shared memory) layout
```

**Important**: Use the `--gfx {942,950,1250}` argument in subcommands to automatically set architecture-specific
defaults (waveSize, banks, kWidth, kGroup) for your target GPU. When `--gfx` is not specified, the tool defaults
to waveSize=64 (suitable for most AMD GPUs).

## Installation
This script does not require torch or triton to be installed. The only package
it depends on is latex. On Ubuntu, do
```bash
sudo apt-get install texlive-latex-base texlive-latex-extra texlive-fonts-recommended texlive-fonts-extra

```

## Draw blocked layout (`python plot_layout.py blocked`)

The blocked subcommand supports three GPU architectures:
- **gfx942** (CDNA3/MI300): 64-thread waves
- **gfx950** (CDNA4/MI350): 64-thread waves
- **gfx1250** (MI450): 32-thread waves

### Quick Start with `--gfx`

```bash
# gfx942/gfx950 (64-thread waves)
python3 plot_layout.py blocked --gfx 942 --sizePerThread 1 8 --threadsPerWarp 16 4 --warpsPerCTA 1 2

# gfx1250 (MI450, 32-thread waves)
python3 plot_layout.py blocked --gfx 1250 --sizePerThread 1 8 --threadsPerWarp 8 4 --warpsPerCTA 1 2
```

### Manual Configuration

```bash
>$ python plot_layout.py blocked --help
usage: Draw triton layouts blocked [-h] [--gfx {942,950,1250}] [-r ROW] [-c COL] [-B] [-s s0 s1] [-t t0 t1] [-w w0 w1] [-o minor major] [-b b0 b1]

options:
  -h, --help                           show this help message and exit
  --gfx {942,950,1250}                 GPU architecture (auto-sets waveSize). 942=MI300, 950=MI350, 1250=MI450
  -r ROW, --rowName ROW                tensor dim0 name (default: M)
  -c COL, --colName COL                tensor dim1 name (default: K)
  -B, --matrixB                        shortcut to plot operand B with dimension name of (K, N) (default: False)
  -s s0 s1, --sizePerThread s0 s1      how many elements each thread holds in the 2D block per CTA (default: (1, 4))
  -t t0 t1, --threadsPerWarp t0 t1     how threads are partitioned into a 2D grid in a warp (default: (16, 4))
  -w w0 w1, --warpsPerCTA w0 w1        how warps tile a CTA (default: (1, 4))
  -o minor major, --order minor major  order from most minor to most major (default: (1, 0))
  -b b0 b1, --blockShape b0 b1         block size (dim0, dim1) of the tile. If not specified it presumably equals to the shape of CTA
```

### Examples
```bash
python3 plot_layout.py blocked --sizePerThread 1 8 --threadsPerWarp 8 8 --warpsPerCTA 4 1
python3 plot_layout.py blocked --blockShape 16 64 --sizePerThread 1 8 --threadsPerWarp 16 4 --warpsPerCTA 1 2
python3 plot_layout.py blocked --blockShape 32 64 --sizePerThread 8 1 --threadsPerWarp 4 16 --warpsPerCTA 1 2 --order 0 1
```

Blocked layouts are used during global load. It is used to describe the layout of the tensor
for pointers and results.
We can provide blocked layout parameters (
`--sizePerThread x y`, `--threadsPerWarp x y`, and `--warpsPerCTA x y`).
We can also provide the order of the tensor as `--order x y` to control which dim
is the fastest changing dimension.

Note that the parameters above forms a Cooperative Thread Array (CTA) and the block may contain multiple of it.
Please specifiy block shape (`--blockShape b0 b1`) to explicitly set the block size, otherwise it will assumes block size equals to CTA size.

Notes
- The number of threads per warp depends on waveSize (64 for gfx942/gfx950, 32 for gfx1250).
  Ensure `threadsPerWarp[0] * threadsPerWarp[1] == waveSize`.
- The script does not support the case when threads are loading elements that are
  out of the boundary of the tensor dimensions. This means
  - For dim0: sizePerThread[0] * threadsPerWarp[0] * warpsPerCTA[0] <= dim0
  - For dim1: sizePerThread[1] * threadsPerWarp[1] * warpsPerCTA[1] <= dim1


## Draw mfma/wmma operand and result layouts (`python plot_layout.py dot`)

The dot subcommand supports three GPU architectures:
- **gfx942** (CDNA3/MI300): 64-thread waves, MFMA instructions
- **gfx950** (CDNA4/MI350): 64-thread waves, MFMA instructions with f4/f6 support
- **gfx1250** (MI450): 32-thread waves, WMMA instructions

### Quick Start with `--gfx`

The easiest way to use this tool is with the `--gfx` argument, which automatically sets
the correct `waveSize`, `kWidth`, and `kGroup` based on the target architecture:

```bash
## gfx942 (MI300/CDNA3)
python3 plot_layout.py dot --gfx 942 --dotShape 64 64 64 --warpsPerCTA 1 2 --dtypeA fp16
python3 plot_layout.py dot --gfx 942 --dotShape 64 64 64 --warpsPerCTA 1 2 --dtypeA fp8

## gfx950 (MI350/CDNA4) - supports f4/f6 types
python3 plot_layout.py dot --gfx 950 --dotShape 64 64 64 --warpsPerCTA 1 2 --dtypeA fp16
python3 plot_layout.py dot --gfx 950 --dotShape 128 128 128 --warpsPerCTA 1 2 --dtypeA f4

## gfx1250 (MI450) - uses WMMA instructions
python3 plot_layout.py dot --gfx 1250 --dotShape 64 64 64 --warpsPerCTA 1 2 --dtypeA fp16
python3 plot_layout.py dot --gfx 1250 --dotShape 64 64 64 --warpsPerCTA 1 2 --dtypeA fp8
```

### Architecture Default Configurations

| GFX | waveSize | Supported dtypes | Default kWidth by dtype |
|-----|----------|------------------|------------------------|
| 942 | 64 | fp16, bf16, fp8, bf8, i8 | fp16/bf16: 4, fp8/bf8/i8: 8 |
| 950 | 64 | fp16, bf16, fp8, bf8, fp6, bf6, f4, i8 | fp16/bf16/fp8/bf8: 8, i8: 16, f4/f6: 32 |
| 1250 | 32 | fp16, bf16, fp8, bf8, fp6, bf6, f4 | fp16/bf16: 8 (kGroup=2), fp8/bf8: 8 (kGroup=4), f4/f6: 32 (kGroup=2) |

### Manual Configuration

You can also manually specify all parameters:

```bash
>$ python plot_layout.py dot --help
usage: Draw triton layouts dot [-h] [--gfx {942,950,1250}] [--dotShape M N K] [--warpsPerCTA w0 w1] [--nonKDim {16,32}]
                               [--kWidth {4,8,16,32,64}] [--kGroup {1,2,4,8}]
                               [--dtypeA {fp16,bf16,fp8,bf8,fp6,bf6,f4,i8}] [--dtypeB {fp16,bf16,fp8,bf8,fp6,bf6,f4,i8}]
                               [--mfmaTrans] [--scale]

options:
  -h, --help                                  show this help message and exit
  --gfx {942,950,1250}                        GPU architecture (auto-sets waveSize, kWidth, kGroup)
  --dotShape M N K                            Dot op shape in the form of M, N, K (default: (32, 128, 64))
  --warpsPerCTA w0 w1                         how warps tile the dot result matrix (default: (1, 4))
  --tilesPerWarp y0 y1                        how many contiguous tiles per warp (default: (1, 1))
  --nonKDim {16,32}                           mfma instruction dimension of M/N (default: 16)
  --kWidth {4,8,16,32,64}                     number of contiguous elements each thread owns (auto-set if --gfx provided)
  --kGroup {1,2,4,8}                          total number of elements / kWidth per instruction (auto-set if --gfx provided)
  --dtypeA {fp16,bf16,fp8,bf8,fp6,bf6,f4,i8}  element type of operand A (default: fp16)
  --dtypeB {fp16,bf16,fp8,bf8,fp6,bf6,f4,i8}  element type of operand B (default: fp16)
  --mfmaTrans                                 If set, then use mfma.trans layout (default: False)
  --scale                                     If set, plot the scale tensor for mfma_f8f6f4 instructions (default: False)
```

### Examples with Manual Configuration

```bash
## i8 inputs
python3 plot_layout.py dot --dotShape 128 128 128 --warpsPerCTA 2 4 --kWidth 8 --dtypeA i8 --dtypeB i8
python3 plot_layout.py dot --dotShape 128 128 128 --warpsPerCTA 2 4 --kWidth 16 --dtypeA i8 --dtypeB i8
## fp16/bf16 inputs
python3 plot_layout.py dot --dotShape 128 128 128 --warpsPerCTA 2 4 --kWidth 4 --dtypeA fp16 --dtypeB fp16
python3 plot_layout.py dot --dotShape 128 128 128 --warpsPerCTA 2 4 --kWidth 8 --dtypeA fp16 --dtypeB fp16
## fp8/bf8 inputs
python3 plot_layout.py dot --dotShape 128 128 128 --warpsPerCTA 2 4 --kWidth 8 --dtypeA fp8 --dtypeB bf8
python3 plot_layout.py dot --dotShape 128 128 128 --warpsPerCTA 2 4 --kWidth 16 --dtypeA fp8 --dtypeB bf8
python3 plot_layout.py dot --dotShape 128 128 128 --warpsPerCTA 2 4 --kWidth 16 --kGroup 2 --dtypeA fp8 --dtypeB bf8
## f4 and fp6/bf6 inputs (gfx950 only)
python3 plot_layout.py dot --dotShape 128 128 128 --warpsPerCTA 2 4 --kWidth 32 --kGroup 1 --dtypeA f4 --dtypeB bf6
## fp8/bf8 and fp6/bf6/f4 inputs
python3 plot_layout.py dot --dotShape 128 128 128 --warpsPerCTA 2 4 --kWidth 16 --kGroup 2 --dtypeA fp6 --dtypeB bf8
## mixed precision with scaling
python3 plot_layout.py dot --dotShape 128 128 128 --warpsPerCTA 2 4 --kWidth 16 --kGroup 2 --dtypeA fp6 --dtypeB bf8 --scale
```

One can add `--nonKDim [16,32]` and `--mfmaTrans` to all of the above examples.

This mode draws two graphs:
1. The layout of the dot operation, i.e. tile C = tile A x tile B
2. The layout of a single mfma/wmma block, operands and results of one or more
   instructions that share the same accumulating VGPRs.

### Supported Instructions

**MFMA Instructions (gfx942/gfx950, waveSize=64)**

| Data Type | nonKDim | kWidth | kGroup | Instruction |
|-----------|---------|--------|--------|-------------|
| fp16/bf16 | 16 | 4 | 1 | `mfma_f32_16x16x16_fp16` |
| fp16/bf16 | 16 | 8 | 1 | `mfma_f32_16x16x32_fp16` |
| fp8/bf8 | 16 | 8 | 1 | `mfma_f32_16x16x32_fp8_fp8` |
| i8 | 16 | 8 | 1 | `mfma_i32_16x16x32_i8` |
| i8 | 16 | 16 | 1 | `mfma_i32_16x16x64_i8` |
| f4/f6 (gfx950) | 16 | 32 | 1 | `mfma_f32_16x16x128_f8f6f4` |

**WMMA Instructions (gfx1250, waveSize=32)**

| Data Type | nonKDim | kWidth | kGroup | Instruction |
|-----------|---------|--------|--------|-------------|
| fp16/bf16 | 16 | 8 | 2 | `wmma_f32_16x16x32_fp16` |
| fp16/bf16 | 16 | 16 | 1 | `wmma_f32_16x16x32_fp16` |
| fp8/bf8 | 16 | 8 | 4 | `wmma_f32_16x16x64_fp8_fp8` |
| fp8/bf8 | 16 | 16 | 2 | `wmma_f32_16x16x64_fp8_fp8` |
| fp8/bf8 | 16 | 32 | 2 | `wmma_f32_16x16x128_fp8_fp8` |
| f4/f6 | 16 | 32 | 2 | `wmma_f32_16x16x128_f8f6f4` |

### Knobs
- `--gfx [942,950,1250]`: GPU architecture. Auto-sets waveSize, kWidth, and kGroup.
- `--kWidth [4,8,16,32,64]`: the number of elements that will be loaded into one thread at once
- `--kGroup [1,2,4,8]`: total number of elements / kWidth for one mfma/wmma instruction.
- `--nonKDim [16,32]`: mfma/wmma instruction size. The default is 16. Note: gfx1250 only supports 16.
- `--mfmaTrans`: if set, the transposed mfma layout will be plotted.
- `--dtypeA` and `--dtypeB`: element types of operand A and B. The default value is fp16.
- `--scale`: plot scale tensors for A and B. This is only supported with f4/f6 and f8 with `kGroup>=2`.
  If `--scale` is set but not supported, it's ignored.

### Notes
- The layout shows the mapping from the threads/wave to the elements in the
  original tensor. It does not matter if LDS is used.
- The script does not allow settings for k dim of the mfma instruction.
  This can be controlled by the `--kWidth` and `--kGroup`.
- For gfx1250 (MI450), only `nonKDim=16` is supported.

## Draw LDS access (`python plot_layout.py lds`)

The lds subcommand supports three GPU architectures with different LDS configurations:
- **gfx942** (CDNA3/MI300): 32 banks, 64-thread waves
- **gfx950** (CDNA4/MI350): 64 banks, 64-thread waves
- **gfx1250** (MI450): 64 banks, 32-thread waves

### Quick Start with `--gfx`

The easiest way to use this tool is with the `--gfx` argument, which automatically sets
the correct `banks` and `waveSize` based on the target architecture:

```bash
# gfx942 (MI300): 32 banks, waveSize=64
python3 plot_layout.py lds --gfx 942 --tensorShape 16 64 --kWidth 8 --dtype fp16 --layout swizzle --access read

# gfx950 (MI350): 64 banks, waveSize=64
python3 plot_layout.py lds --gfx 950 --tensorShape 16 64 --kWidth 8 --dtype fp16 --layout swizzle --access read

# gfx1250 (MI450): 64 banks, waveSize=32
python3 plot_layout.py lds --gfx 1250 --tensorShape 16 64 --kWidth 8 --dtype fp16 --layout swizzle --access read
```

### Architecture LDS Configurations

| GFX | Banks | waveSize | GPU |
|-----|-------|----------|-----|
| 942 | 32 | 64 | MI300 (CDNA3) |
| 950 | 64 | 64 | MI350 (CDNA4) |
| 1250 | 64 | 32 | MI450 |

### Manual Configuration

```bash
>$ python plot_layout.py lds --help
usage: Draw triton layouts lds [-h] [--gfx {942,950,1250}] [--tensorShape TENSORSHAPE TENSORSHAPE] [--kWidth {4,8,16,32}]
                               [--dtype {fp16,bf16,fp8,bf8,fp6,bf6,f4,i8}] [--nonKDim {16,32}]
                               [--banks {32,64}] [--layout {swizzle,padding,none}] [--access {read,write,none}] [--mnContig] [--mfma-trans-load]
                               [--swizzleVec {4,8,16,32}] [--padInterval PADINTERVAL] [--padAmount PADAMOUNT] [--sharedLayout SHAREDLAYOUT]

options:
  -h, --help                                 show this help message and exit
  --gfx {942,950,1250}                       GPU architecture (auto-sets banks, waveSize). 942=MI300, 950=MI350, 1250=MI450
  --tensorShape TENSORSHAPE TENSORSHAPE      2D block shape in the form of (dim0, dim1) (default: (128, 64))
  --kWidth {4,8,16,32}                       number of contiguous elements per thread (default: 4)
  --dtype {fp16,bf16,fp8,bf8,fp6,bf6,f4,i8}  element type of tensor to be stored in LDS (default: fp16)
  --nonKDim {16,32}                          mfma instruction dim (default: 16)
  --banks {32,64}                            choose the number of banks in LDS (default: 64)
  --layout {swizzle,padding,none}            choose the LDS data layout (default: none)
  --access {read,write,none}                 choose LDS access mode (default: none)
  --mnContig                                 If set, the tensor is K x N and n-contig (default: False)
  --mfma-trans-load                          If set, use MFMA transpose load instructions (default: False)
  --swizzleVec {4,8,16,32}                   number of contiguous elements in a vector to swizzle (default: 4)
  --padInterval PADINTERVAL                  Add padding for every padInterval bytes (default: 1)
  --padAmount PADAMOUNT                      Pad padAmount bytes for every padInterval bytes (default: 0)
  --sharedLayout SHAREDLAYOUT                Triton shared layout string: "[[padInterval, padAmount], ...], [[r,c], ...]" where pads are in elements (default: )
```

### Examples
```bash
# Using --gfx (recommended)
python3 plot_layout.py lds --gfx 942 --layout swizzle --access read --tensorShape 128 128 --kWidth 8 --dtype fp16
python3 plot_layout.py lds --gfx 950 --layout swizzle --access read --tensorShape 128 128 --kWidth 16 --dtype fp8
python3 plot_layout.py lds --gfx 1250 --layout swizzle --access read --tensorShape 128 128 --kWidth 8 --dtype fp16

# Manual configuration
python3 plot_layout.py lds --layout none --access none --tensorShape 128 128 --kWidth 8
python3 plot_layout.py lds --layout none --access none --tensorShape 128 128 --kWidth 32 --dtype f4
python3 plot_layout.py lds --layout swizzle --access write --tensorShape 128 128 --kWidth 16 --dtype f4 --banks 32
python3 plot_layout.py lds --layout none --access read --tensorShape 128 32 --kWidth 4 --dtype fp16 --banks 64 --mnContig
python3 plot_layout.py lds --layout swizzle --access read --tensorShape 128 32 --kWidth 16 --dtype fp8 --banks 64 --mnContig --mfma-trans-load
python3 plot_layout.py lds --layout padding --access none --tensorShape 128 32 --kWidth 8 --dtype fp16 --banks 32 --padInterval 128 --padAmount 16
python3 plot_layout.py lds --tensorShape 256 64 --kWidth 8 --dtype fp16 --banks 32 --sharedLayout "[[512,16], [1024,32]], [[0,1], [0,2], [0,4], [0,8], [0,16], [0,32], [16,0], [32,0], [64,0], [1,0], [2,0], [4,0], [8,0], [128,0]]"
```

Knobs
- `--gfx [942,950,1250]`: GPU architecture. Auto-sets `banks` and `waveSize`.
- `--kWidth`: the vector size (in unit of elements) when accessing LDS
- `--banks`: the number of banks in LDS. (64 for gfx950/gfx1250, 32 for gfx942; default: 64)
- `--dtype`: element data type
- Three options for `--layout`:
  - `none`: no swizzling, no padding
  - `swizzle`: apply the swizzling pattern, which is derived from tensor shape and kWidth.
  - `padding`: pad `padAmount` bytes for every `padInterval` bytes of data
    - `padAmount`: default is 0
    - `padInterval`: default is 1
  - `sharedLayout`: if provided, overrides `--layout`, `--padInterval`, and `--padAmount`
    using a Triton shared layout string with multi-level padding (pad interval/amount are element counts and must be multiples of swizzleVec) and row swizzle basis
- Three options for `--access`:
  - `none`: do not plot access pattern
  - `read`: plot accessed elements at the first cycle of ds_read
  - `write`: plot accessed elements during ds_write. For global load access, we assume
    a fully coalesced dwordx4 access pattern along the K dim.
- `--mnContig`: If set, the tile is stored in mn-contig layout. In this layout, elements along
  the M/N dim are contiguous in both global memory and LDS.
- `--mfma-trans-load`: This flag only works when `--mnContig` is set. When set, `ds_read_b64_tr_bx`
  instructions are used to read from LDS. Note that current triton LDS layout mechanism will
  lead to bank conflicts.

# Linear Layout Visualizer (matplotlib version)

There is a new Python script that can visualize **Triton linear layouts**, showing how registers, lanes, and warps map to tensor coordinates.
It generates a color-coded **PDF** where each rectangle represents a vector region, labeled with all lanes that map to it.


## Usage

```bash
python3 plot_ll.py \
  --regBase "[[0,1],[0,2],[0,4]]" \
  --laneBase "[[1,0],[2,0],[4,0]]" \
  --warpBase "[[0,8]]" \
  --warpId 0 \
  --o layout_demo
```
Output
```bash
- register:
  register=1 --> (0, 1)
  register=2 --> (0, 2)
  register=4 --> (0, 4)
- lane:
  lane=1 --> (1, 0)
  lane=2 --> (2, 0)
  lane=4 --> (4, 0)
- warp:
  warp=1 --> (0, 8)

Tensor shape: [0, 16]
Vector dimension: 1, vector size: 8
✅ Layout visualization saved to: layout_demo.pdf
```

## Overview

The tool takes in three optional layout base matrices:

- Register bases (--regBase)
- Lane bases (--laneBase)
- Warp bases (--warpBase)

It then computes, for every lane and register combination, the corresponding 2D tensor coordinates 
(e.g., (dim0, dim1)) using XOR-based mapping.
All lanes that map to the same tensor coordinate are grouped together and displayed as one rectangle in the final plot, labeled with their lane IDs.

The output is a PDF visualization showing the tensor layout from hardware indices.

## Dependencies

- Python ≥ 3.8
- NumPy
- Matplotlib

Install via pip:
```bash
pip install numpy matplotlib
```
