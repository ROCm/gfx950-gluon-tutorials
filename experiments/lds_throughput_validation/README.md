# Experiment of `ds_read` throughput and bank conflicts

Command to run the benchmark:
```python
python kernel.py
```
The default settings and where you can change it:
- `num_warps=4`. You can change it to `num_warps=1` at [this line](kernel.py#L104).
- Padding is `1024:+16`. You can change it at [this line](kernel.py#L21).

Command to collect thread trace (rocm7.0 docker)
```
ROCPROF_ATT_LIBRARY_PATH=/root/rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux/opt/rocm/lib/ rocprofv3 --att -i att.json -d att_output -- python kernel.py
```

## 1 wave

Experiment design
- Simplified version of FA kernel, i.e. only keep th e1st dot.
- Load A directly into dotOperandLayout outside of the loop.
- Allocate LDS for B. Use B LDS buffer size to control occupancy and make sure
  there is only 1 workgroup per CU.
- B tile size: `BLOCK_K` x `BLOCK_N` = 128 x (`num_warps` x 256)
  - This is to make sure for difference number of warps, there are always
    32 x `ds_read_b128` inside the loop.

Padding parameters and bank conflicts

| padding info      | bank conflicts | cycles/ds_read | output                |
|-------------------|----------------|----------------|-----------------------|
| 1024:+16          | 2              | 8              | `nW1_1024-16`         |
| 1024:+32          | 2              | 8              | `nW1_1024-32`         |
| 1024:+64          | 2              | 8              | `nW1_1024-64`         |
| 1024:+128         | 4              | 16             | `nW1_1024-128`        |
| 1024:+256         | 8              | 32             | `nW1_1024-256`        |
| 1024:+4           | unaligned      | 64             | `nW1_1024-4`          |
| 1024:+16,2048:+32 | 0              | 8              | `nW1_1024-16_2048-32` |

Thread traces screenshots can be found [here](./attviewer_traces/lds_throughput_1-wave.png).

## 4 wave

Compared to 1-wave experiment, the 4-wave one
- uses 128x512 as the B tile size. Therefore, there are 16 `ds_read_b128`
  instructions to load dat for B.
  This is due to the limit of the offset field of `ds_read`.
  If we use 128x1024 tile size, we cannot use a single vgpr as the addr,
  since the offset field is not large enough to address data in the whole tensor.
  
Padding parameters and bank conflicts

| padding info      | bank conflicts | cycles/ds_read | output                |
|-------------------|----------------|----------------|-----------------------|
| 1024:+16          | 2              | 32             | `nW4_1024-16`         |
| 1024:+32          | 2              | 32             | `nW4_1024-32`         |
| 1024:+64          | 2              | 32             | `nW4_1024-64`         |
| 1024:+128         | 4              | 64             | `nW4_1024-128`        |
| 1024:+256         | 8              | 128            | `nW4_1024-256`        |
| 1024:+4           | unaligned      | 256            | `nW4_1024-4`          |
| 1024:+16,2048:+32 | 0              | 16             | `nW4_1024-16_2048-32` |

Thread traces screenshots can be found in [here](./attviewer_traces/lds_throughput_4-waves.png)
