# Collecting private / non-public SQ perf counters on gfx950 (MI35x)

Recipe + findings from the FAv3 mfma-bubble investigation. The dominant stall was
found to be **`SQ_VALU_SRC_C_CONFLICT`** (VALU src-C VGPR-port contention on the
MFMA accumulator), via the PMC path below.

---

## 1. Why they need special handling

Counters like `SQ_VALU_SRC_C_CONFLICT`, `SQ_STALL_PIT`, `SQ_PT_POWER_STALL`,
`SQ_VALU_DEP_STALL` are **not shipped** in ROCm's counter definitions
(`/opt/rocm/share/rocprofiler-sdk/counter_defs.yaml`,
`/opt/rocm/libexec/rocprofiler/counters/basic_counters.xml`, etc.), so
rocprofv3/aqlprofile reject them by name:

```
W ... tool.cpp:1307] Agent NNNN counter not found: SQ_VALU_SRC_C_CONFLICT
```

## 2. Define them yourself via `-E extra_counters.yaml`

Map each name to its **SQ block + raw event id** (same schema as counter_defs.yaml):

```yaml
rocprofiler-sdk:
  counters-schema-version: 1
  counters:
  - name: SQ_VALU_SRC_C_CONFLICT
    description: "VALU stalled by arbitration due to src-C port conflict"
    properties: []
    definitions:
    - architectures: [gfx950]
      block: SQ
      event: 158
  # ... repeat for the others
```

Pass it with `rocprofv3 ... -E extra_counters.yaml`. (This is
`kernels/attention/att_stall_counters.yaml`.)

## 3. Event ids = the gfx950 `SQ_PERF_SEL_*` enum values

Get the enum from AMD's SQ perf-sel table (Giovanni's list). Verified that the
enum value **equals** the `event:` number ROCm's counter_defs uses for gfx950,
by cross-checking known public counters:

| public counter | SQ_PERF_SEL enum | counter_defs gfx950 event | match |
|---|---:|---:|:--:|
| VALU_MFMA_BUSY_CYCLES | 93 | 93 | ✅ |
| ACTIVE_INST_VALU | 120 | 120 | ✅ |
| WAIT_INST_LDS | 112 | 112 | ✅ |
| INSTS_VALU_CVT | 41 | 41 | ✅ |

Stall / bubble-diagnosis counters:

| counter | event | meaning |
|---|---:|---|
| `SQ_VALU_DEP_STALL` | 148 | VALU stalled by dependencies |
| `SQ_VALU_SRC_C_CONFLICT` | 158 | VALU stalled by src-C VGPR-port conflict |
| `SQ_PT_POWER_STALL` | 164 | power-throttle stall (unwindowed) |
| `SQ_STALL_PIT` | 226 | PIT stall |
| `SQ_VALU_MFMA_BUSY_CYCLES` | 93 | MFMA ALU busy |
| `SQ_VALU_MFMA_COEXEC_CYCLES` | 94 | MFMA busy AND a normal VALU co-issued |
| `SQ_WAIT_OTHER` | 105 | dependency-stall wait cycles |

## 4. PMC aggregate — **works** (this gave the answer)

```bash
LLVM_PASS_PLUGIN_PATH=.../libLlirSched.so LLVM_PASS_PLUGIN_KEEP_TARGET_MACHINE=1 \
DISABLE_LLVM_OPT=disable-vector-combine FA_MODULE=fav3 \
rocprofv3 --pmc SQ_VALU_SRC_C_CONFLICT SQ_STALL_PIT SQ_VALU_DEP_STALL SQ_PT_POWER_STALL \
          SQ_VALU_MFMA_BUSY_CYCLES SQ_BUSY_CYCLES \
  -E att_stall_counters.yaml --kernel-include-regex gluon_attn_fwd -f csv -d OUT -- \
  python bench.py --seqlen 16320 --layout bshd --hq 64 --d 128 --batch 1 --dtype fp16
```

Parse `OUT/*_counter_collection.csv` (rows: Dispatch_Id, Counter_Name, Counter_Value).
Result (fav3 + llirSched, steady-state dispatch):

```
SQ_VALU_SRC_C_CONFLICT   112,875,052   <- DOMINANT (~17x next)
SQ_STALL_PIT               6,669,745
SQ_VALU_DEP_STALL                  0
SQ_PT_POWER_STALL                  0
```

Caveat: `SQ_BUSY_CYCLES` and `SQ_VALU_MFMA_BUSY_CYCLES` are on different scales
(MFMA_BUSY is ~24x); compare the stall counters to **each other**, not as a % of a
busy counter.

## 5. ATT per-window (visualize counter over the instruction timeline)

Command (SE0 + SIMD0, ~36-cyc sampling, max 4 counters, `:0x1` per-counter SIMD mask):

```bash
rocprofv3 --att -i att_attn_simd0.json -E att_stall_counters.yaml \
  --att-perfcounter-ctrl 1 --att-perfcounters "SQ_VALU_SRC_C_CONFLICT:0x1" \
  -d OUT -- python bench.py ...
```
(`att_attn_simd0.json` sets `att_simd_select 0x1`, `att_shader_engine_mask 0x1`,
`att_target_cu 0`, `kernel_iteration_range [8]`.)

**Status on this stack (ROCm 7.2.4): the counter resolves, but NO per-window data
is produced.** Diagnosed by decoding the `.att` directly with the decoder's public
API (`rocprof_trace_decoder_parse_data`; headers in `ROCm/rocprof-trace-decoder`;
`perfevent` record type 2 = `{int64 time; uint16 events0..3; uint8 CU,bank}`): the
decoder parses the whole trace (GFXIP, 128 WAVE, OCCUPANCY, REALTIME) but returns
**0 PERFEVENT records** for every trace and counter (windowed or unwindowed), with
both the stock and the latest `amd-staging` aqlprofile. So the SQTT perfcounter
tokens are never emitted into the `.att` here.

Dead ends checked:
- `rocprofv3 --att` ui_output web viewer never carries counters (decoder doesn't
  write them to ui_output; `rocpd_pmc_event` table is empty in ATT mode).
- `rocprofv2 --plugin att` (the classic "edit -> counter options" viewer):
  **fatal error "Unsupported hardware. Use rocprofv3 ... for mi35x"** — dead on gfx950.
- Building the latest **aqlprofile** (collector) from AMD-ROCm-Internal did not add
  the tokens.
- Latest **public** `rocprof-trace-decoder` (0.1.6) == the installed one; newest
  `therock-7.12` ships no downloadable `.so`.

**Conclusion:** the per-window ATT overlay is gated on a newer **decoder
(internal therock-7.12) or rocprofiler-sdk**, not on aqlprofile or on knowing the
counter. For the aggregate value of any private SQ counter, the `-E` + `--pmc`
path above is reliable.
