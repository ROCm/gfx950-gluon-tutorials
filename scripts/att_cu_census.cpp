/******************************************************************************
 * MIT License
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 ******************************************************************************/

// Census of which compute units actually execute waves, per (XCC, shader engine).
//
// Why this exists: a shader array reports 9 CU slots but only 8 are enabled -- one is
// harvested for yield, and *which* index differs per die. rocprofv3's ATT takes a single
// `att_target_cu` and does NOT validate it, so pointing it at a harvested CU yields a
// silent empty trace (exit 0, a few KB of .att, no wave files). `att_pick_cu.py` uses this
// census to choose an index that exists.
//
// Method: every workgroup reads HW_ID and XCC_ID and reports where it ran. Slots that
// never appear across a saturating launch are the harvested ones.
//   gfx9 HW_ID: WAVE[3:0] SIMD[5:4] PIPE[7:6] CU[11:8] SH[12] SE[15:13]
//
// Output (stdout), one line per array, CU set as a bitmask:
//   xcc <n> se <n> cu_mask 0x<hex>

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_OK(expr)                                                                     \
  do {                                                                                   \
    hipError_t err_ = (expr);                                                            \
    if (err_ != hipSuccess) {                                                            \
      fprintf(stderr, "att_cu_census: %s failed: %s\n", #expr, hipGetErrorString(err_));  \
      return 1;                                                                          \
    }                                                                                    \
  } while (0)

__global__ void census(unsigned *out, int spin) {
  unsigned hwid = 0, xcc = 0;
  // Linger so the dispatcher fills every CU rather than recycling a few.
  for (int i = 0; i < spin; ++i)
    __builtin_amdgcn_s_sleep(1);
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID)" : "=s"(hwid));
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=s"(xcc));
  if (threadIdx.x == 0)
    out[blockIdx.x] = ((xcc & 0xF) << 24) | (hwid & 0xFFFF);
}

int main(int argc, char **argv) {
  int blocks = argc > 1 ? atoi(argv[1]) : 32768;
  if (blocks < 1)
    blocks = 32768;

  unsigned *dev = nullptr;
  HIP_OK(hipMalloc(&dev, blocks * sizeof(unsigned)));
  HIP_OK(hipMemset(dev, 0xFF, blocks * sizeof(unsigned)));
  hipLaunchKernelGGL(census, dim3(blocks), dim3(256), 0, 0, dev, 60);
  HIP_OK(hipDeviceSynchronize());

  std::vector<unsigned> host(blocks);
  HIP_OK(hipMemcpy(host.data(), dev, blocks * sizeof(unsigned), hipMemcpyDeviceToHost));
  HIP_OK(hipFree(dev));

  // [xcc][se] -> bitmask of CU ids observed
  unsigned mask[16][8] = {};
  bool seen[16][8] = {};
  for (unsigned v : host) {
    if (v == 0xFFFFFFFFu)
      continue; // workgroup never ran (should not happen)
    unsigned cu = (v >> 8) & 0xF, se = (v >> 13) & 0x7, xcc = (v >> 24) & 0xF;
    mask[xcc][se] |= 1u << cu;
    seen[xcc][se] = true;
  }
  for (unsigned x = 0; x < 16; ++x)
    for (unsigned s = 0; s < 8; ++s)
      if (seen[x][s])
        printf("xcc %u se %u cu_mask 0x%X\n", x, s, mask[x][s]);
  return 0;
}
