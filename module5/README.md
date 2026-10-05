# Module 5 — CUDA Memory

One computation run five times, with the operands in a different CUDA memory
space each pass. All five kernels call the same `stencil_dot()`, so the
arithmetic is identical and any timing difference comes from memory placement.

The computation is a 33-tap weighted sum over 4,194,304 samples:
`out[i] = sum(coeff[k] * in[i+k])`.

## Memory used

| Space | Variable | How |
|---|---|---|
| Host | `h_coeff`, `h_in`, `h_ref` | `malloc`, plus the CPU reference |
| Global | `d_coeff`, `d_in`, `d_out` | `cudaMalloc` + `cudaMemcpy` |
| Constant | `c_coeff[TAPS]` | `__constant__`, `cudaMemcpyToSymbol` |
| Shared | `s_tile[]` | `extern __shared__`, sized at launch |
| Register | `r_coeff[TAPS]`, `acc` | per-thread locals, no spill |
| Host, mapped | `hm_in`, `hm_out` | `cudaHostAlloc` — read over PCIe |

## Build and run

```bash
./build.sh
./run.sh 1048576 256          # <total_threads> <threads_per_block>
```

Both arguments are optional (default 1048576 and 256). The kernels use
grid-stride loops, so any thread and block count works. `make TAPS=9` changes
the filter width; it is compile-time because it sizes the coefficient arrays.

All 7 configurations run from the repository root:

```bash
../assignment_build_execution/run_assignments.sh
```

## Results

Tesla T4. Mean of 20 launches after an untimed warm-up. Speedup is relative to
global. `output/` holds the full command-line output (`results.txt` at 33 taps,
`results_taps9.txt` at 9) and screen captures of all 7 thread/block runs.

| Grid | Global | Constant | Register | Shared | Host (mapped) |
|---|---|---|---|---|---|
| 256 x 256 | 0.6695 ms | 1.51x | 1.55x | 1.01x | 0.24x |
| 16384 x 64 | 0.3561 ms | 1.53x | 1.40x | 1.07x | 0.12x |
| 4096 x 256 | 0.4227 ms | 1.59x | 1.39x | 1.03x | 0.19x |
| 1024 x 1024 | 0.3300 ms | 1.59x | 1.36x | 0.81x | 0.17x |

All five variants matched the CPU reference in every configuration, with an
identical max error (3.58e-07).

## What the numbers show

- **Constant and register won (~1.5x).** Both stop re-reading `coeff[k]` from
  global memory on every tap. A warp reads the same coefficient at the same
  instant, which the constant cache serves in one broadcast.

- **Shared memory did not help**, and got worse as blocks grew (1.07x at 64
  threads, 0.81x at 1024). The input reads are already contiguous and
  overlapping across a warp, so L1 likely served the reuse first — leaving two
  `__syncthreads()` barriers that cost more as block size rises.

- **At 9 taps the memory choice stopped mattering** — every device variant
  converged to ~1.05x, because global already reached 224 GB/s, near this
  card's ceiling. Placement cannot help a kernel that is saturating DRAM.

- **Mapped host memory is PCIe-bound.** 1.77 ms at 9 taps, 2.23 ms at 33 —
  nearly unchanged, while the device variants got 3x faster.

- **Occupancy mattered more than memory.** 65,536 threads took 0.94 ms;
  1,048,576 took 0.33 ms — about 3x, against 1.5x from the best memory choice.
