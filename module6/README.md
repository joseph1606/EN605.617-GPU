# Module 6 — CUDA Streams and Events

Encrypts a random 64–256 MB "file" on the GPU. The file is split into 4 MB
chunks, and each chunk goes through the same four steps on a stream:

1. copy the chunk to the GPU
2. `encryptKernel`: XOR each 4-byte word with a keystream made from the
   secret key and the word's position in the file
3. `checksumKernel`: shared-memory sum of the chunk, as a fingerprint
4. copy the encrypted chunk and its checksum back to the host

Chunks are dealt round-robin across 1, 2, 4 and 8 streams. The kernels and
data are the same in every row, so any timing difference comes from the
streams alone.

## Streams and events used

| What | How |
|---|---|
| Streams | 8 non-blocking streams, each with its own device buffer; chunk `c` runs on stream `c % numStreams` |
| Timing | `start`/`stop` events on stream 0, `cudaEventElapsedTime` |
| Start sync | other streams `cudaStreamWaitEvent` on `start`, so no work begins before the timer |
| Finish sync | each stream records a `done` event, stream 0 waits on all of them before `stop` |
| Pinned memory | `cudaMallocHost`, required for `cudaMemcpyAsync` to overlap |

## Verification

Every run checks that:

- the encrypted data differs from the original
- the CPU checksum of every chunk matches the GPU's
- encrypting the encrypted file again restores the original byte for byte

## Build and run

```bash
./build.sh
./run.sh 256 256          # <file_MB> <threads_per_block>
```

File size must be a multiple of 4 MB, up to 256. Block size must be a power of
2, up to 1024.

## Results

Tesla T4 on Google Colab, run through `assignment_build_execution/run_assignments.sh`
with the root `assignment_config.yaml`. Each time is the mean of 3 trials after
a warm-up.

```
🚀 Project: EN605.617-GPU
HEAD is now at 55e712f Clear checksums between stream counts, add encrypt/decrypt sample to output
ℹ️  On branch 'main' — running git pull...
✅ git pull complete
Already up to date.
🚀 BUILD: executing build.sh 
✅ BUILD completed successfully
🚀 RUN [1/5]: executing run.sh 64 256
✅ RUN [1/5] completed successfully
64 MB file | 256 threads per block
16 chunks of 4 MB | 4096 blocks per kernel | averaged over 3 trials

Streams   Time (ms)   Throughput (GB/s)   Speedup   Saved vs 1 stream (ms)
      1       12.35                5.43     1.00x                     0.00
      2        7.26                9.25     1.70x                     5.10
      4        7.04                9.53     1.75x                     5.31
      8        6.84                9.82     1.81x                     5.52

Fastest: 8 streams (1.81x)

Sample (first 4 words of the file):
  Original : 947a4464 f6250b29 c5603586 d1662d9c
  Encrypted: 1635b83a f1041784 05dfe4a7 5cda9adb
  Decrypted: 947a4464 f6250b29 c5603586 d1662d9c
Chunk 0 checksum: GPU 80cb83f8 | CPU 80cb83f8

Verification: PASS (data changed, checksums match, decrypting restores original)
🚀 RUN [2/5]: executing run.sh 256 256
✅ RUN [2/5] completed successfully
256 MB file | 256 threads per block
64 chunks of 4 MB | 4096 blocks per kernel | averaged over 3 trials

Streams   Time (ms)   Throughput (GB/s)   Speedup   Saved vs 1 stream (ms)
      1       52.17                5.15     1.00x                     0.00
      2       27.70                9.69     1.88x                    24.46
      4       28.43                9.44     1.84x                    23.74
      8       28.05                9.57     1.86x                    24.12

Fastest: 2 streams (1.88x)

Sample (first 4 words of the file):
  Original : 947a4464 f6250b29 c5603586 d1662d9c
  Encrypted: 1635b83a f1041784 05dfe4a7 5cda9adb
  Decrypted: 947a4464 f6250b29 c5603586 d1662d9c
Chunk 0 checksum: GPU 80cb83f8 | CPU 80cb83f8

Verification: PASS (data changed, checksums match, decrypting restores original)
🚀 RUN [3/5]: executing run.sh 256 128
✅ RUN [3/5] completed successfully
256 MB file | 128 threads per block
64 chunks of 4 MB | 8192 blocks per kernel | averaged over 3 trials

Streams   Time (ms)   Throughput (GB/s)   Speedup   Saved vs 1 stream (ms)
      1       51.14                5.25     1.00x                     0.00
      2       27.67                9.70     1.85x                    23.46
      4       28.40                9.45     1.80x                    22.73
      8       28.02                9.58     1.83x                    23.12

Fastest: 2 streams (1.85x)

Sample (first 4 words of the file):
  Original : 947a4464 f6250b29 c5603586 d1662d9c
  Encrypted: 1635b83a f1041784 05dfe4a7 5cda9adb
  Decrypted: 947a4464 f6250b29 c5603586 d1662d9c
Chunk 0 checksum: GPU 80cb83f8 | CPU 80cb83f8

Verification: PASS (data changed, checksums match, decrypting restores original)
🚀 RUN [4/5]: executing run.sh 256 512
✅ RUN [4/5] completed successfully
256 MB file | 512 threads per block
64 chunks of 4 MB | 2048 blocks per kernel | averaged over 3 trials

Streams   Time (ms)   Throughput (GB/s)   Speedup   Saved vs 1 stream (ms)
      1       51.76                5.19     1.00x                     0.00
      2       27.38                9.81     1.89x                    24.39
      4       28.33                9.48     1.83x                    23.43
      8       27.98                9.59     1.85x                    23.78

Fastest: 2 streams (1.89x)

Sample (first 4 words of the file):
  Original : 947a4464 f6250b29 c5603586 d1662d9c
  Encrypted: 1635b83a f1041784 05dfe4a7 5cda9adb
  Decrypted: 947a4464 f6250b29 c5603586 d1662d9c
Chunk 0 checksum: GPU 80cb83f8 | CPU 80cb83f8

Verification: PASS (data changed, checksums match, decrypting restores original)
🚀 RUN [5/5]: executing run.sh 256 1024
✅ RUN [5/5] completed successfully
256 MB file | 1024 threads per block
64 chunks of 4 MB | 1024 blocks per kernel | averaged over 3 trials

Streams   Time (ms)   Throughput (GB/s)   Speedup   Saved vs 1 stream (ms)
      1       54.22                4.95     1.00x                     0.00
      2       28.22                9.51     1.92x                    25.99
      4       28.41                9.45     1.91x                    25.81
      8       28.06                9.57     1.93x                    26.16

Fastest: 8 streams (1.93x)

Sample (first 4 words of the file):
  Original : 947a4464 f6250b29 c5603586 d1662d9c
  Encrypted: 1635b83a f1041784 05dfe4a7 5cda9adb
  Decrypted: 947a4464 f6250b29 c5603586 d1662d9c
Chunk 0 checksum: GPU 80cb83f8 | CPU 80cb83f8

Verification: PASS (data changed, checksums match, decrypting restores original)
✅ RUN summary: 5/5 configurations succeeded
ℹ️  Full output written to assignment_output.txt
```

## Block size comparison

256 MB file, from the four runs above.

```
Block size   1 stream (ms)   2 streams (ms)   Speedup
       128          51.14            27.67     1.85x
       256          52.17            27.70     1.88x
       512          51.76            27.38     1.89x
      1024          54.22            28.22     1.92x
```

## Analysis

**Streams roughly halve the run time.** With one stream, every chunk copies
in, runs both kernels, and copies out before the next chunk starts, so only
one PCIe direction is busy at a time. That gives about 5.2 GB/s. With two or
more streams, one chunk copies in while another copies out. The T4's copy
engines run both directions at once, and throughput rises to about 9.6 GB/s,
a 1.7–1.9x speedup.

**More than two streams does not help.** Two streams already keep both copy
directions busy, so 4 and 8 streams have nothing extra to overlap. Their times
are within 1 ms of the 2-stream time, which is run-to-run noise. The speedup
stays below 2x because the first chunk's copy in and the last chunk's copy out
cannot overlap with anything. On the 64 MB file there are only 16 chunks
instead of 64, so that startup and finish cost is a larger share of the total,
and the speedup is lower (1.81x vs about 1.85–1.92x).

**Block size has little effect.** The run is limited by copying, not by the
kernels. With one stream, 1024 threads per block is 2–3 ms slower than the
smaller block sizes. With two or more streams, kernel time is hidden behind
the copies and all block sizes finish in about 28 ms.
