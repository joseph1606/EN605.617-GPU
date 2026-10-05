// Module 5: the same stencil run five times, with one operand in a
// different CUDA memory space each pass, so host, global, constant,
// shared and register memory can be timed against each other. All five
// kernels call stencil_dot(), so only the memory space differs.
//
// Usage: assignment.exe [total_threads] [threads_per_block]
// Build: make TAPS=<n> to change the filter width (default 33).

#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Filter width, and so the reuse factor: each input sample is read by
// TAPS different output threads. Compile-time because it sizes the
// constant and register coefficient arrays.
#ifndef TAPS_DEFAULT
#define TAPS_DEFAULT 33
#endif
static constexpr int TAPS = TAPS_DEFAULT;

// Fixed problem size. The kernels use grid-stride loops, so changing the
// thread count changes occupancy, not the amount of work.
static constexpr size_t NUM_ELEMENTS = (1 << 22);
static constexpr size_t INPUT_COUNT = NUM_ELEMENTS + TAPS - 1;

static constexpr size_t INPUT_BYTES = INPUT_COUNT * sizeof(float);
static constexpr size_t OUTPUT_BYTES = NUM_ELEMENTS * sizeof(float);
static constexpr size_t COEFF_BYTES = TAPS * sizeof(float);

static constexpr int ITERATIONS = 20;
static constexpr unsigned int SEED = 67u;
static constexpr float ERROR_TOLERANCE = 1.0e-3f;
static constexpr long long DEFAULT_TOTAL_THREADS = (1 << 20);
static constexpr long long DEFAULT_BLOCK_SIZE = 256;
static constexpr int NUM_VARIANTS = 5;

// Constant memory. Every thread reads the same addresses, which is the
// broadcast pattern the constant cache serves.
__constant__ float c_coeff[TAPS];

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s:%d -- %s\n", __FILE__,          \
                    __LINE__, cudaGetErrorString(err));                    \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// Shared by all five kernels so one timing routine can launch any of them.
typedef void (*StencilKernel)(const float *, const float *, float *,
                              size_t);

struct Result {
    float ms;
    float max_error;
};

struct Buffers {
    float *h_coeff;
    float *h_in;
    float *h_ref;
    float *h_gpu;
    float *d_coeff;
    float *d_in;
    float *d_out;
    float *hm_in;   // Mapped host memory, host-side pointers.
    float *hm_out;
    float *hmd_in;  // The same memory, device-side pointers.
    float *hmd_out;
};

// Device code

// The only copy of the stencil arithmetic.
__device__ __forceinline__ float stencil_dot(const float *coeff,
                                             const float *window)
{
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < TAPS; ++k) {
        acc += coeff[k] * window[k];
    }
    return acc;
}

// Baseline: coefficients and samples both in global memory.
__global__ void stencil_global(const float *coeff, const float *in,
                               float *out, size_t n)
{
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    for (; i < n; i += stride) {
        out[i] = stencil_dot(coeff, &in[i]);
    }
}

// Coefficients from __constant__ instead of global memory.
__global__ void stencil_constant(const float *coeff, const float *in,
                                 float *out, size_t n)
{
    (void)coeff;

    const size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    for (; i < n; i += stride) {
        out[i] = stencil_dot(c_coeff, &in[i]);
    }
}

// Coefficients copied into registers. r_coeff is indexed only by unrolled
// counters, so it stays in registers instead of spilling to local memory.
__global__ void stencil_register(const float *coeff, const float *in,
                                 float *out, size_t n)
{
    float r_coeff[TAPS];
#pragma unroll
    for (int k = 0; k < TAPS; ++k) {
        r_coeff[k] = coeff[k];
    }

    const size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    for (; i < n; i += stride) {
        out[i] = stencil_dot(r_coeff, &in[i]);
    }
}

// Samples staged in shared memory: one per thread plus the halo, so each
// is fetched once rather than once per window. Block-uniform loop bound.
__global__ void stencil_shared(const float *coeff, const float *in,
                               float *out, size_t n)
{
    extern __shared__ float s_tile[];

    const unsigned int tile_count = blockDim.x + TAPS - 1;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    const size_t in_count = n + TAPS - 1;

    for (size_t base = (size_t)blockIdx.x * blockDim.x; base < n;
         base += stride) {
        for (unsigned int t = threadIdx.x; t < tile_count;
             t += blockDim.x) {
            const size_t src = base + t;
            s_tile[t] = (src < in_count) ? in[src] : 0.0f;
        }
        __syncthreads();

        const size_t i = base + threadIdx.x;
        if (i < n) {
            out[i] = stencil_dot(coeff, &s_tile[threadIdx.x]);
        }
        __syncthreads();
    }
}

// Same body as stencil_global, but in and out are mapped host memory, so
// every sample crosses PCIe instead of coming from device DRAM.
__global__ void stencil_host_mapped(const float *coeff, const float *in,
                                    float *out, size_t n)
{
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    for (; i < n; i += stride) {
        out[i] = stencil_dot(coeff, &in[i]);
    }
}

// Host helpers

// Normalised triangular blur weights.
static void build_coefficients(float *coeff)
{
    const int centre = TAPS / 2;
    float sum = 0.0f;

    for (int k = 0; k < TAPS; ++k) {
        coeff[k] = (float)(centre + 1 - abs(k - centre));
        sum += coeff[k];
    }
    for (int k = 0; k < TAPS; ++k) {
        coeff[k] /= sum;
    }
}

static void generate_input(float *in, size_t count)
{
    srand(SEED);
    for (size_t i = 0; i < count; ++i) {
        in[i] = (float)rand() / (float)RAND_MAX;
    }
}

// CPU reference, used as the correctness oracle.
static void stencil_host(const float *coeff, const float *in, float *out,
                         size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        float acc = 0.0f;
        for (int k = 0; k < TAPS; ++k) {
            acc += coeff[k] * in[i + k];
        }
        out[i] = acc;
    }
}

static float max_abs_error(const float *a, const float *b, size_t n)
{
    float worst = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const float diff = fabsf(a[i] - b[i]);
        if (diff > worst) {
            worst = diff;
        }
    }
    return worst;
}

// Minimum traffic a correct stencil must move, as GB/s.
static float effective_gbps(float ms)
{
    const double bytes = (double)(INPUT_BYTES + OUTPUT_BYTES);
    return (float)(bytes / ((double)ms * 1.0e6));
}

// Timing

static Result benchmark_kernel(StencilKernel kernel, const float *d_coeff,
                               const float *d_in, float *d_out,
                               float *h_gpu, const float *h_ref,
                               int num_blocks, int block_size,
                               size_t shmem_bytes)
{
    // Untimed warm-up: a first launch pays one-time setup costs.
    kernel<<<num_blocks, block_size, shmem_bytes>>>(d_coeff, d_in, d_out,
                                                    NUM_ELEMENTS);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERATIONS; ++i) {
        kernel<<<num_blocks, block_size, shmem_bytes>>>(d_coeff, d_in,
                                                        d_out,
                                                        NUM_ELEMENTS);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // cudaMemcpyDefault serves both device memory and the mapped host
    // buffer the final variant writes to.
    CUDA_CHECK(cudaMemcpy(h_gpu, d_out, OUTPUT_BYTES, cudaMemcpyDefault));

    Result result;
    result.ms = total_ms / (float)ITERATIONS;
    result.max_error = max_abs_error(h_gpu, h_ref, NUM_ELEMENTS);
    return result;
}

static bool print_results(const Result *results, int num_blocks,
                          int block_size)
{
    static const char *const names[NUM_VARIANTS] = {
        "Global", "Constant", "Register", "Shared", "Host (mapped)"};

    printf("%-15s %10s %9s %9s %11s %7s\n", "Memory", "Time (ms)", "GB/s",
           "Speedup", "Max error", "Status");
    printf("----------------------------------------------------------"
           "----------\n");

    bool passed = true;
    for (int v = 0; v < NUM_VARIANTS; ++v) {
        const bool ok = (results[v].max_error <= ERROR_TOLERANCE);
        passed = passed && ok;
        printf("%-15s %10.4f %9.2f %8.2fx %11.2e %7s\n", names[v],
               results[v].ms, effective_gbps(results[v].ms),
               results[0].ms / results[v].ms, results[v].max_error,
               ok ? "PASS" : "FAIL");
    }
    printf("----------------------------------------------------------"
           "----------\n");
    printf("%s\n", passed ? "[+] All variants matched the CPU reference"
                          : "[-] At least one variant FAILED");

    printf("\nCSV,%d,%d", block_size, num_blocks);
    for (int v = 0; v < NUM_VARIANTS; ++v) {
        printf(",%.6f", results[v].ms);
    }
    printf("\n");
    return passed;
}

// Setup and teardown

static void allocate_buffers(Buffers *buf)
{
    // Host memory: input generation and the reference result.
    buf->h_coeff = (float *)malloc(COEFF_BYTES);
    buf->h_in = (float *)malloc(INPUT_BYTES);
    buf->h_ref = (float *)malloc(OUTPUT_BYTES);
    buf->h_gpu = (float *)malloc(OUTPUT_BYTES);
    if (!buf->h_coeff || !buf->h_in || !buf->h_ref || !buf->h_gpu) {
        fprintf(stderr, "host allocation failed\n");
        exit(EXIT_FAILURE);
    }

    build_coefficients(buf->h_coeff);
    generate_input(buf->h_in, INPUT_COUNT);
    stencil_host(buf->h_coeff, buf->h_in, buf->h_ref, NUM_ELEMENTS);

    // Global memory.
    CUDA_CHECK(cudaMalloc(&buf->d_coeff, COEFF_BYTES));
    CUDA_CHECK(cudaMalloc(&buf->d_in, INPUT_BYTES));
    CUDA_CHECK(cudaMalloc(&buf->d_out, OUTPUT_BYTES));
    CUDA_CHECK(cudaMemcpy(buf->d_coeff, buf->h_coeff, COEFF_BYTES,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buf->d_in, buf->h_in, INPUT_BYTES,
                          cudaMemcpyHostToDevice));

    // Constant memory.
    CUDA_CHECK(cudaMemcpyToSymbol(c_coeff, buf->h_coeff, COEFF_BYTES));

    // Mapped host memory: pinned so the GPU can address it directly.
    CUDA_CHECK(cudaHostAlloc(&buf->hm_in, INPUT_BYTES,
                             cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(&buf->hm_out, OUTPUT_BYTES,
                             cudaHostAllocMapped));
    memcpy(buf->hm_in, buf->h_in, INPUT_BYTES);
    CUDA_CHECK(cudaHostGetDevicePointer(&buf->hmd_in, buf->hm_in, 0));
    CUDA_CHECK(cudaHostGetDevicePointer(&buf->hmd_out, buf->hm_out, 0));
}

static void free_buffers(Buffers *buf)
{
    CUDA_CHECK(cudaFree(buf->d_coeff));
    CUDA_CHECK(cudaFree(buf->d_in));
    CUDA_CHECK(cudaFree(buf->d_out));
    CUDA_CHECK(cudaFreeHost(buf->hm_in));
    CUDA_CHECK(cudaFreeHost(buf->hm_out));
    free(buf->h_coeff);
    free(buf->h_in);
    free(buf->h_ref);
    free(buf->h_gpu);
}

static bool run_experiment(int num_blocks, int block_size,
                           size_t shmem_bytes)
{
    Buffers buf;
    Result results[NUM_VARIANTS];

    allocate_buffers(&buf);
    results[0] = benchmark_kernel(stencil_global, buf.d_coeff, buf.d_in,
                                  buf.d_out, buf.h_gpu, buf.h_ref,
                                  num_blocks, block_size, 0);
    results[1] = benchmark_kernel(stencil_constant, buf.d_coeff, buf.d_in,
                                  buf.d_out, buf.h_gpu, buf.h_ref,
                                  num_blocks, block_size, 0);
    results[2] = benchmark_kernel(stencil_register, buf.d_coeff, buf.d_in,
                                  buf.d_out, buf.h_gpu, buf.h_ref,
                                  num_blocks, block_size, 0);
    results[3] = benchmark_kernel(stencil_shared, buf.d_coeff, buf.d_in,
                                  buf.d_out, buf.h_gpu, buf.h_ref,
                                  num_blocks, block_size, shmem_bytes);
    results[4] = benchmark_kernel(stencil_host_mapped, buf.d_coeff,
                                  buf.hmd_in, buf.hmd_out, buf.h_gpu,
                                  buf.h_ref, num_blocks, block_size, 0);

    const bool passed = print_results(results, num_blocks, block_size);
    free_buffers(&buf);

    return passed;
}

// Command line

static bool configure_launch(long long *total_threads,
                             long long block_size, int *num_blocks,
                             size_t *shmem_bytes)
{
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    if (block_size <= 0 || block_size > prop.maxThreadsPerBlock) {
        fprintf(stderr, "block size must be between 1 and %d\n",
                prop.maxThreadsPerBlock);
        return false;
    }
    if (block_size % prop.warpSize != 0) {
        printf("Warning: block size is not a multiple of the warp size "
               "(%d)\n", prop.warpSize);
    }

    // One sample per thread plus the halo.
    *shmem_bytes = ((size_t)block_size + TAPS - 1) * sizeof(float);
    if (*shmem_bytes > prop.sharedMemPerBlock) {
        fprintf(stderr, "block size %lld needs %zu bytes of shared "
                "memory, device allows %zu\n", block_size, *shmem_bytes,
                prop.sharedMemPerBlock);
        return false;
    }

    const long long blocks =
        (*total_threads + block_size - 1) / block_size;
    if (blocks > prop.maxGridSize[0]) {
        fprintf(stderr, "grid of %lld blocks exceeds the device limit of "
                "%d\n", blocks, prop.maxGridSize[0]);
        return false;
    }
    if (blocks * block_size != *total_threads) {
        *total_threads = blocks * block_size;
        printf("Warning: Total thread count is not evenly divisible by "
               "the block size\nThe total number of threads will be "
               "rounded up to %lld\n", *total_threads);
    }

    *num_blocks = (int)blocks;
    return true;
}

static bool parse_arguments(int argc, char **argv,
                            long long *total_threads,
                            long long *block_size, int *num_blocks,
                            size_t *shmem_bytes)
{
    *total_threads = DEFAULT_TOTAL_THREADS;
    *block_size = DEFAULT_BLOCK_SIZE;

    if (argc > 3) {
        fprintf(stderr, "too many arguments\n");
        return false;
    }
    if (argc >= 2) {
        *total_threads = atoll(argv[1]);
    }
    if (argc >= 3) {
        *block_size = atoll(argv[2]);
    }
    if (*total_threads <= 0) {
        fprintf(stderr, "total threads must be greater than zero\n");
        return false;
    }

    return configure_launch(total_threads, *block_size, num_blocks,
                            shmem_bytes);
}

static void print_configuration(long long total_threads,
                                long long block_size, int num_blocks,
                                size_t shmem_bytes)
{
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("[+] Device: %s, %d SMs, %zu KB shared memory per block\n",
           prop.name, prop.multiProcessorCount,
           prop.sharedMemPerBlock / 1024);
    printf("[+] Stencil: %d taps over %zu samples\n", TAPS, NUM_ELEMENTS);
    printf("[+] Grid: %d blocks x %lld threads = %lld threads\n",
           num_blocks, block_size, total_threads);
    printf("[+] Shared memory for the shared variant: %zu bytes/block\n",
           shmem_bytes);
    printf("[+] Each time is the mean of %d launches\n", ITERATIONS);
    printf("[+] NOTE: times cover the kernel only, not host transfers\n\n");
}

int main(int argc, char **argv)
{
    // Must precede any call that creates a context, so the mapped host
    // allocations are addressable by the device.
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));

    long long total_threads = 0;
    long long block_size = 0;
    int num_blocks = 0;
    size_t shmem_bytes = 0;

    if (!parse_arguments(argc, argv, &total_threads, &block_size,
                         &num_blocks, &shmem_bytes)) {
        fprintf(stderr, "usage: %s [total_threads] [threads_per_block]\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    print_configuration(total_threads, block_size, num_blocks,
                        shmem_bytes);

    if (!run_experiment(num_blocks, (int)block_size, shmem_bytes)) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
