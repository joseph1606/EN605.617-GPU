//Based on the work of Andrew Krepps
//
// Module 3: solves a batch of quadratic equations on the CPU and the GPU,
// once without an if/else and once with one, and times each version.
//
// Usage: assignment.exe <total threads> <threads per block>

#include <cuda_runtime.h>

#include <chrono>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define ITERATIONS 20
#define SEED       67u

#define CUDA_CHECK(call)                                                       \
    do { cudaError_t err = (call);                                             \
         if (err != cudaSuccess) {                                             \
             fprintf(stderr, "CUDA error %s:%d -- %s\n", __FILE__, __LINE__,   \
                     cudaGetErrorString(err));                                 \
             exit(EXIT_FAILURE); } } while (0)

// No bounds check needed, since the grid covers the data exactly.
__global__ void solveBranchless(float* roots, const float* a, const float* b,
                                const float* c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float disc = b[i] * b[i] - 4.0f * a[i] * c[i];
    roots[i] = (-b[i] + sqrtf(fabsf(disc))) / (2.0f * a[i]);
}

// Real roots when the discriminant is non-negative, otherwise the imaginary
// part of the complex pair.
__global__ void solveBranching(float* roots, const float* a, const float* b,
                               const float* c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float disc = b[i] * b[i] - 4.0f * a[i] * c[i];

    if (disc >= 0.0f) {
        roots[i] = (-b[i] + sqrtf(disc)) / (2.0f * a[i]);
    } else {
        roots[i] = sqrtf(-disc) / (2.0f * a[i]);
    }
}

// CPU version of solveBranchless.
static void solveBranchlessHost(float* roots, const float* a, const float* b,
                                const float* c, int n)
{
    for (int i = 0; i < n; ++i) {
        float disc = b[i] * b[i] - 4.0f * a[i] * c[i];
        roots[i] = (-b[i] + sqrtf(fabsf(disc))) / (2.0f * a[i]);
    }
}

// CPU version of solveBranching.
static void solveBranchingHost(float* roots, const float* a, const float* b,
                               const float* c, int n)
{
    for (int i = 0; i < n; ++i) {
        float disc = b[i] * b[i] - 4.0f * a[i] * c[i];

        if (disc >= 0.0f) {
            roots[i] = (-b[i] + sqrtf(disc)) / (2.0f * a[i]);
        } else {
            roots[i] = sqrtf(-disc) / (2.0f * a[i]);
        }
    }
}

// Picks the discriminant first, then solves for c so that b*b - 4ac equals it.
// Odd elements get a negative one, so neighbouring threads take different
// sides of the if/else.
static void generateData(float* a, float* b, float* c, int n)
{
    srand(SEED);

    for (int i = 0; i < n; ++i) {
        a[i] = 0.5f + ((float)rand() / RAND_MAX) * 2.0f;
        b[i] = (((float)rand() / RAND_MAX) * 2.0f - 1.0f) * 4.0f;

        float disc = 1.0f + ((float)rand() / RAND_MAX) * 15.0f;   // 1 to 16, never near 0
        if (i % 2 != 0) {
            disc = -disc;
        }
        c[i] = (b[i] * b[i] - disc) / (4.0f * a[i]);
    }
}

// CUDA events timestamp the GPU's own queue. 
static float timeKernel(bool branching, float* roots, const float* a,
                        const float* b, const float* c, int blocks, int threads)
{
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < ITERATIONS; ++i) {
        if (branching) {
            solveBranching<<<blocks, threads>>>(roots, a, b, c);
        } else {
            solveBranchless<<<blocks, threads>>>(roots, a, b, c);
        }
    }
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaGetLastError());   

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return ms / ITERATIONS;
}

static float timeHost(bool branching, float* roots, const float* a,
                      const float* b, const float* c, int n)
{
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < ITERATIONS; ++i) {
        if (branching) {
            solveBranchingHost(roots, a, b, c, n);
        } else {
            solveBranchlessHost(roots, a, b, c, n);
        }
    }
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    return (float)(std::chrono::duration<double, std::milli>(t1 - t0).count() / ITERATIONS);
}

int main(int argc, char** argv)
{
	// read command line arguments
	int totalThreads = (1 << 20);
	int blockSize = 256;

	if (argc >= 2) {
		totalThreads = atoi(argv[1]);
	}
	if (argc >= 3) {
		blockSize = atoi(argv[2]);
	}

    if (totalThreads <= 0 || blockSize <= 0) {
        fprintf(stderr, "usage: %s <total threads> <threads per block>\n", argv[0]);
        return EXIT_FAILURE;
    }

	int numBlocks = totalThreads/blockSize;

	// validate command line arguments
	if (totalThreads % blockSize != 0) {
		++numBlocks;
		totalThreads = numBlocks*blockSize;

		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (blockSize > prop.maxThreadsPerBlock) {
        fprintf(stderr, "block size %d exceeds device limit of %d\n",
                blockSize, prop.maxThreadsPerBlock);
        return EXIT_FAILURE;
    }

    const int n = totalThreads;             // one equation per thread
    const size_t bytes = (size_t)n * sizeof(float);

    printf("Device: %s, %d SMs\n", prop.name, prop.multiProcessorCount);
    printf("Equations: %d   Grid: %d blocks x %d threads   Mean of %d runs\n\n",
           n, numBlocks, blockSize, ITERATIONS);

    float* a = (float*)malloc(bytes);
    float* b = (float*)malloc(bytes);
    float* c = (float*)malloc(bytes);
    float* cpuRoots = (float*)malloc(bytes);
    float* gpuRoots = (float*)malloc(bytes);
    generateData(a, b, c, n);

    float *da, *db, *dc, *dRoots;
    CUDA_CHECK(cudaMalloc((void**)&da, bytes));
    CUDA_CHECK(cudaMalloc((void**)&db, bytes));
    CUDA_CHECK(cudaMalloc((void**)&dc, bytes));
    CUDA_CHECK(cudaMalloc((void**)&dRoots, bytes));

    // Warm up: the first copy and first launch of each kernel pay one-time
    // setup costs, so do them untimed.
    CUDA_CHECK(cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dc, c, bytes, cudaMemcpyHostToDevice));
    solveBranchless<<<numBlocks, blockSize>>>(dRoots, da, db, dc);
    solveBranching<<<numBlocks, blockSize>>>(dRoots, da, db, dc);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    CUDA_CHECK(cudaEventRecord(t0));
    CUDA_CHECK(cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dc, c, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float upMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&upMs, t0, t1));

    // Both kernels run back to back so the GPU does not idle between them.
    float gpuPlain = timeKernel(false, dRoots, da, db, dc, numBlocks, blockSize);
    float gpuBranch = timeKernel(true, dRoots, da, db, dc, numBlocks, blockSize);

    CUDA_CHECK(cudaEventRecord(t0));
    CUDA_CHECK(cudaMemcpy(gpuRoots, dRoots, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float downMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&downMs, t0, t1));

    float cpuPlain = timeHost(false, cpuRoots, a, b, c, n);
    float cpuBranch = timeHost(true, cpuRoots, a, b, c, n);

    printf("%-16s %10s %10s %10s\n", "", "CPU (ms)", "GPU (ms)", "speedup");
    printf("%-16s %10.4f %10.4f %9.2fx\n", "no branching", cpuPlain, gpuPlain, cpuPlain / gpuPlain);
    printf("%-16s %10.4f %10.4f %9.2fx\n", "branching", cpuBranch, gpuBranch, cpuBranch / gpuBranch);

    printf("\nbranch cost     GPU %.2f%%   CPU %.2f%%\n",
           (gpuBranch / gpuPlain - 1.0f) * 100.0f,
           (cpuBranch / cpuPlain - 1.0f) * 100.0f);
    printf("transfers       %.4f ms up, %.4f ms down\n", upMs, downMs);
    printf("GPU round trip  %.4f ms\n", gpuPlain + upMs + downMs);

    printf("\nCSV,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
           n, blockSize, numBlocks, gpuPlain, gpuBranch, cpuPlain, cpuBranch,
           upMs, downMs);

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dc));
    CUDA_CHECK(cudaFree(dRoots));
    free(a); free(b); free(c); free(cpuRoots); free(gpuRoots);

    return EXIT_SUCCESS;
}
