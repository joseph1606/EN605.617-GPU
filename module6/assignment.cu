// Module 6: CUDA Streams and Events
// Encrypts a random "file" chunk by chunk (copy in, encrypt, checksum, copy out)
// and times the same kernels spread across 1, 2, 4 and 8 streams.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

constexpr unsigned int CHUNK_BYTES = 4u << 20;
constexpr unsigned int CHUNK_WORDS = CHUNK_BYTES / sizeof(unsigned int);
constexpr int MAX_STREAMS = 8;
constexpr int MAX_FILE_MB = 256;
constexpr int DEFAULT_FILE_MB = 256;
constexpr int DEFAULT_THREADS_PER_BLOCK = 256;
constexpr int MAX_THREADS_PER_BLOCK = 1024;
constexpr int NUM_TRIALS = 3;
constexpr unsigned int SECRET_KEY = 0x5EC12E7u;
constexpr unsigned int RANDOM_SEED = 42;
constexpr int SAMPLE_WORDS = 4;

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

struct Pipeline {
    cudaStream_t streams[MAX_STREAMS];
    cudaEvent_t done[MAX_STREAMS];
    unsigned int *d_data[MAX_STREAMS];
    unsigned int *d_sum[MAX_STREAMS];
    cudaEvent_t start, stop;
    int numChunks;
    int threadsPerBlock;
};

// XOR each word with a keystream based on its position in the file, so every
// chunk encrypts independently and running it twice gives the original back
__global__ void encryptKernel(unsigned int *data, unsigned int firstWord)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= CHUNK_WORDS)
        return;

    unsigned int key = (firstWord + i) ^ SECRET_KEY;
    key ^= key >> 16;
    key *= 0x7feb352du;
    key ^= key >> 15;
    key *= 0x846ca68bu;
    key ^= key >> 16;
    data[i] ^= key;
}

// Each block sums its words in shared memory, then adds its total to the chunk sum
__global__ void checksumKernel(const unsigned int *data, unsigned int *chunkSum)
{
    extern __shared__ unsigned int partial[];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    partial[threadIdx.x] = (i < CHUNK_WORDS) ? data[i] : 0;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        atomicAdd(chunkSum, partial[0]);
}

void generateFile(unsigned int *file, size_t numWords)
{
    srand(RANDOM_SEED);
    for (size_t i = 0; i < numWords; i++)
        file[i] = ((unsigned int)rand() << 16) ^ (unsigned int)rand();
}

// Returns elapsed ms for pushing every chunk of input through the streams
float runEncryption(Pipeline &p, int numStreams, const unsigned int *input,
                    unsigned int *output, unsigned int *checksums)
{
    int blocks = (CHUNK_WORDS + p.threadsPerBlock - 1) / p.threadsPerBlock;
    size_t sharedBytes = p.threadsPerBlock * sizeof(unsigned int);

    CUDA_CHECK(cudaEventRecord(p.start, p.streams[0]));
    for (int s = 1; s < numStreams; s++)
        CUDA_CHECK(cudaStreamWaitEvent(p.streams[s], p.start, 0));

    for (int c = 0; c < p.numChunks; c++) {
        int s = c % numStreams;
        cudaStream_t stream = p.streams[s];
        size_t offset = (size_t)c * CHUNK_WORDS;

        CUDA_CHECK(cudaMemcpyAsync(p.d_data[s], input + offset, CHUNK_BYTES,
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemsetAsync(p.d_sum[s], 0, sizeof(unsigned int), stream));
        encryptKernel<<<blocks, p.threadsPerBlock, 0, stream>>>(p.d_data[s], offset);
        checksumKernel<<<blocks, p.threadsPerBlock, sharedBytes, stream>>>(p.d_data[s], p.d_sum[s]);
        CUDA_CHECK(cudaMemcpyAsync(output + offset, p.d_data[s], CHUNK_BYTES,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(&checksums[c], p.d_sum[s], sizeof(unsigned int),
                                   cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaGetLastError());

    // Stream 0 waits for the others so the stop event covers all the work
    for (int s = 1; s < numStreams; s++) {
        CUDA_CHECK(cudaEventRecord(p.done[s], p.streams[s]));
        CUDA_CHECK(cudaStreamWaitEvent(p.streams[0], p.done[s], 0));
    }
    CUDA_CHECK(cudaEventRecord(p.stop, p.streams[0]));
    CUDA_CHECK(cudaEventSynchronize(p.stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, p.start, p.stop));
    return ms;
}

bool verifyResults(const unsigned int *original, const unsigned int *encrypted,
                   const unsigned int *roundTrip, const unsigned int *checksums,
                   int numChunks)
{
    size_t fileBytes = (size_t)numChunks * CHUNK_BYTES;
    bool changed = memcmp(original, encrypted, fileBytes) != 0;
    bool restored = memcmp(original, roundTrip, fileBytes) == 0;

    bool checksumsMatch = true;
    for (int c = 0; c < numChunks; c++) {
        unsigned int sum = 0;
        for (unsigned int w = 0; w < CHUNK_WORDS; w++)
            sum += encrypted[(size_t)c * CHUNK_WORDS + w];
        if (sum != checksums[c])
            checksumsMatch = false;
    }

    return changed && checksumsMatch && restored;
}

void printSample(const unsigned int *original, const unsigned int *encrypted,
                 const unsigned int *roundTrip, const unsigned int *checksums)
{
    const unsigned int *rows[] = {original, encrypted, roundTrip};
    const char *labels[] = {"Original ", "Encrypted", "Decrypted"};

    printf("\nSample (first %d words of the file):\n", SAMPLE_WORDS);
    for (int r = 0; r < 3; r++) {
        printf("  %s:", labels[r]);
        for (int w = 0; w < SAMPLE_WORDS; w++)
            printf(" %08x", rows[r][w]);
        printf("\n");
    }

    unsigned int cpuSum = 0;
    for (unsigned int w = 0; w < CHUNK_WORDS; w++)
        cpuSum += encrypted[w];
    printf("Chunk 0 checksum: GPU %08x | CPU %08x\n", checksums[0], cpuSum);
}

int main(int argc, char **argv)
{
    int fileMB = (argc > 1) ? atoi(argv[1]) : DEFAULT_FILE_MB;
    int threadsPerBlock = (argc > 2) ? atoi(argv[2]) : DEFAULT_THREADS_PER_BLOCK;
    int chunkMB = CHUNK_BYTES >> 20;

    // Block size must be a power of two for the shared memory reduction
    if (fileMB <= 0 || fileMB > MAX_FILE_MB || fileMB % chunkMB != 0 ||
        threadsPerBlock <= 0 || threadsPerBlock > MAX_THREADS_PER_BLOCK ||
        (threadsPerBlock & (threadsPerBlock - 1)) != 0) {
        fprintf(stderr, "Usage: %s [fileMB (multiple of %d, max %d)] "
                        "[threadsPerBlock (power of 2, max %d)]\n",
                argv[0], chunkMB, MAX_FILE_MB, MAX_THREADS_PER_BLOCK);
        return EXIT_FAILURE;
    }

    Pipeline p;
    p.numChunks = fileMB / chunkMB;
    p.threadsPerBlock = threadsPerBlock;
    size_t fileBytes = (size_t)p.numChunks * CHUNK_BYTES;
    size_t sumBytes = p.numChunks * sizeof(unsigned int);

    int blocksPerKernel = (CHUNK_WORDS + threadsPerBlock - 1) / threadsPerBlock;

    printf("%d MB file | %d threads per block\n", fileMB, threadsPerBlock);
    printf("%d chunks of %d MB | %d blocks per kernel | averaged over %d trials\n\n",
           p.numChunks, chunkMB, blocksPerKernel, NUM_TRIALS);

    // Pinned host memory so the async copies can overlap with kernels
    unsigned int *original, *encrypted, *roundTrip, *checksums, *roundTripSums;
    CUDA_CHECK(cudaMallocHost(&original, fileBytes));
    CUDA_CHECK(cudaMallocHost(&encrypted, fileBytes));
    CUDA_CHECK(cudaMallocHost(&roundTrip, fileBytes));
    CUDA_CHECK(cudaMallocHost(&checksums, sumBytes));
    CUDA_CHECK(cudaMallocHost(&roundTripSums, sumBytes));
    generateFile(original, fileBytes / sizeof(unsigned int));

    for (int s = 0; s < MAX_STREAMS; s++) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&p.streams[s], cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&p.done[s], cudaEventDisableTiming));
        CUDA_CHECK(cudaMalloc(&p.d_data[s], CHUNK_BYTES));
        CUDA_CHECK(cudaMalloc(&p.d_sum[s], sizeof(unsigned int)));
    }
    CUDA_CHECK(cudaEventCreate(&p.start));
    CUDA_CHECK(cudaEventCreate(&p.stop));

    // Warm-up so startup cost isn't in the 1-stream time
    runEncryption(p, 1, original, encrypted, checksums);

    printf("Streams   Time (ms)   Throughput (GB/s)   Speedup   Saved vs 1 stream (ms)\n");
    bool allPassed = true;
    float baselineMs = 0.0f;
    float fastestMs = 0.0f;
    int fastestStreams = 1;
    for (int numStreams = 1; numStreams <= MAX_STREAMS; numStreams *= 2) {
        memset(encrypted, 0, fileBytes);
        memset(roundTrip, 0, fileBytes);
        memset(checksums, 0, sumBytes);

        float totalMs = 0.0f;
        for (int t = 0; t < NUM_TRIALS; t++)
            totalMs += runEncryption(p, numStreams, original, encrypted, checksums);
        float ms = totalMs / NUM_TRIALS;
        if (numStreams == 1)
            baselineMs = ms;
        if (numStreams == 1 || ms < fastestMs) {
            fastestMs = ms;
            fastestStreams = numStreams;
        }

        // Encrypting the encrypted file again should give back the original
        runEncryption(p, numStreams, encrypted, roundTrip, roundTripSums);
        allPassed = allPassed &&
                    verifyResults(original, encrypted, roundTrip, checksums, p.numChunks);

        printf("%7d   %9.2f   %17.2f   %6.2fx   %22.2f\n", numStreams, ms,
               fileBytes / (ms * 1.0e6), baselineMs / ms, baselineMs - ms);
    }

    printf("\nFastest: %d streams (%.2fx)\n", fastestStreams, baselineMs / fastestMs);
    printSample(original, encrypted, roundTrip, checksums);
    printf("\nVerification: %s (data changed, checksums match, decrypting restores original)\n\n",
           allPassed ? "PASS" : "FAIL");

    for (int s = 0; s < MAX_STREAMS; s++) {
        CUDA_CHECK(cudaStreamDestroy(p.streams[s]));
        CUDA_CHECK(cudaEventDestroy(p.done[s]));
        CUDA_CHECK(cudaFree(p.d_data[s]));
        CUDA_CHECK(cudaFree(p.d_sum[s]));
    }
    CUDA_CHECK(cudaEventDestroy(p.start));
    CUDA_CHECK(cudaEventDestroy(p.stop));
    CUDA_CHECK(cudaFreeHost(original));
    CUDA_CHECK(cudaFreeHost(encrypted));
    CUDA_CHECK(cudaFreeHost(roundTrip));
    CUDA_CHECK(cudaFreeHost(checksums));
    CUDA_CHECK(cudaFreeHost(roundTripSums));

    return allPassed ? EXIT_SUCCESS : EXIT_FAILURE;
}
