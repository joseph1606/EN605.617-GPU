// Module 9: GPU photo denoising experiment
// cuRAND generates camera noise directly in device memory, NPP adds it to a
// real photo, removes it with three filters and scores every result. Each
// case is repeated with several seeds and reported as mean +/- std dev.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <curand.h>
#include <npp.h>

#define CHECK(call)                                                       \
    do {                                                                  \
        int err = (int)(call);                                            \
        if (err != 0) {                                                   \
            fprintf(stderr, "error %d at %s:%d\n", err, __FILE__,         \
                    __LINE__);                                            \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

constexpr int LEVELS = 3;
constexpr int TYPES = 3;
constexpr int ROWS = TYPES + 1;  // the noisy image plus each filter
constexpr int METRICS = 4;       // MAE, PSNR, SSIM, GPU ms
constexpr int TRIALS = 3;
constexpr unsigned long long SEEDS[TRIALS] = {67ULL, 21ULL, 16ULL};
constexpr float GAUSS_SIGMA[LEVELS] = {10.0f, 25.0f, 50.0f};  // gray levels
constexpr float SPECKLE_SIGMA[LEVELS] = {0.1f, 0.25f, 0.5f};  // x pixel
constexpr float SALT_PEPPER[LEVELS] = {0.02f, 0.05f, 0.15f};  // pixels hit
constexpr int MASK = 5;  // every filter looks at a 5x5 neighbourhood
constexpr int BILATERAL_RADIUS = MASK / 2;
constexpr float BILATERAL_VAL_SIGMA2 = 40.0f * 40.0f;
constexpr float BILATERAL_POS_SIGMA2 = 2.0f * 2.0f;
const char *NOISE_NAMES[TYPES] = {"gaussian", "speckle", "saltpepper"};
const char *ROW_NAMES[ROWS] = {"none", "gaussian", "median", "bilateral"};

struct Gpu {
    int w, h, step32;
    size_t count;
    NppiSize size;
    NppStreamContext ctx;
    curandGenerator_t gen;
    cudaEvent_t start, stop;
    Npp8u *clean, *noisy, *out, *mask, *buf, *host;
    Npp32f *img32, *noise, *scores;
    Npp64f *l1;
};

Npp8u *readPgm(const char *path, int *w, int *h)
{
    FILE *f = fopen(path, "rb");
    int maxVal;
    if (!f || fscanf(f, "P5 %d %d %d", w, h, &maxVal) != 3 ||
        maxVal != 255) {
        fprintf(stderr, "cannot read 8-bit PGM %s\n", path);
        exit(EXIT_FAILURE);
    }
    fgetc(f);
    size_t px = (size_t)*w * *h;
    Npp8u *pixels = (Npp8u *)malloc(px);
    if (fread(pixels, 1, px, f) != px) {
        fprintf(stderr, "PGM %s is shorter than its header says\n", path);
        exit(EXIT_FAILURE);
    }
    fclose(f);
    return pixels;
}

void writePgm(const char *path, const Npp8u *pixels, int w, int h)
{
    FILE *f = fopen(path, "wb");
    fprintf(f, "P5\n%d %d\n255\n", w, h);
    fwrite(pixels, 1, (size_t)w * h, f);
    fclose(f);
}

// CUDA 13 dropped nppGetStreamContext, so fill the context by hand
NppStreamContext makeContext()
{
    NppStreamContext ctx = {};
    cudaDeviceProp p;
    CHECK(cudaGetDevice(&ctx.nCudaDeviceId));
    CHECK(cudaGetDeviceProperties(&p, ctx.nCudaDeviceId));
    ctx.hStream = 0;
    ctx.nMultiProcessorCount = p.multiProcessorCount;
    ctx.nMaxThreadsPerMultiProcessor = p.maxThreadsPerMultiProcessor;
    ctx.nMaxThreadsPerBlock = p.maxThreadsPerBlock;
    ctx.nSharedMemPerBlock = p.sharedMemPerBlock;
    ctx.nCudaDevAttrComputeCapabilityMajor = p.major;
    ctx.nCudaDevAttrComputeCapabilityMinor = p.minor;
    return ctx;
}

// One scratch buffer, sized for the largest need of the median and metrics
size_t scratchSize(const Gpu &g)
{
    size_t psnr, ssim, l1;
    Npp32u median;
    NppiSize m = {MASK, MASK};
    CHECK(nppiPSNRGetBufferHostSize_8u_C1R_Ctx(g.size, &psnr, g.ctx));
    CHECK(nppiSSIMGetBufferHostSize_8u_C1R_Ctx(g.size, &ssim, g.ctx));
    CHECK(nppiNormDiffL1GetBufferHostSize_8u_C1R_Ctx(g.size, &l1, g.ctx));
    CHECK(nppiFilterMedianBorderGetBufferSize_8u_C1R_Ctx(
        g.size, m, &median, NPP_BORDER_REPLICATE, g.ctx));
    return std::max({psnr, ssim, l1, (size_t)median});
}

Gpu setup(const Npp8u *pixels, int w, int h)
{
    Gpu g;
    size_t px = (size_t)w * h;
    g.w = w;
    g.h = h;
    g.step32 = w * sizeof(Npp32f);
    g.count = (px + 1) & ~(size_t)1;  // cuRAND normals come in pairs
    g.size = {w, h};
    g.ctx = makeContext();
    CHECK(cudaMalloc(&g.clean, px));
    CHECK(cudaMalloc(&g.noisy, px));
    CHECK(cudaMalloc(&g.out, px));
    CHECK(cudaMalloc(&g.mask, px));
    CHECK(cudaMalloc(&g.img32, px * sizeof(Npp32f)));
    CHECK(cudaMalloc(&g.noise, g.count * sizeof(Npp32f)));
    CHECK(cudaMalloc(&g.scores, 2 * sizeof(Npp32f)));
    CHECK(cudaMalloc(&g.l1, sizeof(Npp64f)));
    CHECK(cudaMalloc(&g.buf, scratchSize(g)));
    CHECK(cudaMemcpy(g.clean, pixels, px, cudaMemcpyHostToDevice));
    g.host = (Npp8u *)malloc(px);
    CHECK(curandCreateGenerator(&g.gen, CURAND_RNG_PSEUDO_PHILOX4_32_10));
    CHECK(cudaEventCreate(&g.start));
    CHECK(cudaEventCreate(&g.stop));
    return g;
}

// Gaussian: clean + N(0, s). Speckle: clean + clean * N(0, s).
// Salt and pepper: cuRAND uniforms below p/2 become 0, above 1 - p/2 255.
void addNoise(Gpu &g, int type, int level)
{
    if (type == 2) {
        float p = SALT_PEPPER[level];
        CHECK(curandGenerateUniform(g.gen, g.noise, g.count));
        CHECK(nppiCopy_8u_C1R_Ctx(g.clean, g.w, g.noisy, g.w, g.size,
                                  g.ctx));
        CHECK(nppiCompareC_32f_C1R_Ctx(g.noise, g.step32, p / 2, g.mask,
                                       g.w, g.size, NPP_CMP_LESS, g.ctx));
        CHECK(nppiSet_8u_C1MR_Ctx(0, g.noisy, g.w, g.size, g.mask, g.w,
                                  g.ctx));
        CHECK(nppiCompareC_32f_C1R_Ctx(g.noise, g.step32, 1 - p / 2, g.mask,
                                       g.w, g.size, NPP_CMP_GREATER, g.ctx));
        CHECK(nppiSet_8u_C1MR_Ctx(255, g.noisy, g.w, g.size, g.mask, g.w,
                                  g.ctx));
        return;
    }
    float sigma = type == 0 ? GAUSS_SIGMA[level] : SPECKLE_SIGMA[level];
    CHECK(curandGenerateNormal(g.gen, g.noise, g.count, 0.0f, sigma));
    CHECK(nppiConvert_8u32f_C1R_Ctx(g.clean, g.w, g.img32, g.step32,
                                    g.size, g.ctx));
    if (type == 1)
        CHECK(nppiMul_32f_C1R_Ctx(g.img32, g.step32, g.noise, g.step32,
                                  g.noise, g.step32, g.size, g.ctx));
    CHECK(nppiAdd_32f_C1R_Ctx(g.img32, g.step32, g.noise, g.step32,
                              g.img32, g.step32, g.size, g.ctx));
    CHECK(nppiConvert_32f8u_C1R_Ctx(g.img32, g.step32, g.noisy, g.w,
                                    g.size, NPP_RND_NEAR, g.ctx));
}

void applyFilter(Gpu &g, int filter)
{
    NppiPoint origin = {0, 0};
    NppiSize mask = {MASK, MASK};
    NppiPoint anchor = {MASK / 2, MASK / 2};
    if (filter == 0)
        CHECK(nppiFilterGaussBorder_8u_C1R_Ctx(
            g.noisy, g.w, g.size, origin, g.out, g.w, g.size,
            NPP_MASK_SIZE_5_X_5, NPP_BORDER_REPLICATE, g.ctx));
    else if (filter == 1)
        CHECK(nppiFilterMedianBorder_8u_C1R_Ctx(
            g.noisy, g.w, g.size, origin, g.out, g.w, g.size, mask, anchor,
            g.buf, NPP_BORDER_REPLICATE, g.ctx));
    else
        CHECK(nppiFilterBilateralGaussBorder_8u_C1R_Ctx(
            g.noisy, g.w, g.size, origin, g.out, g.w, g.size,
            BILATERAL_RADIUS, 1, BILATERAL_VAL_SIGMA2, BILATERAL_POS_SIGMA2,
            NPP_BORDER_REPLICATE, g.ctx));
}

// MAE, PSNR and SSIM of img against the clean photo
void score(Gpu &g, const Npp8u *img, double *metrics)
{
    Npp32f scores[2];
    Npp64f l1;
    CHECK(nppiNormDiff_L1_8u_C1R_Ctx(g.clean, g.w, img, g.w, g.size, g.l1,
                                     g.buf, g.ctx));
    CHECK(nppiPSNR_8u_C1R_Ctx(g.clean, g.w, img, g.w, g.size, g.scores,
                              g.buf, g.ctx));
    CHECK(nppiSSIM_8u_C1R_Ctx(g.clean, g.w, img, g.w, g.size,
                              g.scores + 1, g.buf, g.ctx));
    CHECK(cudaMemcpy(&l1, g.l1, sizeof(l1), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(scores, g.scores, sizeof(scores),
                     cudaMemcpyDeviceToHost));
    metrics[0] = l1 / ((double)g.w * g.h);
    metrics[1] = scores[0];
    metrics[2] = scores[1];
}

void save(Gpu &g, const Npp8u *img, const char *name, int level,
          const char *noise, const char *filter)
{
    char path[256];
    snprintf(path, sizeof(path), "output/%s_L%d_%s_%s.pgm", name, level,
             noise, filter);
    CHECK(cudaMemcpy(g.host, img, (size_t)g.w * g.h,
                     cudaMemcpyDeviceToHost));
    writePgm(path, g.host, g.w, g.h);
}

void printRow(const char *noise, const char *filter, const double *sum,
              const double *sq)
{
    const char *formats[METRICS] = {"  %6.2f +/- %-5.2f", "  %6.2f +/- %-5.2f",
                                    "  %6.4f +/- %-7.4f", "  %6.3f +/- %.3f"};
    printf("%-11s %-10s", noise, filter);
    for (int k = 0; k < METRICS; k++) {
        double mean = sum[k] / TRIALS;
        double var = (sq[k] - TRIALS * mean * mean) / (TRIALS - 1);
        printf(formats[k], mean, sqrt(std::max(var, 0.0)));
    }
    printf("\n");
}

// Each seed adds fresh noise; the first seed's images are saved.
// Row 0 times making the noise, the other rows time each filter.
void runNoise(Gpu &g, int type, int level, const char *name)
{
    double sum[ROWS][METRICS] = {}, sq[ROWS][METRICS] = {};
    for (int t = 0; t < TRIALS; t++) {
        CHECK(curandSetPseudoRandomGeneratorSeed(g.gen, SEEDS[t]));
        CHECK(curandSetGeneratorOffset(g.gen, 0));
        for (int r = 0; r < ROWS; r++) {
            float ms;
            CHECK(cudaEventRecord(g.start));
            if (r == 0)
                addNoise(g, type, level - 1);
            else
                applyFilter(g, r - 1);
            CHECK(cudaEventRecord(g.stop));
            CHECK(cudaEventSynchronize(g.stop));
            CHECK(cudaEventElapsedTime(&ms, g.start, g.stop));
            const Npp8u *img = r == 0 ? g.noisy : g.out;
            double m[METRICS];
            score(g, img, m);
            m[3] = ms;
            for (int k = 0; k < METRICS; k++) {
                sum[r][k] += m[k];
                sq[r][k] += m[k] * m[k];
            }
            if (t == 0)
                save(g, img, name, level, NOISE_NAMES[type], ROW_NAMES[r]);
        }
    }
    for (int r = 0; r < ROWS; r++)
        printRow(NOISE_NAMES[type], ROW_NAMES[r], sum[r], sq[r]);
}

void release(Gpu &g)
{
    curandDestroyGenerator(g.gen);
    cudaEventDestroy(g.start);
    cudaEventDestroy(g.stop);
    cudaFree(g.clean);
    cudaFree(g.noisy);
    cudaFree(g.out);
    cudaFree(g.mask);
    cudaFree(g.buf);
    cudaFree(g.img32);
    cudaFree(g.noise);
    cudaFree(g.scores);
    cudaFree(g.l1);
    free(g.host);
}

int main(int argc, char **argv)
{
    int level = argc > 2 ? atoi(argv[2]) : 0;
    if (level < 1 || level > LEVELS) {
        printf("usage: %s <image.pgm> <level 1-%d>\n", argv[0], LEVELS);
        return 1;
    }
    const char *base = strrchr(argv[1], '/') ? strrchr(argv[1], '/') + 1
                                             : argv[1];
    char name[128];
    snprintf(name, sizeof(name), "%.*s", (int)strcspn(base, "."), base);

    int w, h;
    Npp8u *pixels = readPgm(argv[1], &w, &h);
    Gpu g = setup(pixels, w, h);
    printf("\n%s %dx%d  noise level %d  seeds %llu %llu %llu\n", name, w,
           h, level, SEEDS[0], SEEDS[1], SEEDS[2]);
    printf("%-11s %-10s  %-15s  %-15s  %-19s %s\n", "noise", "filter",
           "MAE", "PSNR (dB)", "SSIM", "GPU ms");

    // Warm up every noise and filter path so no timed call pays first use
    for (int i = 0; i < TYPES; i++) {
        addNoise(g, i, 0);
        applyFilter(g, i);
    }
    for (int n = 0; n < TYPES; n++)
        runNoise(g, n, level, name);
    release(g);
    free(pixels);
    return 0;
}
