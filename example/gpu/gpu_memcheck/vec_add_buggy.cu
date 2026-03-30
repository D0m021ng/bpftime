/*
 * vec_add_buggy.cu - 故意含有显存非法访问的 CUDA 向量加法程序
 *
 * 演示三种典型的 GPU 显存越界场景：
 *
 *   场景 1 (OOB_CLASSIC)  : launch 线程数多于数组元素数，尾部线程越界
 *   场景 2 (OOB_STRIDE)   : grid-stride loop 边界未对齐，部分线程越界
 *   场景 3 (SAFE_KERNEL)  : 添加了正确的 bounds guard，无越界
 *
 * 运行方法（配合 gpu_memcheck 使用）：
 *   Terminal 1: ./gpu_memcheck 1024
 *   Terminal 2: LD_PRELOAD=...libbpftime-agent.so ./vec_add_buggy [场景编号]
 *
 * 场景编号：1（默认）= OOB_CLASSIC, 2 = OOB_STRIDE, 3 = SAFE_KERNEL
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <unistd.h>

/* =====================================================================
 * 数组规模定义
 * ===================================================================== */
#define N_ELEMENTS      1024    /* 实际分配的数组元素数 */
#define N_THREADS_BUGGY 1280    /* 有意多分配 25% 的线程（>N_ELEMENTS）*/
#define THREADS_PER_BLK  256    /* 每个 block 的线程数 */

/* =====================================================================
 * 场景 1：经典越界（无边界检查）
 *
 * 问题：launch 了 ceil(N_THREADS_BUGGY/256) = 5 个 block，共 1280 个线程，
 *       但数组只有 1024 个元素。索引 1024~1279 的线程将写入未分配的显存。
 * ===================================================================== */
__global__ void vectorAdd(const float *A, const float *B, float *C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    /* ⚠️ 缺少 if (idx < N) 的 bounds guard —— 这是 bug！ */
    C[idx] = A[idx] + B[idx];
}

/* =====================================================================
 * 场景 2：Grid-stride loop 边界未对齐越界
 *
 * 问题：当 N_ELEMENTS 不能被 gridDim.x * blockDim.x 整除时，
 *       最后一批迭代的部分线程仍会进入循环体并越界访问。
 * ===================================================================== */
__global__ void vectorAdd_OOB_stride(const float *A, const float *B,
                                      float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    /*
     * ⚠️ 若 N 不被 stride 整除，最后一轮循环中越界的线程
     *    会执行 C[idx] = A[idx] + B[idx]，其中 idx >= N。
     *
     * 正确写法应为：while (idx < N) { ...; idx += stride; }
     */
    while (idx <= N) {   /* <= 而非 <，故意多走一步 */
        C[idx] = A[idx] + B[idx];
        idx += stride;
    }
}

/* =====================================================================
 * 场景 3：正确实现（有边界检查），用于对比
 * ===================================================================== */
__global__ void vectorAdd_SAFE(const float *A, const float *B,
                                float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {          /* ✅ 正确的 bounds guard */
        C[idx] = A[idx] + B[idx];
    }
}

/* =====================================================================
 * CUDA 错误检查宏
 * ===================================================================== */
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _err = (call);                                          \
        if (_err != cudaSuccess) {                                          \
            fprintf(stderr, "[CUDA ERROR] %s:%d  %s: %s\n",               \
                    __FILE__, __LINE__, #call,                              \
                    cudaGetErrorString(_err));                              \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

/* =====================================================================
 * main
 * ===================================================================== */
int main(int argc, char **argv)
{
    int scenario = (argc > 1) ? atoi(argv[1]) : 1;

    if (scenario < 1 || scenario > 3) {
        fprintf(stderr, "Usage: %s [1|2|3]\n", argv[0]);
        fprintf(stderr, "  1: OOB_CLASSIC (default)\n");
        fprintf(stderr, "  2: OOB_STRIDE\n");
        fprintf(stderr, "  3: SAFE_KERNEL\n");
        return 1;
    }

    /* ── 分配 host 内存 ─────────────────────────── */
    size_t bytes = N_ELEMENTS * sizeof(float);
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    for (int i = 0; i < N_ELEMENTS; i++) {
        h_A[i] = (float)i;
        h_B[i] = (float)(i * 2);
    }

    /* ── 分配 GPU 显存（仅 N_ELEMENTS 个元素！） ── */
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));   /* 仅 1024 * 4 = 4096 字节 */
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    /* ── 打印运行配置 ────────────────────────────── */
    printf("======================================\n");
    printf(" GPU Memory Bug Demo — Scenario %d\n", scenario);
    printf("======================================\n");
    printf(" Array size (N)    : %d elements (%zu bytes)\n",
           N_ELEMENTS, bytes);

    switch (scenario) {
    case 1: {
        /* 场景 1：经典越界 */
        int num_blocks = (N_THREADS_BUGGY + THREADS_PER_BLK - 1)
                         / THREADS_PER_BLK;
        int total_threads = num_blocks * THREADS_PER_BLK;

        printf(" Launch config     : %d blocks × %d threads = %d threads\n",
               num_blocks, THREADS_PER_BLK, total_threads);
        printf(" OOB threads       : %d (index %d ~ %d)\n",
               total_threads - N_ELEMENTS, N_ELEMENTS, total_threads - 1);
        printf(" Bug type          : Missing bounds guard\n");
        printf("--------------------------------------\n");
        printf(" gpu_memcheck cmd  : ./gpu_memcheck %d\n\n", N_ELEMENTS);

        for (int iter = 0; iter < 10; iter++) {
            CUDA_CHECK(cudaMemset(d_C, 0, bytes));
            vectorAdd<<<num_blocks, THREADS_PER_BLK>>>(
                d_A, d_B, d_C);
            CUDA_CHECK(cudaDeviceSynchronize());
            printf("[iter %2d] kernel done — %d OOB threads!\n",
                   iter + 1, total_threads - N_ELEMENTS);
            sleep(1);
        }
        break;
    }
    case 2: {
        /*
         * 场景 2：Grid-stride 越界（启动线程数超过数组大小 + loop 边界 <= 错误）
         *
         * 启动 1280 线程（同场景 1），gpu_memcheck 在入口检测到线程 1024~1279
         * 的 linear_tid >= N，上报 256 个 OOB 事件。
         * 注意：循环内部 idx=1024（线程 256 第 2 次迭代）的额外越界由入口
         * kprobe 无法捕获，这是 grid-stride 模式的检测限制。
         */
        int num_blocks = (N_THREADS_BUGGY + THREADS_PER_BLK - 1)
                         / THREADS_PER_BLK;   /* 5 blocks = 1280 threads */
        int total_threads = num_blocks * THREADS_PER_BLK;

        printf(" Launch config     : %d blocks × %d threads = %d threads\n",
               num_blocks, THREADS_PER_BLK, total_threads);
        printf(" OOB threads       : %d (index %d ~ %d at entry)\n",
               total_threads - N_ELEMENTS, N_ELEMENTS, total_threads - 1);
        printf(" Bug type          : over-launch + grid-stride loop uses <= instead of <\n");
        printf("--------------------------------------\n");
        printf(" gpu_memcheck cmd  : ./gpu_memcheck %d\n\n", N_ELEMENTS);

        for (int iter = 0; iter < 10; iter++) {
            CUDA_CHECK(cudaMemset(d_C, 0, bytes));
            vectorAdd_OOB_stride<<<num_blocks, THREADS_PER_BLK>>>(
                d_A, d_B, d_C, N_ELEMENTS);
            CUDA_CHECK(cudaDeviceSynchronize());
            printf("[iter %2d] kernel done — stride boundary violated!\n",
                   iter + 1);
            sleep(1);
        }
        break;
    }
    case 3: {
        /*
         * 场景 3：正确实现（对比）
         * 精确启动 N 个线程（4 blocks × 256 = 1024），配合 if (idx < N) guard，
         * 所有线程初始 linear_tid < N，入口检查不触发任何告警。
         */
        int num_blocks = (N_ELEMENTS + THREADS_PER_BLK - 1)
                         / THREADS_PER_BLK;   /* 4 blocks = 1024 threads */

        printf(" Launch config     : %d blocks × %d threads = %d threads\n",
               num_blocks, THREADS_PER_BLK,
               num_blocks * THREADS_PER_BLK);
        printf(" Bounds guard      : if (idx < N) — CORRECT\n");
        printf(" Expected          : No violations\n");
        printf("--------------------------------------\n");
        printf(" gpu_memcheck cmd  : ./gpu_memcheck %d\n\n", N_ELEMENTS);

        for (int iter = 0; iter < 10; iter++) {
            CUDA_CHECK(cudaMemset(d_C, 0, bytes));
            vectorAdd_SAFE<<<num_blocks, THREADS_PER_BLK>>>(
                d_A, d_B, d_C, N_ELEMENTS);
            CUDA_CHECK(cudaDeviceSynchronize());
            printf("[iter %2d] kernel done — no OOB\n", iter + 1);
            sleep(1);
        }

        /* 验证结果 */
        CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
        printf("\nVerification (first 5 elements):\n");
        bool ok = true;
        for (int i = 0; i < 5; i++) {
            float expected = h_A[i] + h_B[i];
            printf("  C[%d] = %.1f (expected %.1f) %s\n",
                   i, h_C[i], expected,
                   (h_C[i] == expected) ? "✓" : "✗");
            if (h_C[i] != expected) ok = false;
        }
        printf("Result: %s\n", ok ? "PASS" : "FAIL");
        break;
    }
    }

    /* ── 释放资源 ─────────────────────────────────── */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
