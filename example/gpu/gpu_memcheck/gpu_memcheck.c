// SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
/*
 * gpu_memcheck.c - GPU 显存非法访问检测工具（用户态加载器）
 *
 * 用法：
 *   Terminal 1（server，加载 eBPF 程序）：
 *     BPFTIME_LOG_OUTPUT=console \
 *     LD_PRELOAD=build/runtime/syscall-server/libbpftime-syscall-server.so \
 *     example/gpu/gpu_memcheck/gpu_memcheck <array_size_N> [ptr_a] [ptr_b] [ptr_c]
 *
 *   Terminal 2（agent，运行目标 CUDA 应用）：
 *     BPFTIME_LOG_OUTPUT=console \
 *     LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
 *     example/gpu/gpu_memcheck/vec_add_buggy
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <stdbool.h>
#include <inttypes.h>
#include <unistd.h>
#include <errno.h>
#include <sys/resource.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include "./.output/gpu_memcheck.skel.h"

/* =====================================================================
 * 与 BPF 侧同步的结构体定义
 * ===================================================================== */

#define VIOLATION_OOB       0
#define VIOLATION_NULL_A    1
#define VIOLATION_NULL_B    2
#define VIOLATION_NULL_C    3

struct violation_event {
	uint64_t timestamp_ns;
	uint64_t linear_tid;
	uint64_t thread_x, thread_y, thread_z;
	uint64_t block_x,  block_y,  block_z;
	uint64_t expected_max;
	uint64_t overflow_by;
	uint32_t violation_type;
	uint32_t reserved;
};

struct bounds_config {
	uint64_t total_n;
	uint64_t ptr_a;
	uint64_t ptr_b;
	uint64_t ptr_c;
	uint32_t elem_size;
	uint32_t enabled;
};

struct alloc_info {
	uint64_t ptr;
	uint64_t size;
	uint64_t alloc_ts;
};

/* =====================================================================
 * 全局状态
 * ===================================================================== */

static volatile bool exiting = false;
static uint64_t total_violations = 0;
static uint64_t oob_count  = 0;
static uint64_t null_count = 0;

static void sig_handler(int sig)
{
	exiting = true;
}

/* =====================================================================
 * libbpf 日志回调（静默模式，减少噪音）
 * ===================================================================== */
static int libbpf_print_fn(enum libbpf_print_level level,
			   const char *format, va_list args)
{
	/* 只打印 WARNING 及以上级别 */
	if (level < LIBBPF_WARN)
		return 0;
	return vfprintf(stderr, format, args);
}

/* =====================================================================
 * 输出辅助函数
 * ===================================================================== */

static void print_timestamp(void)
{
	time_t t = time(NULL);
	struct tm *tm_info = localtime(&t);
	char buf[32];
	strftime(buf, sizeof(buf), "%H:%M:%S", tm_info);
	printf("[%s] ", buf);
}

static const char *violation_type_str(uint32_t type)
{
	switch (type) {
	case VIOLATION_OOB:    return "OUT-OF-BOUNDS";
	case VIOLATION_NULL_A: return "NULL-PTR (array A)";
	case VIOLATION_NULL_B: return "NULL-PTR (array B)";
	case VIOLATION_NULL_C: return "NULL-PTR (array C)";
	default:               return "UNKNOWN";
	}
}

/*
 * poll_callback：每次从 GPU ring buffer 读到一个事件时调用
 *
 * 由 bpftime_syscall_server__poll_gpu_ringbuf_map 内部调用。
 * 参数 ctx 是我们传入的 void* 上下文（此处为 NULL）。
 */
static void poll_callback(const void *data, uint64_t size, void *ctx)
{
	const struct violation_event *evt = data;

	total_violations++;
	if (evt->violation_type == VIOLATION_OOB)
		oob_count++;
	else
		null_count++;

	/* ── 打印告警 ─────────────────────────────────── */
	print_timestamp();
	printf("*** GPU MEMORY VIOLATION #%" PRIu64 " ***\n", total_violations);
	printf("  ├─ Type      : %s\n", violation_type_str(evt->violation_type));

	if (evt->violation_type == VIOLATION_OOB) {
		printf("  ├─ LinearTID : %" PRIu64
		       "  (valid range: 0 ~ %" PRIu64 ")\n",
		       evt->linear_tid, evt->expected_max - 1);
		printf("  ├─ Overflow  : +%" PRIu64 " elements beyond bound\n",
		       evt->overflow_by);
	}

	printf("  ├─ Thread    : (%" PRIu64 ", %" PRIu64 ", %" PRIu64 ")\n",
	       evt->thread_x, evt->thread_y, evt->thread_z);
	printf("  ├─ Block     : (%" PRIu64 ", %" PRIu64 ", %" PRIu64 ")\n",
	       evt->block_x,  evt->block_y,  evt->block_z);
	printf("  └─ GPU Time  : %" PRIu64 " ns\n\n", evt->timestamp_ns);
	fflush(stdout);
}

/* =====================================================================
 * 显存分配表打印
 * ===================================================================== */

static void print_alloc_table(int alloc_fd)
{
	uint64_t key = 0, next_key;
	struct alloc_info info;
	int count = 0;

	printf("\n┌─ GPU Memory Allocation Table ─────────────────────────────────┐\n");
	printf("│  %-20s  %-14s  %-22s  │\n",
	       "Address", "Size (bytes)", "Alloc Time (CPU ns)");
	printf("│  %-20s  %-14s  %-22s  │\n",
	       "──────────────────", "────────────", "──────────────────────");

	while (bpf_map_get_next_key(alloc_fd,
				    count == 0 ? NULL : &key,
				    &next_key) == 0) {
		key = next_key;
		if (bpf_map_lookup_elem(alloc_fd, &key, &info) == 0) {
			printf("│  0x%-18" PRIx64 "  %-14" PRIu64
			       "  %-22" PRIu64 "  │\n",
			       info.ptr, info.size, info.alloc_ts);
			count++;
		}
	}

	if (count == 0)
		printf("│  (no allocations tracked yet)                                   │\n");

	printf("├─────────────────────────────────────────────────────────────────┤\n");
	printf("│  Total: %-3d active allocation(s)                                │\n", count);
	printf("└─────────────────────────────────────────────────────────────────┘\n\n");
}

/* =====================================================================
 * 每线程违规统计打印
 * ===================================================================== */

static void print_violation_stats(int stats_fd, uint64_t thread_count)
{
	uint64_t values[4096] = {};
	uint32_t key = 0;
	uint64_t total = 0;
	uint64_t violating = 0;

	if (bpf_map_lookup_elem(stats_fd, &key, values) != 0)
		return;

	printf("┌─ Per-Thread Violation Statistics ─────────────────────────────┐\n");
	printf("│  Thread  │ OOB Count  │ Status                                │\n");
	printf("│  ──────  │ ─────────  │ ──────                                │\n");

	for (uint64_t i = 0; i < thread_count && i < 4096; i++) {
		if (values[i] > 0) {
			printf("│  %-6" PRIu64 "  │ %-10" PRIu64
			       " │ %-38s│\n",
			       i, values[i], "!!! OUT-OF-BOUNDS !!!");
			total += values[i];
			violating++;
		}
	}

	if (total == 0)
		printf("│  No per-thread violations recorded yet.                      │\n");

	printf("├───────────────────────────────────────────────────────────────┤\n");
	printf("│  Violating threads: %-4" PRIu64
	       "  Total violation count: %-8" PRIu64 "  │\n",
	       violating, total);
	printf("└───────────────────────────────────────────────────────────────┘\n\n");
}

/* =====================================================================
 * 用法帮助
 * ===================================================================== */

static void usage(const char *prog)
{
	fprintf(stderr,
		"Usage: %s <N> [ptr_a] [ptr_b] [ptr_c]\n"
		"\n"
		"  N      : 数组元素个数（合法访问范围 0 ~ N-1）\n"
		"  ptr_a  : GPU array A 基地址（十六进制，可选，0 = 不检测 null）\n"
		"  ptr_b  : GPU array B 基地址（可选）\n"
		"  ptr_c  : GPU array C 基地址（可选）\n"
		"\n"
		"环境变量：\n"
		"  BPFTIME_MAP_GPU_THREAD_COUNT : 监控的最大线程数（默认 1024）\n"
		"\n"
		"示例：\n"
		"  %s 1024\n"
		"  %s 1024 0x7f0000000000\n",
		prog, prog, prog);
}

/* =====================================================================
 * main
 * ===================================================================== */

int main(int argc, char **argv)
{
	struct gpu_memcheck_bpf *skel = NULL;
	int err = 0;
	uint64_t total_n;
	uint64_t ptr_a = 0, ptr_b = 0, ptr_c = 0;

	/* ── 参数解析 ─────────────────────────────────── */
	if (argc < 2) {
		usage(argv[0]);
		return 1;
	}

	total_n = strtoull(argv[1], NULL, 10);
	if (total_n == 0) {
		fprintf(stderr, "Error: N must be > 0\n");
		return 1;
	}
	if (argc > 2) ptr_a = strtoull(argv[2], NULL, 16);
	if (argc > 3) ptr_b = strtoull(argv[3], NULL, 16);
	if (argc > 4) ptr_c = strtoull(argv[4], NULL, 16);

	/* ── 信号处理 ─────────────────────────────────── */
	signal(SIGINT,  sig_handler);
	signal(SIGTERM, sig_handler);

	/* ── libbpf 配置 ──────────────────────────────── */
	libbpf_set_print(libbpf_print_fn);

	/* ── 加载并验证 BPF skeleton ─────────────────── */
	skel = gpu_memcheck_bpf__open();
	if (!skel) {
		fprintf(stderr, "Failed to open BPF skeleton\n");
		return 1;
	}

	err = gpu_memcheck_bpf__load(skel);
	if (err) {
		fprintf(stderr, "Failed to load BPF skeleton: %d\n", err);
		goto cleanup;
	}

	err = gpu_memcheck_bpf__attach(skel);
	if (err) {
		fprintf(stderr, "Failed to attach BPF programs: %d\n", err);
		goto cleanup;
	}

	/* ── 向 bounds_config_map 写入边界配置 ─────────
	 *
	 * 此 map 类型为 GPU_ARRAY_HOST_MAP，数据存在 host 内存，
	 * GPU 侧可直接通过 UVA（Unified Virtual Addressing）访问，
	 * 无需触发 host call spinlock 协议。
	 */
	{
		struct bounds_config cfg = {
			.total_n   = total_n,
			.ptr_a     = ptr_a,
			.ptr_b     = ptr_b,
			.ptr_c     = ptr_c,
			.elem_size = sizeof(float),
			.enabled   = 1,
		};
		uint32_t key = 0;
		err = bpf_map_update_elem(
			bpf_map__fd(skel->maps.bounds_config_map),
			&key, &cfg, BPF_ANY);
		if (err) {
			fprintf(stderr,
				"Failed to set bounds config: %s\n",
				strerror(errno));
			goto cleanup;
		}
	}

	/* ── 查找 bpftime 提供的 GPU ring buffer poll 函数 ──
	 *
	 * bpftime_syscall_server__poll_gpu_ringbuf_map 是 bpftime
	 * syscall-server 动态库导出的函数，用于从 GPU_RINGBUF_MAP 中
	 * 读取事件并调用用户提供的回调。
	 *
	 * 签名：int fn(int mapfd, void *ctx,
	 *              void (*cb)(const void*, uint64_t, void*))
	 */
	int (*poll_fn)(int, void *,
		       void (*)(const void *, uint64_t, void *)) =
		dlsym(RTLD_DEFAULT,
		      "bpftime_syscall_server__poll_gpu_ringbuf_map");
	if (!poll_fn) {
		fprintf(stderr,
			"This tool must run under bpftime syscall-server!\n"
			"Use: LD_PRELOAD=...libbpftime-syscall-server.so %s N\n",
			argv[0]);
		err = -ENOENT;
		goto cleanup;
	}

	int violation_fd = bpf_map__fd(skel->maps.violation_events);
	int alloc_fd     = bpf_map__fd(skel->maps.alloc_tracker);

	/* ── 打印欢迎信息 ─────────────────────────────── */
	printf("╔══════════════════════════════════════════════════════════════╗\n");
	printf("║            GPU Memory Illegal Access Checker                 ║\n");
	printf("╠══════════════════════════════════════════════════════════════╣\n");
	printf("║  Array size N  : %-10" PRIu64
	       "  (valid index: 0 ~ %-10" PRIu64 ")║\n",
	       total_n, total_n - 1);
	printf("║  ptr_a         : 0x%-40" PRIx64 " ║\n", ptr_a);
	printf("║  Kernel target : _Z9vectorAddPKfS0_Pf                        ║\n");
	printf("║  Press Ctrl-C to stop and print summary                      ║\n");
	printf("╚══════════════════════════════════════════════════════════════╝\n\n");
	printf("Waiting for GPU kernel activity...\n\n");

	/* ── 主轮询循环 ───────────────────────────────── */
	int tick = 0;
	while (!exiting) {
		sleep(1);
		tick++;

		/* 轮询 GPU ring buffer，触发 poll_callback */
		err = poll_fn(violation_fd, NULL, poll_callback);
		if (err < 0 && err != -EINTR) {
			fprintf(stderr, "Ring buffer poll error: %d\n", err);
			break;
		}
		err = 0;

		/* 每 10 秒打印一次分配表 */
		if (tick % 10 == 0) {
			print_alloc_table(alloc_fd);
		}
	}

	/* ── 最终汇总 ─────────────────────────────────── */
	printf("\n╔══════════════════════════════════════════════════════════════╗\n");
	printf("║                    Detection Summary                         ║\n");
	printf("╠══════════════════════════════════════════════════════════════╣\n");
	printf("║  Total violations : %-40" PRIu64 " ║\n", total_violations);
	printf("║  Out-of-bounds    : %-40" PRIu64 " ║\n", oob_count);
	printf("║  Null pointer     : %-40" PRIu64 " ║\n", null_count);
	printf("╚══════════════════════════════════════════════════════════════╝\n\n");

	if (total_violations > 0) {
		print_alloc_table(alloc_fd);
		printf("WARNING: GPU memory violations detected!\n");
		printf("  Recommendation: check kernel launch parameters and add bounds guards.\n");
	} else {
		printf("No GPU memory violations detected.\n");
	}

cleanup:
	gpu_memcheck_bpf__destroy(skel);
	return err < 0 ? 1 : 0;
}
