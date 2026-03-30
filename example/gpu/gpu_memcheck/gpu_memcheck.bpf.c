// SPDX-License-Identifier: GPL-2.0
/*
 * gpu_memcheck.bpf.c - GPU 显存非法访问检测工具
 *
 * 架构：
 *
 *  ┌─────────────────────────────────────────────────────────────────┐
 *  │                      Detection Pipeline                          │
 *  │                                                                  │
 *  │  CPU side (uprobe)          GPU side (kprobe / kretprobe)       │
 *  │  ─────────────────          ──────────────────────────────       │
 *  │  cudaMalloc ──────────────► alloc_tracker (BPF_MAP_HASH)        │
 *  │  cudaFree   ──────────────► remove from alloc_tracker           │
 *  │                                                                  │
 *  │  bounds_config_map ────────► GPU kprobe reads bounds            │
 *  │  (GPU_ARRAY_HOST_MAP)        ↓                                  │
 *  │                          linear_tid = bid.x*bdim.x + tid.x      │
 *  │                          if linear_tid >= total_n:               │
 *  │                              write → violation_events            │
 *  │                              write → violation_stats (per-thd)  │
 *  │                                                                  │
 *  │  userspace polls ◄──── violation_events (GPU_RINGBUF_MAP)       │
 *  │                  ◄──── alloc_tracker (shows leaked/freed ptrs)  │
 *  └─────────────────────────────────────────────────────────────────┘
 *
 * Map 类型说明：
 *   bounds_config_map  : GPU_ARRAY_HOST_MAP (1513) — host写、GPU读，无需 host call
 *   violation_events   : GPU_RINGBUF_MAP    (1527) — GPU写、host via poll_gpu_ringbuf 读
 *   violation_stats    : PERGPUTD_ARRAY_HOST_MAP (1512) — 每线程统计
 *   alloc_tracker      : BPF_MAP_TYPE_HASH — CPU侧跟踪 cudaMalloc/Free
 *   pending_alloc      : BPF_MAP_TYPE_HASH — 暂存 cudaMalloc 入参
 */

#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

/* bpftime 自定义 map 类型 ID */
#define BPF_MAP_TYPE_GPU_RINGBUF_MAP          1527
#define BPF_MAP_TYPE_GPU_ARRAY_HOST_MAP       1513   /* 非 per-thread 共享，host 内存 */
#define BPF_MAP_TYPE_PERGPUTD_ARRAY_HOST_MAP  1512   /* per-thread，host 内存 */

/* 非法访问类型 */
#define VIOLATION_OOB       0   /* Out-of-bounds: thread index >= total_n */
#define VIOLATION_NULL_A    1   /* Array A pointer is null */
#define VIOLATION_NULL_B    2   /* Array B pointer is null */
#define VIOLATION_NULL_C    3   /* Array C pointer is null */

/* GPU 专用 helper 函数（通过 function pointer call 触发 trampoline） */
static const void (*ebpf_puts)(const char *) = (void *)501;
static const u64  (*bpf_get_globaltimer)(void) = (void *)502;
static const u64  (*bpf_get_block_idx)(u64 *x, u64 *y, u64 *z) = (void *)503;
static const u64  (*bpf_get_block_dim)(u64 *x, u64 *y, u64 *z) = (void *)504;
static const u64  (*bpf_get_thread_idx)(u64 *x, u64 *y, u64 *z) = (void *)505;
static const void (*bpf_gpu_membar)(void) = (void *)506;

/* =====================================================================
 * 数据结构定义
 * ===================================================================== */

/*
 * 显存范围配置（由 host 写入，GPU 侧 kprobe 读取）
 *
 * 使用 GPU_ARRAY_HOST_MAP 使 GPU 能无需 host call 直接读取此配置，
 * 避免每次 kernel 调用都触发 spinlock 协议的性能开销。
 */
struct bounds_config {
	u64 total_n;     /* 数组元素个数（合法访问范围: [0, total_n) */
	u64 ptr_a;       /* array A 的 GPU 基地址（用于 null 检测） */
	u64 ptr_b;       /* array B 的 GPU 基地址 */
	u64 ptr_c;       /* array C 的 GPU 基地址 */
	u32 elem_size;   /* 元素大小（字节），用于计算字节偏移量 */
	u32 enabled;     /* 检测开关：0=禁用, 1=启用 */
};

/*
 * 违规事件（从 GPU 侧发送到 host）
 *
 * 通过 GPU_RINGBUF_MAP + bpf_perf_event_output 从 GPU 线程内异步上报。
 */
struct violation_event {
	u64 timestamp_ns;     /* GPU 全局时钟纳秒时间戳 */
	u64 linear_tid;       /* 线程线性索引 = bid.x * bdim.x + tid.x */
	u64 thread_x;         /* threadIdx.x */
	u64 thread_y;         /* threadIdx.y */
	u64 thread_z;         /* threadIdx.z */
	u64 block_x;          /* blockIdx.x */
	u64 block_y;          /* blockIdx.y */
	u64 block_z;          /* blockIdx.z */
	u64 expected_max;     /* 合法最大索引值 (= total_n) */
	u64 overflow_by;      /* 越界了多少（linear_tid - total_n） */
	u32 violation_type;   /* VIOLATION_* 常量 */
	u32 reserved;
};

/*
 * CPU 侧 cudaMalloc 分配信息
 */
struct alloc_info {
	u64 ptr;       /* GPU 设备指针 */
	u64 size;      /* 申请字节数 */
	u64 alloc_ts;  /* 分配时间（CPU MONOTONIC ns） */
};

/*
 * CPU 侧 cudaMalloc 入口时暂存的 (void**) 参数
 */
struct pending_malloc {
	u64 ptr_ptr;  /* 指向 (void*) 的地址，用于 uretprobe 时读取分配结果 */
	u64 size;     /* 请求字节数 */
};

/* =====================================================================
 * BPF Maps
 * ===================================================================== */

/*
 * Map 1：显存边界配置（host 写 → GPU 读）
 *
 * 类型：GPU_ARRAY_HOST_MAP (1513) — 数据存在 host 内存，GPU 通过 UVA 零拷贝访问。
 * max_entries=1，只有一个配置项（key=0）。
 */
struct {
	__uint(type, BPF_MAP_TYPE_GPU_ARRAY_HOST_MAP);
	__uint(max_entries, 1);
	__type(key, u32);
	__type(value, struct bounds_config);
} bounds_config_map SEC(".maps");

/*
 * Map 2：违规事件 ring buffer（GPU 写 → host 读）
 *
 * 类型：GPU_RINGBUF_MAP (1527) — per-thread lock-free ring buffer。
 * max_entries=16：支持最多 16 个并发线程的 ring buffer slot。
 *
 * GPU 侧通过 bpf_perf_event_output() 写入；
 * host 侧通过 bpftime_syscall_server__poll_gpu_ringbuf_map() 轮询读取。
 */
struct {
	__uint(type, BPF_MAP_TYPE_GPU_RINGBUF_MAP);
	__uint(max_entries, 16);
	__type(key, u32);
	__type(value, struct violation_event);
} violation_events SEC(".maps");

/*
 * violation_stats（PERGPUTD_ARRAY_HOST_MAP）已移除：
 * 线程槽由 BPFTIME_MAP_GPU_THREAD_COUNT 控制，超出时越界写共享内存
 * 导致进程退出崩溃。统计改由用户态累计 ring buffer 事件数实现。
 */

/*
 * Map 4：CPU 侧 GPU 分配追踪（纯 CPU 侧读写）
 *
 * key = GPU 设备指针；value = struct alloc_info。
 * 由 uprobe(cudaMalloc)/uprobe(cudaFree) 维护，可发现：
 *   - 越界访问对应的分配大小
 *   - 访问已释放内存（use-after-free）
 */
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 1024);
	__type(key, u64);
	__type(value, struct alloc_info);
} alloc_tracker SEC(".maps");

/*
 * Map 5：cudaMalloc 入口参数暂存（CPU 侧）
 *
 * key = PID (u32)；value = struct pending_malloc。
 * 在 uprobe(cudaMalloc) 入口保存 (void**) 参数，
 * 在 uretprobe(cudaMalloc) 返回时读取实际分配地址。
 */
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 256);
	__type(key, u32);
	__type(value, struct pending_malloc);
} pending_alloc SEC(".maps");

/* =====================================================================
 * GPU 侧探针：在 kernel 入口检测越界线程
 * ===================================================================== */

/*
 * 共享检测逻辑：每个 GPU 线程在 kernel 入口执行。
 *  1. 读取 bounds_config_map（UVA 直接访问 host 内存，无需 host call）
 *  2. 计算线性索引：linear_tid = blockIdx.x * blockDim.x + threadIdx.x
 *  3. 若 linear_tid >= total_n：向 violation_events 写入事件，递增 violation_stats
 *
 * 以 static __always_inline 形式定义，被下面三个 kprobe 共享，
 * 分别挂载到 vec_add_buggy.cu 中的三个不同 kernel 函数。
 *
 * 注意：SEC 中的符号名为 C++ mangled name，使用以下命令查找目标 kernel：
 *   cuobjdump -symbols your_app | grep your_kernel_name
 */
static __always_inline int oob_check(void *ctx)
{
	u32 zero = 0;
	struct bounds_config *cfg;
	u64 tid_x, tid_y, tid_z;
	u64 bid_x, bid_y, bid_z;
	u64 bdim_x, bdim_y, bdim_z;
	u64 linear_tid;
	struct violation_event evt = {};

	/* 读取边界配置；若未启用则直接返回 */
	cfg = bpf_map_lookup_elem(&bounds_config_map, &zero);
	if (!cfg || !cfg->enabled || cfg->total_n == 0)
		return 0;

	/* 获取当前线程和 block 的坐标 */
	bpf_get_thread_idx(&tid_x, &tid_y, &tid_z);
	bpf_get_block_idx(&bid_x, &bid_y, &bid_z);
	bpf_get_block_dim(&bdim_x, &bdim_y, &bdim_z);

	/* 1D 线性索引：linear_tid = blockIdx.x * blockDim.x + threadIdx.x */
	linear_tid = bid_x * bdim_x + tid_x;

	/* 检查 OOB：线程索引是否超过数组边界 */
	if (linear_tid >= cfg->total_n) {
		evt.timestamp_ns   = bpf_get_globaltimer();
		evt.linear_tid     = linear_tid;
		evt.thread_x       = tid_x;
		evt.thread_y       = tid_y;
		evt.thread_z       = tid_z;
		evt.block_x        = bid_x;
		evt.block_y        = bid_y;
		evt.block_z        = bid_z;
		evt.expected_max   = cfg->total_n;
		evt.overflow_by    = linear_tid - cfg->total_n;
		evt.violation_type = VIOLATION_OOB;

		/* 上报事件到 ring buffer，供 host 侧实时打印 */
		bpf_perf_event_output(ctx, &violation_events, 0,
				      &evt, sizeof(evt));
	}

	return 0;
}

/*
 * 场景 1：vectorAdd(const float*, const float*, float*)
 * mangled: _Z9vectorAddPKfS0_Pf
 */
SEC("kprobe/_Z9vectorAddPKfS0_Pf")
int cuda__gpu_memcheck_probe(void *ctx)
{
	return oob_check(ctx);
}

/*
 * 场景 2：vectorAdd_OOB_stride(const float*, const float*, float*, int)
 * mangled: _Z20vectorAdd_OOB_stridePKfS0_Pfi
 */
SEC("kprobe/_Z20vectorAdd_OOB_stridePKfS0_Pfi")
int cuda__gpu_memcheck_stride_probe(void *ctx)
{
	return oob_check(ctx);
}

/*
 * 场景 3：vectorAdd_SAFE(const float*, const float*, float*, int)
 * mangled: _Z14vectorAdd_SAFEPKfS0_Pfi
 * 正确实现，预期无越界，用于对比验证。
 */
SEC("kprobe/_Z14vectorAdd_SAFEPKfS0_Pfi")
int cuda__gpu_memcheck_safe_probe(void *ctx)
{
	return oob_check(ctx);
}

/* =====================================================================
 * CPU 侧探针：追踪 cudaMalloc / cudaFree 分配生命周期
 * ===================================================================== */

/*
 * uprobe：cudaMalloc 入口
 *
 * 函数签名：cudaError_t cudaMalloc(void **devPtr, size_t size)
 *   - PARM1 (rdi) = devPtr（二级指针，指向将被写入 GPU 地址的 void*）
 *   - PARM2 (rsi) = size（请求字节数）
 *
 * 在入口保存 (devPtr, size)，在 uretprobe 时读取实际分配地址。
 */
SEC("uprobe/libcudart.so.12:cudaMalloc")
int trace_cudaMalloc_entry(struct pt_regs *ctx)
{
	u32 pid = (u32)(bpf_get_current_pid_tgid() >> 32);
	struct pending_malloc pm = {
		.ptr_ptr = PT_REGS_PARM1(ctx),
		.size    = PT_REGS_PARM2(ctx),
	};
	bpf_map_update_elem(&pending_alloc, &pid, &pm, BPF_ANY);
	return 0;
}

/*
 * uretprobe：cudaMalloc 返回
 *
 * 从 pending_alloc 取出保存的 (void**) 地址，读取实际分配到的 GPU 指针，
 * 记录到 alloc_tracker。
 */
SEC("uretprobe/libcudart.so.12:cudaMalloc")
int trace_cudaMalloc_return(struct pt_regs *ctx)
{
	u32 pid = (u32)(bpf_get_current_pid_tgid() >> 32);
	struct pending_malloc *pm;
	u64 gpu_ptr = 0;

	pm = bpf_map_lookup_elem(&pending_alloc, &pid);
	if (!pm)
		return 0;

	/* bpf_probe_read 在 bpftime 用户态运行时中对 user/kernel 均可用 */
	bpf_probe_read(&gpu_ptr, sizeof(gpu_ptr), (void *)pm->ptr_ptr);

	if (gpu_ptr != 0) {
		struct alloc_info info = {
			.ptr      = gpu_ptr,
			.size     = pm->size,
			.alloc_ts = bpf_ktime_get_ns(),
		};
		bpf_map_update_elem(&alloc_tracker, &gpu_ptr, &info, BPF_ANY);
	}

	bpf_map_delete_elem(&pending_alloc, &pid);
	return 0;
}

/*
 * uprobe：cudaFree
 *
 * 函数签名：cudaError_t cudaFree(void *devPtr)
 *   - PARM1 (rdi) = devPtr（将被释放的 GPU 指针）
 *
 * 从 alloc_tracker 中删除对应条目，后续访问此地址将被视为 use-after-free。
 */
SEC("uprobe/libcudart.so.12:cudaFree")
int trace_cudaFree(struct pt_regs *ctx)
{
	u64 gpu_ptr = PT_REGS_PARM1(ctx);

	if (gpu_ptr != 0)
		bpf_map_delete_elem(&alloc_tracker, &gpu_ptr);
	return 0;
}

char LICENSE[] SEC("license") = "GPL";
