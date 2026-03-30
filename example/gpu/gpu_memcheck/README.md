# gpu_memcheck — GPU 显存非法访问检测工具

## 概述

GPU kernel 内部的内存越界访问（out-of-bounds）是生产环境中最难调试的 bug 之一：
CUDA 不抛异常，程序看似正常运行，却悄悄写坏其他数据。

`gpu_memcheck` 基于 bpftime 的 GPU eBPF 能力，在 GPU kernel **执行时**实时检测线程索引越界，无需重编译 CUDA 程序，无需 Nsight/cuda-memcheck 的大幅性能开销。

```
╔═══════════════════════════════════════════════════════════════════════╗
║                      gpu_memcheck 检测流程                            ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  CPU 侧（uprobe）                GPU 侧（kprobe）                     ║
║  ─────────────────               ────────────────────────────         ║
║  cudaMalloc ─────────────────►  alloc_tracker (BPF_MAP_HASH)         ║
║  cudaFree   ─────────────────►  移出 alloc_tracker                   ║
║                                                                       ║
║  bounds_config_map ───────────► GPU kprobe 读取边界配置               ║
║  (GPU_ARRAY_HOST_MAP)            ↓                                    ║
║                              linear_tid = bid.x * bdim.x + tid.x     ║
║                              if linear_tid >= total_n:                ║
║                                  写入 violation_events（ringbuf）     ║
║                                  递增 violation_stats（per-thread）   ║
║                                                                       ║
║  用户态轮询 ◄──────────────── violation_events（实时告警打印）        ║
║             ◄──────────────── alloc_tracker（显存分配表）             ║
║             ◄──────────────── violation_stats（每线程越界次数）       ║
╚═══════════════════════════════════════════════════════════════════════╝
```

## 检测能力

| 违规类型 | 说明 | 检测方式 |
|----------|------|----------|
| **Out-of-Bounds** | 线程线性索引 ≥ 数组大小 N | GPU kprobe + bounds_config_map |
| **Null Pointer** | 传入 kernel 的指针为 null | GPU kprobe 检查 ptr_a/b/c |
| **分配追踪** | cudaMalloc/cudaFree 生命周期 | CPU uprobe + alloc_tracker |

### 与 cuda-memcheck 的对比

| 特性 | cuda-memcheck | gpu_memcheck (bpftime) |
|------|---------------|------------------------|
| 需重编译 | 否 | 否 |
| 性能开销 | 10-100x | < 5%（仅越界线程有开销） |
| 实时告警 | 否（事后分析） | 是（每次 kernel 执行时） |
| 生产环境 | 不适合 | 适合（always-on） |
| CPU-GPU 关联 | 否 | 是（CPU uprobe + GPU kprobe） |

## 关键技术点

### BPF Map 选型

```
bounds_config_map  → GPU_ARRAY_HOST_MAP (1513)
  • 数据在 Host 内存，GPU 通过 UVA 直接访问
  • 无需 spinlock host call，GPU 侧读取几乎零开销
  • host 侧可随时更新配置，下次 kernel 执行立即生效

violation_events   → GPU_RINGBUF_MAP (1527)
  • per-thread lock-free ring buffer，GPU 侧无争用
  • bpf_perf_event_output() 写入，host 侧 poll 读取
  • 实时上报：每个越界线程触发后立即通知 host

violation_stats    → PERGPUTD_ARRAY_HOST_MAP (1512)
  • 每个 GPU 线程有独立存储槽，计数无冲突
  • host 侧一次 bpf_map_lookup_elem 读取所有线程数据
  • 用于汇总"哪些线程越界最频繁"

alloc_tracker      → BPF_MAP_TYPE_HASH（标准 CPU map）
  • 由 uprobe(cudaMalloc/cudaFree) 维护
  • 可发现越界访问对应的合法分配大小
```

### 线性索引计算

```c
// GPU kprobe 内部（每个线程执行）
bpf_get_thread_idx(&tid_x, &tid_y, &tid_z);
bpf_get_block_idx(&bid_x, &bid_y, &bid_z);
bpf_get_block_dim(&bdim_x, &bdim_y, &bdim_z);

// 1D 展开（3D kernel 可按需扩展为完整 3D 展开）
linear_tid = bid_x * bdim_x + tid_x;

if (linear_tid >= cfg->total_n) {
    // 上报越界事件
    bpf_perf_event_output(ctx, &violation_events, 0, &evt, sizeof(evt));
}
```

## 构建

```bash
# 前置：bpftime 已用 CUDA 支持构建
cmake -Bbuild -DBPFTIME_ENABLE_CUDA_ATTACH=1 \
      -DBPFTIME_CUDA_ROOT=/usr/local/cuda .
cmake --build build -j$(nproc)

# 构建 gpu_memcheck
make -C example/gpu/gpu_memcheck
```

构建产物：
- `gpu_memcheck`    — eBPF 加载器（syscall-server 模式运行）
- `vec_add_buggy`   — 含故意 OOB 的 CUDA 测试程序

## 运行

### 快速开始（场景 1：经典越界）

**Terminal 1 — 启动检测器（server）**
```bash
BPFTIME_MAP_GPU_THREAD_COUNT=2048 \
BPFTIME_LOG_OUTPUT=console \
LD_PRELOAD=build/runtime/syscall-server/libbpftime-syscall-server.so \
  example/gpu/gpu_memcheck/gpu_memcheck 1024
```

参数 `1024` = 数组 N，合法访问范围为索引 `0 ~ 1023`。

**Terminal 2 — 运行有 bug 的 CUDA 程序（agent）**
```bash
BPFTIME_LOG_OUTPUT=console \
LD_PRELOAD=build/runtime/agent/libbpftime-agent.so \
  example/gpu/gpu_memcheck/vec_add_buggy 1
```

### 三种演示场景

| 场景 | 命令 | Bug 类型 |
|------|------|----------|
| 1（默认） | `./vec_add_buggy 1` | launch 1280 线程，N=1024，256 个线程越界 |
| 2 | `./vec_add_buggy 2` | grid-stride loop 使用 `<=` 而非 `<` |
| 3（对照） | `./vec_add_buggy 3` | 正确实现，应无告警 |

## 示例输出

### 检测器（Terminal 1）

```
╔══════════════════════════════════════════════════════════════╗
║            GPU Memory Illegal Access Checker                 ║
╠══════════════════════════════════════════════════════════════╣
║  Array size N  : 1024        (valid index: 0 ~ 1023      )  ║
║  ptr_a         : 0x0                                         ║
║  Kernel target : _Z9vectorAddPKfS0_Pf                        ║
║  Press Ctrl-C to stop and print summary                      ║
╚══════════════════════════════════════════════════════════════╝

Waiting for GPU kernel activity...

[10:23:45] *** GPU MEMORY VIOLATION #1 ***
  ├─ Type      : OUT-OF-BOUNDS
  ├─ LinearTID : 1024  (valid range: 0 ~ 1023)
  ├─ Overflow  : +1 elements beyond bound
  ├─ Thread    : (0, 0, 0)
  ├─ Block     : (4, 0, 0)
  └─ GPU Time  : 809635273301344 ns

[10:23:45] *** GPU MEMORY VIOLATION #2 ***
  ├─ Type      : OUT-OF-BOUNDS
  ├─ LinearTID : 1025  (valid range: 0 ~ 1023)
  ├─ Overflow  : +2 elements beyond bound
  ├─ Thread    : (1, 0, 0)
  ├─ Block     : (4, 0, 0)
  └─ GPU Time  : 809635273301456 ns

... (共 256 个越界线程)

┌─ GPU Memory Allocation Table ─────────────────────────────────┐
│  Address               Size (bytes)    Alloc Time (CPU ns)     │
│  ──────────────────    ────────────    ──────────────────────   │
│  0x7f3a00000000        4096            1748923456789012         │
│  0x7f3a00002000        4096            1748923456790034         │
│  0x7f3a00004000        4096            1748923456791056         │
│  Total: 3 active allocation(s)                                  │
└─────────────────────────────────────────────────────────────────┘

┌─ Per-Thread Violation Statistics ─────────────────────────────┐
│  Thread  │ OOB Count  │ Status                                │
│  ──────  │ ─────────  │ ──────                                │
│  0       │ 5          │ !!! OUT-OF-BOUNDS !!!                 │
│  1       │ 5          │ !!! OUT-OF-BOUNDS !!!                 │
...
│  Violating threads: 256   Total violation count: 1280         │
└───────────────────────────────────────────────────────────────┘
```

### 目标程序（Terminal 2）

```
======================================
 GPU Memory Bug Demo — Scenario 1
======================================
 Array size (N)    : 1024 elements (4096 bytes)
 Launch config     : 5 blocks × 256 threads = 1280 threads
 OOB threads       : 256 (index 1024 ~ 1279)
 Bug type          : Missing bounds guard
--------------------------------------
 gpu_memcheck cmd  : ./gpu_memcheck 1024

[iter  1] kernel done — 256 OOB threads!
[iter  2] kernel done — 256 OOB threads!
```

## 如何追踪自定义 kernel

1. 找到 kernel 的 C++ mangled name：
```bash
cuobjdump -symbols your_app | grep your_kernel_name
```

2. 修改 `gpu_memcheck.bpf.c` 中的 SEC 注解：
```c
// 将下面的符号替换为你的 kernel
SEC("kprobe/_Z9vectorAddPKfS0_Pf")
int gpu_memcheck_probe(void *ctx)
```

3. 重新构建：
```bash
make -C example/gpu/gpu_memcheck
```

4. 用实际数组大小启动检测器：
```bash
./gpu_memcheck <你的数组N>
```

## 文件结构

```
gpu_memcheck/
├── gpu_memcheck.bpf.c   # eBPF 程序（CPU uprobe + GPU kprobe）
├── gpu_memcheck.c        # 用户态加载器（ring buffer 轮询 + 告警打印）
├── vec_add_buggy.cu      # 含故意 OOB 的 CUDA 测试程序（3 个场景）
├── Makefile
└── README.md
```

## 局限性与扩展方向

**当前局限：**
- 仅支持 1D 线性索引的 OOB 检测（3D 需扩展展开公式）
- `total_n` 需手动配置；若要自动感知，需 hook `cudaLaunchKernel` 提取参数
- 仅检测 kernel 入口处的线程范围，不追踪 kernel 内部的动态指针运算

**可扩展的方向：**
- 结合 `__memcapture` 探针追踪 kernel 内的实际内存访问地址
- 通过 uprobe 拦截 `cudaLaunchKernel` 的 `void **args` 自动提取 N 和指针
- 添加 use-after-free 检测：访问 alloc_tracker 已删除的地址时告警
- 扩展到 ROCm（修改 SEC 符号 + 使用 ROCm attach 类型）

## 相关示例

- [mem_trace](../mem_trace/) — CUDA kernel 调用次数追踪
- [threadhist](../threadhist/) — 每线程执行次数直方图
- [kernelretsnoop](../kernelretsnoop/) — 精确线程退出时间戳
- [host_map_test](../host_map_test/) — GPU_ARRAY_HOST_MAP / PERGPUTD_ARRAY_HOST_MAP 用法
