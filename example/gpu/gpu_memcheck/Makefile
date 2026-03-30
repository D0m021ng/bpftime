# SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
#
# gpu_memcheck Makefile
# 模仿 example/gpu/kernelretsnoop/Makefile 结构

OUTPUT := .output
CLANG  ?= clang
CC     ?= gcc
NVCC   ?= nvcc

LIBBPF_SRC    := $(abspath ../../../third_party/libbpf/src)
BPFTOOL_SRC   := $(abspath ../../../third_party/bpftool/src)
LIBBPF_OBJ    := $(abspath $(OUTPUT)/libbpf.a)
BPFTOOL_OUTPUT ?= $(abspath $(OUTPUT)/bpftool)
BPFTOOL       ?= $(BPFTOOL_OUTPUT)/bootstrap/bpftool

ARCH ?= $(shell uname -m | sed 's/x86_64/x86/' \
         | sed 's/arm.*/arm/' \
         | sed 's/aarch64/arm64/' \
         | sed 's/ppc64le/powerpc/' \
         | sed 's/mips.*/mips/' \
         | sed 's/riscv64/riscv/' \
         | sed 's/loongarch64/loongarch/')

VMLINUX := ../../../third_party/vmlinux/$(ARCH)/vmlinux.h

INCLUDES    := -I$(OUTPUT) \
               -I../../../third_party/libbpf/include/uapi \
               -I$(dir $(VMLINUX))
CFLAGS      := -g -Wall
ALL_LDFLAGS := $(LDFLAGS) $(EXTRA_LDFLAGS)

# clang 系统头（bpf target 编译时需要）
CLANG_BPF_SYS_INCLUDES ?= $(shell $(CLANG) -v -E - </dev/null 2>&1 \
	| sed -n '/<...> search starts here:/,/End of search list./{ s| \(/.*\)|-idirafter \1|p }')

# ─────────────────────────────────────────────────────
# 静默模式
# ─────────────────────────────────────────────────────
ifeq ($(V),1)
  Q =
  msg =
else
  Q = @
  msg = @printf '  %-8s %s%s\n' \
        "$(1)" \
        "$(patsubst $(abspath $(OUTPUT))/%,%,$(2))" \
        "$(if $(3), $(3))";
  MAKEFLAGS += --no-print-directory
endif

define allow-override
  $(if $(or $(findstring environment,$(origin $(1))),\
            $(findstring command line,$(origin $(1)))),,\
    $(eval $(1) = $(2)))
endef

$(call allow-override,CC,$(CROSS_COMPILE)cc)
$(call allow-override,LD,$(CROSS_COMPILE)ld)

# ─────────────────────────────────────────────────────
# 顶层目标
# ─────────────────────────────────────────────────────
.PHONY: all clean

all: gpu_memcheck vec_add_buggy

# ─────────────────────────────────────────────────────
# CUDA 目标程序（含故意越界的测试程序）
# ─────────────────────────────────────────────────────
vec_add_buggy: vec_add_buggy.cu
	@if command -v nvcc >/dev/null 2>&1; then \
		echo "  NVCC    vec_add_buggy"; \
		nvcc -cudart shared vec_add_buggy.cu -o vec_add_buggy -g -O0; \
	else \
		echo "Warning: nvcc not found, skipping vec_add_buggy build"; \
	fi

# ─────────────────────────────────────────────────────
# 构建目录
# ─────────────────────────────────────────────────────
$(OUTPUT) $(OUTPUT)/libbpf $(BPFTOOL_OUTPUT):
	$(call msg,MKDIR,$@)
	$(Q)mkdir -p $@

# ─────────────────────────────────────────────────────
# libbpf 静态库
# ─────────────────────────────────────────────────────
$(LIBBPF_OBJ): $(wildcard $(LIBBPF_SRC)/*.[ch] $(LIBBPF_SRC)/Makefile) \
               | $(OUTPUT)/libbpf
	$(call msg,LIB,$@)
	$(Q)$(MAKE) -C $(LIBBPF_SRC) BUILD_STATIC_ONLY=1 \
	    OBJDIR=$(dir $@)/libbpf DESTDIR=$(dir $@) \
	    INCLUDEDIR= LIBDIR= UAPIDIR= \
	    install

# ─────────────────────────────────────────────────────
# bpftool（用于生成 vmlinux.h 和 skeleton）
# ─────────────────────────────────────────────────────
$(BPFTOOL): | $(BPFTOOL_OUTPUT)
	$(call msg,BPFTOOL,$@)
	$(Q)$(MAKE) ARCH= CROSS_COMPILE= \
	    OUTPUT=$(BPFTOOL_OUTPUT)/ \
	    -C $(BPFTOOL_SRC) bootstrap

# ─────────────────────────────────────────────────────
# BPF 程序编译（.bpf.c → .bpf.o）
#
# 步骤：
#  1. clang -target bpf 编译到 .tmp.bpf.o
#  2. bpftool gen object 做 BTF 校验并输出最终 .bpf.o
# ─────────────────────────────────────────────────────
$(OUTPUT)/%.bpf.o: %.bpf.c $(LIBBPF_OBJ) $(wildcard %.h) $(VMLINUX) \
                   | $(OUTPUT) $(BPFTOOL)
	$(call msg,BPF,$@)
	$(Q)$(CLANG) -Xlinker --export-dynamic \
	     -g -O2 -target bpf -D__TARGET_ARCH_$(ARCH) \
	     $(INCLUDES) $(CLANG_BPF_SYS_INCLUDES) \
	     -c $(filter %.c,$^) \
	     -o $(patsubst %.bpf.o,%.tmp.bpf.o,$@)
	$(Q)$(BPFTOOL) gen object $@ \
	     $(patsubst %.bpf.o,%.tmp.bpf.o,$@)

# ─────────────────────────────────────────────────────
# BPF skeleton 生成（.bpf.o → .skel.h）
# ─────────────────────────────────────────────────────
$(OUTPUT)/%.skel.h: $(OUTPUT)/%.bpf.o | $(OUTPUT) $(BPFTOOL)
	$(call msg,GEN-SKEL,$@)
	$(Q)$(BPFTOOL) gen skeleton $< > $@

# ─────────────────────────────────────────────────────
# 用户态程序：目标文件（.c → .o）
# ─────────────────────────────────────────────────────
$(OUTPUT)/gpu_memcheck.o: gpu_memcheck.c \
                           $(OUTPUT)/gpu_memcheck.skel.h | $(OUTPUT)
	$(call msg,CC,$@)
	$(Q)$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# ─────────────────────────────────────────────────────
# 最终二进制链接
# ─────────────────────────────────────────────────────
gpu_memcheck: $(OUTPUT)/gpu_memcheck.o $(LIBBPF_OBJ) | $(OUTPUT)
	$(call msg,BINARY,$@)
	$(Q)$(CC) $(CFLAGS) $^ $(ALL_LDFLAGS) -lelf -lz -ldl -o $@

# ─────────────────────────────────────────────────────
# 清理
# ─────────────────────────────────────────────────────
clean:
	$(call msg,CLEAN)
	$(Q)rm -rf $(OUTPUT) gpu_memcheck vec_add_buggy

.DELETE_ON_ERROR:
.SECONDARY:
