/*
 * pvk_internal.h — opaque ctx layout.  NOT part of the public ABI.
 *
 * Python 永远看不见这个文件；ctypes 只操作 pvk_ctx* opaque handle。
 * 改这里的字段不升 PVK_ABI_VERSION。
 */
#ifndef PVK_INTERNAL_H
#define PVK_INTERNAL_H

#include "pvk/pvk.h"
#include "pvk/types.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward decls */
typedef struct {
    int          n_node;     /* 实际网格点数 */
    double*      x;          /* [n_node] cm */
    int*         node_layer; /* [n_node] layer index */
    double*      dx;         /* [n_node-1] 相邻节点间距 */

    /* ── 网格签名 ──
     * solve_dc 每次都调 build_mesh 会把网格和 field 缓存全部重建，从而
     * 丢掉上一次的解（warm start）。用这三个值判断 device 是否变化，
     * 没变就跳过重建。 */
    int          sig_valid;
    int          sig_n_layers;
    int          sig_nodes;
    double       sig_total_nm;
} pvk_mesh;

typedef struct {
    double*      fields[PVK_FIELD_COUNT];  /* 每个 field_id 一个 [n_node] 数组 */
    pvk_solve_info info;
} pvk_state;

/* Opaque ctx — 把外部传入的 device/optics/bias 全部存一份（不持有其原始内存）。 */
struct pvk_ctx {
    /* ── lifecycle ─────────────────────────────────────────── */
    pvk_log_fn   log_cb;
    void*        log_user;
    char         last_error[512];
    int          last_err_code;

    /* ── solver opts ────────────────────────────────────────── */
    pvk_solver_opts solver_opts;

    /* ── mesh + state ──────────────────────────────────────── */
    pvk_mesh   mesh;
    pvk_state  state;

    /* ── cached copies of last solve's inputs ───────────────── */
    pvk_device    last_device;
    pvk_optics_in last_optics;
    pvk_bias      last_bias;
    int           inputs_valid;   /* 1 = 已被 solve 使用过 */

    /* ── 上次收敛解的 warm-start 标记 ──
     * fields[PSI/PHIN/PHIP] 在两次 solve_dc 之间保留，作为下一次
     * Newton 的初值；last_V_applied 用于按 ΔV 做线性外推。 */
    int           have_solution;
    double        last_V_applied;
    double        last_J;         /* 上次收敛点的 J (mA/cm²) */
};

/* ----------------------------------------------------------------- */
/* 内部 util —— 避免 stdlib.h 直接污染 public API 头 */
/* ----------------------------------------------------------------- */
void* pvk__malloc(size_t n);
void  pvk__free(void* p);
char* pvk__strdup(const char* s);

/* 向 last_error 写入一条格式化消息（简单 snprintf 封装） */
void  pvk__set_error(pvk_ctx* ctx, int code, const char* fmt, ...);

/* 日志 */
void  pvk__log(pvk_ctx* ctx, int level, const char* msg);

/* 给某个 field_id 分配 [n_node] 的零初始化数组，已存在则先 free */
int   pvk__ensure_field_buf(pvk_ctx* ctx, int field_id);

/* 释放缓存的上一次 device 副本（含每层 strdup 的 name/material）。
 * 注意：last_optics 里的 wl_nm/flux 是"借用"调用方的指针，不在此释放。 */
void  pvk__release_cached_device(pvk_ctx* ctx);

#ifdef __cplusplus
}
#endif

#endif /* PVK_INTERNAL_H */
