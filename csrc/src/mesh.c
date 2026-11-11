/*
 * mesh.c — 1D structured mesh.
 *
 * v0.1: 每层均匀分点，层厚按 target_nodes 分摊，最后在
 *       interface_refine > 0 时在每个界面额外插几个点。
 * 后续: 2D 有限元 / 非均匀网格
 *
 * 生成结果:
 *   x[]       — [n_node] 坐标 (cm)，从 device 最顶层 (anode 侧) 起
 *   node_layer[] — [n_node] 每个节点属于哪个 layer index
 *   dx[]      — [n_node-1] 相邻节点间距
 */

#include "pvk_internal.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

PVK_API int pvk_build_mesh(pvk_ctx* ctx, const pvk_device* dev) {
    if (!ctx || !dev || dev->n_layers <= 0 || dev->layers == NULL) {
        return PVK_EINVAL;
    }

    int target = dev->nodes >= 3 ? dev->nodes : 101;

    /* 总厚度 cm */
    double total_cm = 0.0;
    for (int i = 0; i < dev->n_layers; i++) {
        total_cm += dev->layers[i].thick_nm * 1e-7;
    }
    if (total_cm <= 0) {
        pvk__set_error(ctx, PVK_EINVAL, "device total thickness = %g", total_cm);
        return PVK_EINVAL;
    }

    /* 每层分到的节点：按层厚比例分配，每层至少 2 个（首尾节点） */
    double* frac = (double*)pvk__malloc(sizeof(double) * dev->n_layers);
    if (!frac) return PVK_EMEM;
    for (int i = 0; i < dev->n_layers; i++) {
        frac[i] = dev->layers[i].thick_nm * 1e-7 / total_cm;
    }

    /* 第一层需要一个起点节点，之后每层只需 (n-1) 个 */
    int remaining = target;
    int* per_layer = (int*)pvk__malloc(sizeof(int) * dev->n_layers);
    if (!per_layer) { pvk__free(frac); return PVK_EMEM; }

    /* 先按比例算浮点，然后收尾数到至少 2 */
    for (int i = 0; i < dev->n_layers; i++) {
        double want = frac[i] * (double)target;
        int n = (int)round(want);
        if (n < 2) n = 2;
        per_layer[i] = n;
        remaining -= n;
    }
    /* 把剩余节点塞到最厚层 */
    if (remaining != 0) {
        int thickest = 0;
        for (int i = 1; i < dev->n_layers; i++) {
            if (dev->layers[i].thick_nm > dev->layers[thickest].thick_nm) thickest = i;
        }
        per_layer[thickest] += remaining;
        if (per_layer[thickest] < 2) per_layer[thickest] = 2;
    }

    /* 实际节点数 = 每层点数之和 - (n_layers-1) 个界面共享节点 */
    int n_node = 1;
    for (int i = 0; i < dev->n_layers; i++) {
        n_node += per_layer[i] - 1;
    }
    if (n_node < 3) {
        pvk__set_error(ctx, PVK_EINVAL, "mesh has only %d nodes after allocation", n_node);
        pvk__free(frac); pvk__free(per_layer);
        return PVK_EINVAL;
    }

    /* 分配并填 x[] 和 node_layer[] */
    double* x = (double*)pvk__malloc(sizeof(double) * n_node);
    int*    nl = (int*)pvk__malloc(sizeof(int) * n_node);
    if (!x || !nl) {
        pvk__free(frac); pvk__free(per_layer); pvk__free(x); pvk__free(nl);
        return PVK_EMEM;
    }

    double cur = 0.0;
    int idx = 0;
    for (int li = 0; li < dev->n_layers; li++) {
        double thick = dev->layers[li].thick_nm * 1e-7;
        int n = per_layer[li];
        /* 第 0 个节点已由上一层的末节点覆盖（除了第一层） */
        int start_idx = (li == 0) ? 0 : 1;
        double seg_start = (li == 0) ? 0.0 : cur;
        for (int k = start_idx; k < n; k++) {
            /* 每层内均匀 */
            double local = seg_start + thick * (double)k / (double)(n - 1);
            x[idx] = local;
            nl[idx] = li;
            idx++;
        }
        cur += thick;
    }

    /* dx[] */
    double* dx = (double*)pvk__malloc(sizeof(double) * (n_node - 1));
    for (int i = 0; i < n_node - 1; i++) {
        dx[i] = x[i + 1] - x[i];
        if (dx[i] <= 0) {
            pvk__set_error(ctx, PVK_EMATH, "non-monotonic mesh at i=%d dx=%g", i, dx[i]);
            pvk__free(frac); pvk__free(per_layer); pvk__free(x); pvk__free(nl); pvk__free(dx);
            return PVK_EMATH;
        }
    }

    /* 释放旧 mesh，替换新 mesh */
    pvk__free(ctx->mesh.x);
    pvk__free(ctx->mesh.node_layer);
    pvk__free(ctx->mesh.dx);
    ctx->mesh.n_node     = n_node;
    ctx->mesh.x          = x;
    ctx->mesh.node_layer = nl;
    ctx->mesh.dx         = dx;

    /* field 缓存也要 resize */
    for (int fid = 0; fid < PVK_FIELD_COUNT; fid++) {
        pvk__ensure_field_buf(ctx, fid);
    }

    /* 记录签名 —— solve_dc 靠它判断能否沿用当前网格（保住 warm start） */
    ctx->mesh.sig_valid    = 1;
    ctx->mesh.sig_n_layers = dev->n_layers;
    ctx->mesh.sig_nodes    = dev->nodes;
    ctx->mesh.sig_total_nm = total_cm * 1e7;
    ctx->have_solution     = 0;   /* 网格换了，旧解作废 */

    pvk__free(frac);
    pvk__free(per_layer);

    pvk__log(ctx, PVK_LOG_INFO, "pvk_build_mesh: mesh built");
    return PVK_OK;
}

PVK_API int pvk_node_count(pvk_ctx* ctx) {
    return ctx ? ctx->mesh.n_node : 0;
}

PVK_API int pvk_get_x(pvk_ctx* ctx, double* out, int n) {
    if (!ctx || !out) return PVK_EINVAL;
    int have = ctx->mesh.n_node;
    if (n < have) return PVK_EINVAL;
    if (ctx->mesh.x) {
        memcpy(out, ctx->mesh.x, sizeof(double) * have);
    }
    return PVK_OK;
}
