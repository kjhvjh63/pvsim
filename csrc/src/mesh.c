/*
 * mesh.c — 1D structured mesh.
 *
 * 层间节点按层厚比例分摊（与 interface_refine=0 时的均匀网格一致）；
 * interface_refine > 0 时，层内改用「两翼几何级数 + 中间均匀」的分布，
 * 让界面/接触附近的网格加密到亚纳米，用来解析电极旁的积累层。
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

/* 单侧几何级数的最大区间数（refine 合法范围 [0,2] 时实际 ≤ 12） */
#define PVK_REFINE_MAX_SIDE 32

/* 层内节点分布。
 *
 * m     — 该层按均匀分配应有多少个区间（节点数 = m+1）
 * t     — 层厚 (cm)
 * refine— 界面加密比例，<=0 时退化为均匀
 * pos   — 输出 m' + 1 个层内坐标（可为 NULL，此时只返回区间数 m'）
 *
 * 加密方式：两侧各取 ns = ceil(6*refine) 个区间，间距按几何级数
 * h0, h0*q, … 递增（最靠界面的一格最密），h0 = h/R，R = 1+24*refine，
 * q 取到最后一个间距正好回到均匀值 h。中间剩余厚度按 h 均匀铺。
 * 若两翼长度之和已超过层厚，则整体按 t/(2D) 压缩，整层变成对称渐变。
 * 这样界面首格尺寸只由层的均匀间距和 refine 决定，与总节点数解耦；
 * refine=1 时首格约为均匀间距的 1/25。 */
static int layer_grade(double t, int m, double refine, double* pos) {
    if (m < 1) {
        if (pos) pos[0] = 0.0;
        return 1;
    }
    double h = t / (double)m;
    if (!(refine > 0.0) || m < 2) {
        if (pos) {
            for (int k = 0; k <= m; k++) pos[k] = t * (double)k / (double)m;
        }
        return m;
    }

    double h0 = h / (1.0 + 24.0 * refine);
    if (h0 < 1e-8) h0 = 1e-8;          /* 0.1 nm 下限，避免亚原子尺度 */
    if (h0 >= h) {                     /* 基础网格已经比目标更密：保持均匀 */
        if (pos) {
            for (int k = 0; k <= m; k++) pos[k] = t * (double)k / (double)m;
        }
        return m;
    }

    int ns = (int)ceil(6.0 * refine);
    if (ns < 1) ns = 1;
    if (ns > PVK_REFINE_MAX_SIDE) ns = PVK_REFINE_MAX_SIDE;
    if (ns > m + 8) ns = m + 8;

    double q = (ns > 1) ? pow(h / h0, 1.0 / (double)(ns - 1)) : 1.0;
    double s[PVK_REFINE_MAX_SIDE];
    double D = 0.0;
    for (int i = 0; i < ns; i++) {
        s[i] = h0 * pow(q, (double)i);
        D += s[i];
    }

    int total;
    double a = 1.0;      /* 两翼相遇时的整体压缩因子 */
    double hmid = 0.0;   /* 中间均匀段间距 */
    int kmid = 0;
    if (2.0 * D >= t) {
        a = t / (2.0 * D);
        total = 2 * ns;
    } else {
        double mid = t - 2.0 * D;
        kmid = (int)lround(mid / h);
        if (kmid < 1) kmid = 1;
        hmid = mid / (double)kmid;
        total = 2 * ns + kmid;
    }

    if (!pos) return total;

    int k = 0;
    double x = 0.0;
    pos[k++] = 0.0;
    for (int i = 0; i < ns; i++) { x += s[i] * a; pos[k++] = x; }
    for (int i = 0; i < kmid; i++) { x += hmid; pos[k++] = x; }
    for (int i = ns - 1; i >= 0; i--) { x += s[i] * a; pos[k++] = x; }

    /* 收尾：舍入可能让末点偏离层厚，整体缩放保证精确闭合 */
    if (x > 0.0 && fabs(x - t) > 1e-15) {
        double f = t / x;
        for (int i = 1; i <= total; i++) pos[i] *= f;
    } else {
        pos[total] = t;
    }
    return total;
}

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

    /* 每层的实际区间数：均匀分配得到 m = per_layer-1，再按 interface_refine
     * 在层内重新分布（节点数随之增加，界面上首格变成亚纳米） */
    double refine = dev->interface_refine;
    int*    n_int = (int*)pvk__malloc(sizeof(int) * dev->n_layers);
    if (!n_int) { pvk__free(frac); pvk__free(per_layer); return PVK_EMEM; }

    int max_int = 0;
    int n_node  = 1;
    for (int i = 0; i < dev->n_layers; i++) {
        int m = per_layer[i] - 1;
        int mi = layer_grade(dev->layers[i].thick_nm * 1e-7, m, refine, NULL);
        n_int[i] = mi;
        n_node  += mi;
        if (mi > max_int) max_int = mi;
    }
    if (n_node < 3) {
        pvk__set_error(ctx, PVK_EINVAL, "mesh has only %d nodes after allocation", n_node);
        pvk__free(frac); pvk__free(per_layer); pvk__free(n_int);
        return PVK_EINVAL;
    }

    /* 分配并填 x[] 和 node_layer[] */
    double* x   = (double*)pvk__malloc(sizeof(double) * n_node);
    int*    nl  = (int*)pvk__malloc(sizeof(int) * n_node);
    double* lp  = (double*)pvk__malloc(sizeof(double) * (max_int + 1));
    if (!x || !nl || !lp) {
        pvk__free(frac); pvk__free(per_layer); pvk__free(n_int);
        pvk__free(x); pvk__free(nl); pvk__free(lp);
        return PVK_EMEM;
    }

    double cur = 0.0;
    int idx = 0;
    for (int li = 0; li < dev->n_layers; li++) {
        double thick = dev->layers[li].thick_nm * 1e-7;
        layer_grade(thick, per_layer[li] - 1, refine, lp);
        /* 第 0 个节点已由上一层的末节点覆盖（除了第一层） */
        int start_idx = (li == 0) ? 0 : 1;
        for (int k = start_idx; k <= n_int[li]; k++) {
            x[idx] = cur + lp[k];
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
            pvk__free(frac); pvk__free(per_layer); pvk__free(n_int); pvk__free(lp);
            pvk__free(x); pvk__free(nl); pvk__free(dx);
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
    ctx->mesh.sig_refine   = refine;
    ctx->mesh.sig_total_nm = total_cm * 1e7;
    ctx->have_solution     = 0;   /* 网格换了，旧解作废 */
    ctx->cpl_prev_n        = 0;   /* 接触层掩码也按新网格重算 */

    pvk__free(frac);
    pvk__free(per_layer);
    pvk__free(n_int);
    pvk__free(lp);

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
