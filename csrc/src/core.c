/*
 * core.c — lifecycle, malloc/free, logging.
 *
 * 所有分配都走这里的 wrapper，将来替换为 arena/内存池时只改这
 * 一个文件。  public API 函数按 pvk.h 签名顺序排列。
 */

#include "pvk_internal.h"

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ─────────────────────────────────────────────────────────────────────
 * util
 * ───────────────────────────────────────────────────────────────────── */
void* pvk__malloc(size_t n) {
    if (n == 0) return NULL;
    return calloc(1, n);   /* 零初始化，比 malloc 安全 */
}
void pvk__free(void* p) { free(p); }

char* pvk__strdup(const char* s) {
    if (!s) return NULL;
    size_t len = strlen(s) + 1;
    char*  out = (char*)malloc(len);
    return out ? memcpy(out, s, len) : NULL;
}

void pvk__set_error(pvk_ctx* ctx, int code, const char* fmt, ...) {
    if (!ctx) return;
    ctx->last_err_code = code;
    va_list ap, ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int need = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (need > 0 && (size_t)need < sizeof(ctx->last_error)) {
        vsnprintf(ctx->last_error, sizeof(ctx->last_error), fmt, ap2);
    } else {
        /* 截断兜底 */
        vsnprintf(ctx->last_error, sizeof(ctx->last_error), "%s", "error message truncated");
    }
    va_end(ap2);
}

void pvk__log(pvk_ctx* ctx, int level, const char* msg) {
    if (!ctx || !ctx->log_cb) return;
    ctx->log_cb(ctx->log_user, level, msg ? msg : "");
}

int pvk__ensure_field_buf(pvk_ctx* ctx, int field_id) {
    if (!ctx) return PVK_EAPI;
    if (field_id < 0 || field_id >= PVK_FIELD_COUNT) return PVK_EINVAL;
    int n = ctx->mesh.n_node;
    if (n <= 0) return PVK_EAPI;   /* 网格还没建 */
    if (ctx->state.fields[field_id]) {
        pvk__free(ctx->state.fields[field_id]);
    }
    ctx->state.fields[field_id] = (double*)pvk__malloc(sizeof(double) * n);
    if (!ctx->state.fields[field_id]) return PVK_EMEM;
    return PVK_OK;
}

void pvk__release_cached_device(pvk_ctx* ctx) {
    if (!ctx || !ctx->last_device.layers) return;
    for (int i = 0; i < ctx->last_device.n_layers; i++) {
        pvk__free((void*)ctx->last_device.layers[i].name);
        pvk__free((void*)ctx->last_device.layers[i].material);
    }
    pvk__free((void*)ctx->last_device.layers);
    ctx->last_device.layers   = NULL;
    ctx->last_device.n_layers = 0;
}

/* ─────────────────────────────────────────────────────────────────────
 * pvk.h exports
 * ───────────────────────────────────────────────────────────────────── */

PVK_API int pvk_abi_version(void) {
    return PVK_ABI_VERSION;
}

PVK_API pvk_ctx* pvk_create(pvk_log_fn log_cb, void* user) {
    pvk_ctx* ctx = (pvk_ctx*)pvk__malloc(sizeof(pvk_ctx));
    if (!ctx) return NULL;

    ctx->log_cb  = log_cb;
    ctx->log_user = user;
    ctx->last_err_code = PVK_OK;
    ctx->last_error[0] = '\0';
    ctx->inputs_valid   = 0;

    /* 求解器默认选项 —— 与 Python SolverOpts 默认值一致 */
    ctx->solver_opts.res_tol      = 1e-8;
    ctx->solver_opts.max_iter     = 60;
    ctx->solver_opts.method       = 2;   /* NEWTON_WITH_WARMSTART */
    ctx->solver_opts.damping      = 1.0;
    ctx->solver_opts.v_init_step  = 0.05;
    ctx->solver_opts.v_min_step   = 0.002;
    ctx->solver_opts.v_max_step   = 0.10;
    ctx->solver_opts.clamp_residual = 1;

    /* mesh/state 先零初始化 */
    ctx->mesh.n_node   = 0;
    ctx->mesh.x        = NULL;
    ctx->mesh.node_layer = NULL;
    ctx->mesh.dx       = NULL;
    for (int i = 0; i < PVK_FIELD_COUNT; i++) ctx->state.fields[i] = NULL;

    pvk__log(ctx, PVK_LOG_INFO, "pvk_create: ctx initialised");
    return ctx;
}

PVK_API void pvk_destroy(pvk_ctx* ctx) {
    if (!ctx) return;
    pvk__log(ctx, PVK_LOG_DEBUG, "pvk_destroy");

    /* mesh */
    pvk__free(ctx->mesh.x);
    pvk__free(ctx->mesh.node_layer);
    pvk__free(ctx->mesh.dx);

    /* state fields */
    for (int i = 0; i < PVK_FIELD_COUNT; i++) {
        pvk__free(ctx->state.fields[i]);
    }

    /* 缓存的 device 副本（含每层 strdup 的字符串） */
    pvk__release_cached_device(ctx);

    /* 只释放自己 strdup 的 nk_dir；wl_nm/flux 是调用方（numpy）的指针，
     * 属于"借用"，释放它们会造成堆破坏。 */
    pvk__free((void*)ctx->last_optics.nk_dir);
    ctx->last_optics.nk_dir = NULL;

    pvk__free(ctx);
}

PVK_API int pvk_last_error(pvk_ctx* ctx, char* buf, int buflen) {
    if (!buf || buflen <= 0) return 0;
    if (!ctx || ctx->last_err_code == PVK_OK) {
        int n = snprintf(buf, (size_t)buflen, "%s", "PVK_OK");
        return (n > 0 && n < buflen) ? n : 0;
    }
    int n = snprintf(buf, (size_t)buflen, "%s", ctx->last_error);
    return (n > 0 && n < buflen) ? n : (buflen - 1);
}
