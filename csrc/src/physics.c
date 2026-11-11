/*
 * physics.c — field reader (pvk_get_field).
 *
 * v0.1: 只负责把 state.fields[field_id] 拷到 out[]。
 *       真正的物理计算（泊松求解 + 漂移扩散 + SG）在
 *       solver.c 的 solve_dc 里。
 */

#include "pvk_internal.h"

#include <string.h>

PVK_API int pvk_get_field(pvk_ctx* ctx, int field_id, double* out, int n) {
    if (!ctx || !out) return PVK_EINVAL;
    if (field_id < 0 || field_id >= PVK_FIELD_COUNT) {
        pvk__set_error(ctx, PVK_EINVAL, "invalid field_id=%d", field_id);
        return PVK_EINVAL;
    }
    int have = ctx->mesh.n_node;
    if (have <= 0) {
        pvk__set_error(ctx, PVK_EAPI, "mesh not built yet");
        return PVK_EAPI;
    }
    if (n < have) {
        pvk__set_error(ctx, PVK_EINVAL, "out buffer too small: %d < %d", n, have);
        return PVK_EINVAL;
    }
    double* buf = ctx->state.fields[field_id];
    if (!buf) {
        /* 这个 field 还没填过 —— 返回 0 填充 */
        memset(out, 0, sizeof(double) * have);
        return PVK_OK;
    }
    memcpy(out, buf, sizeof(double) * have);
    return PVK_OK;
}
