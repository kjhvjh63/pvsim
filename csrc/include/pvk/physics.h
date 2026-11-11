/*
 * physics.h — field evaluation and material helper queries.
 *
 * 真实的残差/雅可比装配在 pvk_internal.h 里（不对外），
 * 本头只暴露 Python 需要的两个只读能力：
 *
 *   1. pvk_get_field      — 拿某个 field_id 沿 x 的剖面。
 *                           solve 之后自动可用；field_id 枚举见 types.h。
 *   2. pvk_solve_dc       — 单点求解（唯一"重"API）。
 */
#ifndef PVK_PHYSICS_H
#define PVK_PHYSICS_H

#include "pvk.h"
#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 读出某个 field_id 沿 x 方向的网格值到 out[]。
 * out 长度必须 >= node_count（用 pvk_node_count 查）。 */
int pvk_get_field(pvk_ctx* ctx, int field_id, double* out, int n);

#ifdef __cplusplus
}
#endif

#endif
