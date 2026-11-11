/*
 * mesh.h — 1D structured mesh generation.
 *
 * Takes a pvk_device (layer stack + target node count + refine ratio)
 * and produces:
 *   * ctx->x[]         — grid coordinates in cm (out-only)
 *   * ctx->n_node      — actual node count (may exceed target_nodes)
 *   * ctx->node_layer[] — node i 属于哪个 layer index
 *
 * The implementation is deliberately simple for v0.1 (uniform within
 * each layer, interface refine inserts extra points near boundaries).
 * 2D signatures are intentionally NOT stubbed here — they belong in a
 * future ABI bump.
 */
#ifndef PVK_MESH_H
#define PVK_MESH_H

#include "pvk.h"
#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * pvk_build_mesh — 根据 device 重建网格。
 * 必须在 solve_dc / sweep_dc 之前调用；内核内部 solve_dc 首次
 * 调用时也会自动触发，但 Python 层推荐显式调用以便检查。
 */
int pvk_build_mesh(pvk_ctx* ctx, const pvk_device* dev);

/*
 * 网格只读查询 —— Python 需要拿到 x 数组来画曲线 x 轴。
 * out 必须 caller-allocated，长度至少等于 mesh 实际节点数（先
 * 调一次 node_count 得到上限）。
 */
int pvk_node_count(pvk_ctx* ctx);
int pvk_get_x(pvk_ctx* ctx, double* out, int n);

#ifdef __cplusplus
}
#endif

#endif
