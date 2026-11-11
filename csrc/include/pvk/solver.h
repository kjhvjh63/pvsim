/*
 * solver.h — 求解编排 API。
 *
 * 单点求解：pvk_solve_dc
 * 扫描：    pvk_sweep_dc（内部 continuation，每收敛一点回调一次）
 *
 * Python 推荐用 sweep_dc 覆盖单点：n_points=1 就是单点 solve_dc 的
 * 等价调用。
 */
#ifndef PVK_SOLVER_H
#define PVK_SOLVER_H

#include "pvk.h"
#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 可选的每收敛点回调 —— 用于 GUI 进度条。
 * 若返回非 0，sweep_dc 立即以 PVK_EAPI 终止。 */
typedef int (*pvk_point_cb)(void* user, const pvk_solve_info* info);

/* 设置求解器选项（可多次覆盖，保留最后一次）。 */
int pvk_set_solver_opts(pvk_ctx* ctx, const pvk_solver_opts* opts);

/* 单点求解。 */
int pvk_solve_dc(pvk_ctx* ctx,
                 const pvk_device* dev,
                 const pvk_optics_in* opt,
                 const pvk_bias* bias,
                 pvk_solve_info* out_info);

/* continuation 扫描。cb 可传 NULL —— 无回调直接跑完。 */
int pvk_sweep_dc(pvk_ctx* ctx,
                 const pvk_device* dev,
                 const pvk_optics_in* opt,
                 const pvk_bias* bias,
                 double v_start,
                 double v_stop,
                 int    n_points,
                 pvk_point_cb cb,
                 void* cb_user,
                 pvk_solve_info* out_infos_out);   /* 长度 n_points，调用方分配 */

#ifdef __cplusplus
}
#endif

#endif
