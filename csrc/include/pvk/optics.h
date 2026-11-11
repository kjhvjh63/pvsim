/*
 * optics.h — 独立光学入口。
 *
 * 提供两种用法：
 *   1. pvk_optics_compute — 不出 solve 也能算 G(x) 和吸收谱
 *   2. solve_dc / sweep_dc 内部调用它（若 ctx->optics 已设置）
 *
 * out_absorb_spectrum 和 out_gen_profile 都是可选的（传 NULL 忽略）。
 */
#ifndef PVK_OPTICS_H
#define PVK_OPTICS_H

#include "pvk.h"
#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

int pvk_optics_compute(pvk_ctx* ctx,
                       const pvk_device* dev,
                       const pvk_optics_in* opt,
                       double* out_absorb_spectrum,
                       double* out_gen_profile);

#ifdef __cplusplus
}
#endif

#endif
