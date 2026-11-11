/*
 * optics.c — 光学产生率 pvk_optics_compute。
 *
 * 模型（v1）：
 *   1) 逐波长 λ 积分。flux 是光谱光子通量 (cm⁻² s⁻¹ nm⁻¹)，
 *      乘 bin 宽度得到该 bin 的光子数 (cm⁻² s⁻¹)，再乘 intensity/100。
 *   2) 吸收系数用带隙唯象模型：α(λ) = A·sqrt(E - Eg)，E > Eg；
 *      其中 A = 2e5 cm⁻¹·eV^-0.5（MAPbI3 在 Eg+0.25 eV 处约 1e5 cm⁻¹）。
 *      只有 is_gen && !is_contact 的层吸收，带隙掩模由此自动生效。
 *   3) Beer-Lambert：τ(x) = ∫₀ˣ α dx'，正向 G = Φ·α·exp(-τ)。
 *   4) bounces > 1 时叠加一次背反射（反射率 R），
 *      G += Φ·α·R·exp(-(2τ_tot - τ(x)))。bounces 的具体次数不影响
 *      单次往返近似，只用作 "要不要算背反射" 的开关。
 *
 * 输出：G_OPT 场 (cm⁻³ s⁻¹)；可选吸收谱 A(λ) = 1 - exp(-τ_tot)。
 *
 * 后续可补：从 nk_dir 读真实 n,k、菲涅尔界面反射、相干多光束。
 */

#include "pvk_internal.h"
#include "pvk/mesh.h"

#include <math.h>
#include <string.h>

#define PVK_HC_EVNM      1239.841984   /* hc，eV·nm */
#define PVK_ALPHA0       2.0e5         /* cm⁻¹·eV^-0.5 */
#define PVK_R_BACK       0.2           /* 背反射率（单次往返） */

/* 边 (i,i+1) 所在层的 α。规则见 solver.c：节点落在界面上时归属左侧层，
 * 因此两边层号不同时，这条边属于右节点的层。 */
static double alpha_of(const pvk_device* dev, int li, double E_eV) {
    if (li < 0 || li >= dev->n_layers) return 0.0;
    const pvk_layer* L = &dev->layers[li];
    if (!L->is_gen || L->is_contact || L->thick_nm <= 0.0) return 0.0;
    if (L->Eg_eV <= 0.0 || E_eV <= L->Eg_eV) return 0.0;
    return PVK_ALPHA0 * sqrt(E_eV - L->Eg_eV);
}

PVK_API int pvk_optics_compute(pvk_ctx* ctx,
                               const pvk_device* dev,
                               const pvk_optics_in* opt,
                               double* out_absorb_spectrum,
                               double* out_gen_profile) {
    if (!ctx || !dev || !opt) return PVK_EINVAL;
    if (opt->n_wl <= 0 || opt->wl_nm == NULL || opt->flux == NULL) {
        return PVK_EINVAL;
    }
    if (dev->n_layers <= 0 || dev->layers == NULL) return PVK_EINVAL;

    int n_node = ctx->mesh.n_node;
    if (n_node <= 0) {
        int rc = pvk_build_mesh(ctx, dev);
        if (rc != PVK_OK) return rc;
        n_node = ctx->mesh.n_node;
    }
    if (n_node < 2) return PVK_EAPI;

    int rc = pvk__ensure_field_buf(ctx, PVK_FIELD_G_OPT);
    if (rc != PVK_OK) return rc;
    double* g_field = ctx->state.fields[PVK_FIELD_G_OPT];

    /* 逐节点 α 的临时缓冲（每个波长重算） */
    double* alpha = (double*)pvk__malloc(sizeof(double) * n_node);
    double* tau   = (double*)pvk__malloc(sizeof(double) * n_node);
    if (!alpha || !tau) {
        pvk__free(alpha); pvk__free(tau);
        return PVK_EMEM;
    }

    for (int i = 0; i < n_node; i++) g_field[i] = 0.0;

    double scale = (opt->intensity_mW_cm2 > 0 ? opt->intensity_mW_cm2 : 0.0) / 100.0;
    int use_back = (opt->bounces > 1);
    int n_wl = opt->n_wl;

    for (int w = 0; w < n_wl; w++) {
        double lam = opt->wl_nm[w];
        if (!(lam > 0.0)) continue;

        /* bin 宽度 dλ：内部用中心差分，端点用单侧 */
        double dl;
        if (n_wl == 1) {
            dl = 1.0;
        } else if (w == 0) {
            dl = opt->wl_nm[1] - opt->wl_nm[0];
        } else if (w == n_wl - 1) {
            dl = opt->wl_nm[n_wl-1] - opt->wl_nm[n_wl-2];
        } else {
            dl = 0.5 * (opt->wl_nm[w+1] - opt->wl_nm[w-1]);
        }
        if (!(dl > 0.0)) continue;

        double phi = opt->flux[w] * dl * scale;   /* cm⁻² s⁻¹ */
        if (!(phi > 0.0) || !isfinite(phi)) continue;

        double E = PVK_HC_EVNM / lam;             /* eV */

        /* 逐节点 α + 累积光学厚度 τ */
        for (int i = 0; i < n_node; i++) {
            int li = ctx->mesh.node_layer[i];
            alpha[i] = alpha_of(dev, li, E);
        }
        tau[0] = 0.0;
        for (int i = 1; i < n_node; i++) {
            tau[i] = tau[i-1] + 0.5 * (alpha[i-1] + alpha[i]) * ctx->mesh.dx[i-1];
        }
        double tau_tot = tau[n_node-1];

        /* 正向 + 背反射 */
        for (int i = 0; i < n_node; i++) {
            if (alpha[i] <= 0.0) continue;
            double g = phi * alpha[i] * exp(-tau[i]);
            if (use_back) {
                g += phi * alpha[i] * PVK_R_BACK * exp(-(2.0 * tau_tot - tau[i]));
            }
            g_field[i] += g;
        }

        if (out_absorb_spectrum) {
            out_absorb_spectrum[w] = 1.0 - exp(-tau_tot);
        }
    }

    /* 兜底：非负 + 有限 */
    for (int i = 0; i < n_node; i++) {
        if (!isfinite(g_field[i]) || g_field[i] < 0.0) g_field[i] = 0.0;
    }

    if (out_gen_profile) {
        memcpy(out_gen_profile, g_field, sizeof(double) * n_node);
    }

    pvk__free(alpha);
    pvk__free(tau);

    pvk__log(ctx, PVK_LOG_INFO, "pvk_optics_compute: Beer-Lambert G(x) built");
    return PVK_OK;
}
