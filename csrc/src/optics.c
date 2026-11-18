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
 * nk 模式（opt->nk_dir != NULL）：
 *   每层按材料名在目录里找 ``<material>.txt`` /
 *   ``<material>_nk_wavelength.txt`` / ``<material>_nk.txt``，文件格式为
 *   「首行点数，随后 λ(µm) n k」三列（两列时 k=0）。命中后
 *   α(λ) = 4πk/λ，并计入前表面菲涅尔反射（用首层 n 估）。
 *   文件缺失或 λ 超出数据范围时退回带隙唯象模型，保证行为连续。
 *
 * 待补：相干多光束（Atlas 的 beam 默认是射线追踪，单次往返足够做量级对照）。
 */

#include "pvk_internal.h"
#include "pvk/mesh.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#endif

#define PVK_HC_EVNM      1239.841984   /* hc，eV·nm */
#define PVK_ALPHA0       2.0e5         /* cm⁻¹·eV^-0.5 */
#define PVK_R_BACK       0.2           /* 背反射率（单次往返） */

/* 一层的光学常数表：λ 用 µm 存（nk 文件惯例），n/k 无量纲 */
typedef struct {
    double* lam_um;
    double* n;
    double* k;
    int     count;
} pvk_nk;

static FILE* nk_fopen(const char* path) {
#ifdef _WIN32
    wchar_t wpath[1024];
    if (MultiByteToWideChar(CP_UTF8, 0, path, -1, wpath, 1024) > 0) {
        FILE* f = _wfopen(wpath, L"rb");
        if (f) return f;
    }
#endif
    return fopen(path, "rb");
}

static void nk_free(pvk_nk* t) {
    if (!t) return;
    pvk__free(t->lam_um);
    pvk__free(t->n);
    pvk__free(t->k);
    t->lam_um = t->n = t->k = NULL;
    t->count = 0;
}

/* 解析一个 nk 文件；成功返回 1。首行常是「点数」，会被 sscanf<3 自然跳过。 */
static int nk_parse(const char* path, pvk_nk* out) {
    FILE* f = nk_fopen(path);
    if (!f) return 0;

    int cap = 256, cnt = 0;
    double* lam = (double*)pvk__malloc(sizeof(double) * (size_t)cap);
    double* nv  = (double*)pvk__malloc(sizeof(double) * (size_t)cap);
    double* kv  = (double*)pvk__malloc(sizeof(double) * (size_t)cap);
    if (!lam || !nv || !kv) {
        pvk__free(lam); pvk__free(nv); pvk__free(kv);
        fclose(f);
        return 0;
    }

    char line[512];
    while (fgets(line, (int)sizeof(line), f)) {
        double a, b, c = 0.0;
        int got = sscanf(line, "%lf %lf %lf", &a, &b, &c);
        if (got < 2) continue;               /* 表头/注释/空行 */
        if (!(a > 0.0) || !isfinite(a) || !isfinite(b)) continue;
        if (cnt >= cap) {
            int ncap = cap * 2;
            double* nl = (double*)pvk__malloc(sizeof(double) * (size_t)ncap);
            double* nn = (double*)pvk__malloc(sizeof(double) * (size_t)ncap);
            double* nk2 = (double*)pvk__malloc(sizeof(double) * (size_t)ncap);
            if (!nl || !nn || !nk2) { pvk__free(nl); pvk__free(nn); pvk__free(nk2); break; }
            memcpy(nl, lam, sizeof(double) * (size_t)cnt);
            memcpy(nn, nv,  sizeof(double) * (size_t)cnt);
            memcpy(nk2, kv, sizeof(double) * (size_t)cnt);
            pvk__free(lam); pvk__free(nv); pvk__free(kv);
            lam = nl; nv = nn; kv = nk2; cap = ncap;
        }
        lam[cnt] = a;
        nv[cnt]  = b;
        kv[cnt]  = (got >= 3 && isfinite(c) && c >= 0.0) ? c : 0.0;
        cnt++;
        if (cnt >= 200000) break;            /* 异常文件保护 */
    }
    fclose(f);

    if (cnt < 2) {
        pvk__free(lam); pvk__free(nv); pvk__free(kv);
        return 0;
    }
    out->lam_um = lam;
    out->n = nv;
    out->k = kv;
    out->count = cnt;
    return 1;
}

/* 在目录里按材料名找 nk 文件（三种常见命名）。 */
static int nk_load_for_layer(const char* dir, const char* material, pvk_nk* out) {
    if (!dir || !dir[0] || !material || !material[0]) return 0;
    static const char* patterns[] = {
        "%s/%s.txt", "%s/%s_nk_wavelength.txt", "%s/%s_nk.txt"
    };
    char path[1024];
    for (int i = 0; i < 3; i++) {
        int written = snprintf(path, sizeof(path), patterns[i], dir, material);
        if (written <= 0 || written >= (int)sizeof(path)) continue;
        if (nk_parse(path, out)) return 1;
    }
    return 0;
}

/* 线性插值 n / k；λ 用 nm。返回 0 表示超出数据范围。 */
static int nk_lookup(const pvk_nk* t, double lam_nm, double* n_out, double* k_out) {
    if (!t || t->count < 2) return 0;
    double lu = lam_nm * 1e-3;               /* nm → µm */
    if (lu < t->lam_um[0] || lu > t->lam_um[t->count - 1]) return 0;
    int lo = 0, hi = t->count - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (t->lam_um[mid] <= lu) lo = mid; else hi = mid;
    }
    double x0 = t->lam_um[lo], x1 = t->lam_um[hi];
    double w = (x1 > x0) ? (lu - x0) / (x1 - x0) : 0.0;
    if (n_out) *n_out = t->n[lo] + w * (t->n[hi] - t->n[lo]);
    if (k_out) *k_out = t->k[lo] + w * (t->k[hi] - t->k[lo]);
    return 1;
}

/* 菲涅尔反射率（正入射，忽略消光系数）。 */
static double fresnel_r(double n1, double n2) {
    if (!(n1 > 0.0) || !(n2 > 0.0)) return 0.0;
    double r = (n1 - n2) / (n1 + n2);
    return r * r;
}

/* 带隙唯象模型（无 nk 数据时的兜底）。 */
static double alpha_phenom(const pvk_device* dev, int li, double E_eV) {
    if (li < 0 || li >= dev->n_layers) return 0.0;
    const pvk_layer* L = &dev->layers[li];
    if (!L->is_gen || L->is_contact || L->thick_nm <= 0.0) return 0.0;
    if (L->Eg_eV <= 0.0 || E_eV <= L->Eg_eV) return 0.0;
    return PVK_ALPHA0 * sqrt(E_eV - L->Eg_eV);
}

/* 边 (i,i+1) 所在层的 α。规则见 solver.c：节点落在界面上时归属左侧层，
 * 因此两边层号不同时，这条边属于右节点的层。
 * nk 表命中且 λ 在范围内时用 4πk/λ，否则退回唯象模型。 */
static double alpha_of(const pvk_device* dev, int li, double E_eV,
                       double lam_nm, const pvk_nk* nk_table) {
    if (li < 0 || li >= dev->n_layers) return 0.0;
    const pvk_layer* L = &dev->layers[li];
    if (!L->is_gen || L->is_contact || L->thick_nm <= 0.0) return 0.0;

    double k_val;
    if (nk_lookup(nk_table, lam_nm, NULL, &k_val)) {
        if (!(k_val > 0.0) || !isfinite(k_val)) return 0.0;
        return 4.0 * 3.14159265358979323846 * k_val / (lam_nm * 1e-7);  /* cm⁻¹ */
    }
    return alpha_phenom(dev, li, E_eV);
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

    /* nk 模式：按材料名给每层装一张光学常数表（缺失就保持 NULL） */
    pvk_nk* nks = NULL;
    int nk_layers = 0;
    if (opt->nk_dir && opt->nk_dir[0]) {
        nks = (pvk_nk*)pvk__malloc(sizeof(pvk_nk) * (size_t)dev->n_layers);
        if (nks) {
            memset(nks, 0, sizeof(pvk_nk) * (size_t)dev->n_layers);
            for (int li = 0; li < dev->n_layers; li++) {
                const char* mat = dev->layers[li].material;
                if (!mat || !mat[0]) mat = dev->layers[li].name;
                if (nk_load_for_layer(opt->nk_dir, mat, &nks[li])) nk_layers++;
            }
        }
    }
    pvk__log(ctx, PVK_LOG_INFO, nk_layers > 0
             ? "pvk_optics_compute: nk tables loaded, alpha=4*pi*k/lambda"
             : "pvk_optics_compute: Beer-Lambert G(x) built");

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

        /* 逐节点 α */
        for (int i = 0; i < n_node; i++) {
            int li = ctx->mesh.node_layer[i];
            alpha[i] = alpha_of(dev, li, E, lam,
                                (nks && li >= 0 && li < dev->n_layers) ? &nks[li] : NULL);
        }

        if (nks && nk_layers > 0) {
            /* --- 非相干多层光学：逐界面菲涅尔 + 多次往返 ---
             * tau[] 这里改存折射率 n（无 nk 的层用 sqrt(eps)）。 */
            for (int i = 0; i < n_node; i++) {
                int li = ctx->mesh.node_layer[i];
                double nn;
                if (!nk_lookup((li >= 0 && li < dev->n_layers) ? &nks[li] : NULL,
                               lam, &nn, NULL)) {
                    double eps = (li >= 0 && li < dev->n_layers)
                                 ? dev->layers[li].eps_r : 1.0;
                    nn = (eps > 1.0) ? sqrt(eps) : 1.0;
                }
                tau[i] = fmax(nn, 1e-3);
            }

            /* 背接触（末层）反射率：有 nk 时用 ((n-1)²+k²)/((n+1)²+k²)，
             * 否则按半吸收处理。 */
            double n_met, k_met;
            int last = dev->n_layers - 1;
            double r_back = 0.5;
            if (last >= 0 && nk_lookup(&nks[last], lam, &n_met, &k_met) && n_met > 0.0) {
                double a1 = (n_met - 1.0) * (n_met - 1.0) + k_met * k_met;
                double a2 = (n_met + 1.0) * (n_met + 1.0) + k_met * k_met;
                r_back = a1 / a2;
            }

            double absorbed_total = 0.0;
            double I = phi * (1.0 - fresnel_r(1.0, tau[0]));
            for (int pass = 1; pass <= 200 && I > 1e-9 * phi; pass++) {
                int fwd = (pass % 2) == 1;
                int i0 = fwd ? 0 : n_node - 1;
                int st = fwd ? 1 : -1;
                int prev = i0;
                for (int k = 1; k < n_node; k++) {
                    int i = i0 + st * k;
                    int j = prev;
                    if (fabs(tau[i] - tau[j]) > 1e-6) {
                        I *= (1.0 - fresnel_r(tau[j], tau[i]));
                    }
                    double a = 0.5 * (alpha[j] + alpha[i]);
                    if (a > 0.0) {
                        double d = ctx->mesh.dx[fwd ? j : i];
                        if (d > 0.0) {
                            double frac = 1.0 - exp(-a * d);
                            double abs_seg = I * frac;
                            g_field[j] += 0.5 * abs_seg / d;
                            g_field[i] += 0.5 * abs_seg / d;
                            absorbed_total += abs_seg;
                            I *= exp(-a * d);
                        }
                    }
                    prev = i;
                }
                I *= r_back;                 /* 末层界面反射，进入下一程 */
            }

            if (out_absorb_spectrum) {
                out_absorb_spectrum[w] = fmax(0.0, fmin(1.0, absorbed_total / phi));
            }
            continue;
        }

        /* --- 无 nk 数据：带隙唯象模型的 Beer-Lambert --- */
        tau[0] = 0.0;
        for (int i = 1; i < n_node; i++) {
            tau[i] = tau[i-1] + 0.5 * (alpha[i-1] + alpha[i]) * ctx->mesh.dx[i-1];
        }
        double tau_tot = tau[n_node-1];

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

    if (nks) {
        for (int li = 0; li < dev->n_layers; li++) nk_free(&nks[li]);
        pvk__free(nks);
    }
    return PVK_OK;
}
