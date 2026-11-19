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
 *   α(λ) = 4πk/λ，并做非相干多次往返：
 *     - 折射率：表内插值；λ 超出数据范围时钳制到最近端点（与 Atlas 实测
 *       的平滑外推一致），只有整张表缺失时才退回 sqrt(εr)；
 *     - 入射：空气→首层，乘菲涅尔透射 (1-R_front)；
 *     - 层间：每过一层界面乘 (1-R)，用两侧实折射率；
 *     - 背电极：正向程结束乘金属反射率 R_back（末层有 nk 时按复折射率
 *       从相邻半导体侧算，否则取金属典型值 0.95）；
 *     - 内反射：反向程回到前表面时乘 R_front（不是 R_back）。
 *   直到剩余能量 < 1e-9 入射（bounces 在这个模式里不参与限程）。
 *   文件缺失或 λ 超出数据范围时退回带隙唯象模型，保证行为连续。
 *
 * 相干模式（opt->coherent != 0，需 nk_dir 且至少一层有 nk）：
 *   逐 λ 解 1D 亥姆霍兹方程，逐层复折射率 N=n+ik，正入射、空气入射：
 *     - 尾部「is_contact 且无 nk」的层（如 Au）不参与传播，其界面按
 *       完美背镜 E=0；若所有层都有 nk，背边界为空气（透射出去）；
 *     - 由背面波幅向前递推界面处的 A/B，再用空气侧入射波幅=1 定标，
 *       得到 R、T 与内部场 E(x)；
 *     - 吸收密度 q(x)=n·k·|E(x)|²，按 ∫q dx 归一化后乘 A(λ)=1-R(-T)；
 *     - 只有 is_gen && !is_contact 的层得到产生率；bandgap.mask 打开时
 *       E ≤ Eg 的层同样不产生载流子（这部分吸收按热损耗处理）。
 *   默认关闭，非相干路径逐位不变。
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

/* 取折射率 n：表内线性插值；超出数据范围时钳制到最近端点。
 * 实测 Atlas 在 FTO 的 nk 截止（0.898 µm）处没有台阶：单色扫描
 * 0.89→0.90 µm 的吸收率比值 0.953，与端点钳制的 0.952 一致；而旧的
 * √εr 回退（n=3.0）会在 898 nm 突然把前表面反射从 7.8% 抬到 25%
 * （比值 0.766），纯属数值伪影。 */
static int nk_lookup_n_clamped(const pvk_nk* t, double lam_nm, double* n_out) {
    if (!t || t->count < 1 || !n_out) return 0;
    double lu = lam_nm * 1e-3;                 /* nm → µm */
    if (lu < t->lam_um[0]) lu = t->lam_um[0];
    if (lu > t->lam_um[t->count - 1]) lu = t->lam_um[t->count - 1];
    if (t->count == 1) { *n_out = t->n[0]; return 1; }
    return nk_lookup(t, lu * 1e3, n_out, NULL);
}

/* 带隙唯象模型（无 nk 数据时的兜底）。 */
static double alpha_phenom(const pvk_device* dev, int li, double E_eV) {
    if (li < 0 || li >= dev->n_layers) return 0.0;
    const pvk_layer* L = &dev->layers[li];
    if (!L->is_gen || L->is_contact || L->thick_nm <= 0.0) return 0.0;
    if (L->Eg_eV <= 0.0 || E_eV <= L->Eg_eV) return 0.0;
    return PVK_ALPHA0 * sqrt(E_eV - L->Eg_eV);
}

/* n / k 同时取值，超出数据范围时钳制到最近端点（与 n 的既有处理一致）。 */
static int nk_lookup_clamped(const pvk_nk* t, double lam_nm,
                             double* n_out, double* k_out) {
    if (!t || t->count < 1) return 0;
    double lu = lam_nm * 1e-3;
    if (lu < t->lam_um[0]) lu = t->lam_um[0];
    if (lu > t->lam_um[t->count - 1]) lu = t->lam_um[t->count - 1];
    if (t->count == 1) {
        if (n_out) *n_out = t->n[0];
        if (k_out) *k_out = t->k[0];
        return 1;
    }
    return nk_lookup(t, lu * 1e3, n_out, k_out);
}

/* ------------------------------------------------------------ */
/* 相干薄膜光学（TMM）用的小型复数运算。C99 复数在 MSVC 上不可用，
 * 这里用最朴素的 re/im 结构，保证跨编译器一致。 */
typedef struct { double re, im; } pvk_cplx;

static pvk_cplx cx(double re, double im) {
    pvk_cplx z; z.re = re; z.im = im; return z;
}
static pvk_cplx cx_add(pvk_cplx a, pvk_cplx b) { return cx(a.re + b.re, a.im + b.im); }
static pvk_cplx cx_sub(pvk_cplx a, pvk_cplx b) { return cx(a.re - b.re, a.im - b.im); }
static pvk_cplx cx_mul(pvk_cplx a, pvk_cplx b) {
    return cx(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}
static pvk_cplx cx_div(pvk_cplx a, pvk_cplx b) {
    double d = b.re * b.re + b.im * b.im;
    if (!(d > 0.0)) return cx(0.0, 0.0);
    return cx((a.re * b.re + a.im * b.im) / d,
              (a.im * b.re - a.re * b.im) / d);
}
static pvk_cplx cx_scale(pvk_cplx a, double s) { return cx(a.re * s, a.im * s); }
static double cx_abs2(pvk_cplx a) { return a.re * a.re + a.im * a.im; }

/* 单波长相干求解：逐层传输矩阵求内部场 E(x)，把吸收沉积到网格节点。
 *
 * 模型：
 *   - 入射介质为空气，正入射；逐层复折射率 N = n + i k（nk 端点钳制，
 *     缺表时 N = sqrt(eps_r)）；
 *   - 尾部接触层（is_contact 且无 nk，如 Au）不参与传播，其界面用
 *     完美背镜 E=0 代替；若所有层都有 nk，则背边界为空气（透射出去）；
 *   - 吸收密度 q(x) = n·k·|E(x)|²，按 ∫q dx 归一化后乘吸收率 A(λ)；
 *   - 只有 is_gen && !is_contact 的层得到产生率；bandgap.mask 打开时
 *     E ≤ Eg 的层同样不产生载流子（这些光子被当作热损耗，不计入 A）。
 *
 * 返回 A(λ)（含金属/亚带隙吸收）。 */
static double coherent_one(const pvk_ctx* ctx, const pvk_device* dev,
                           const pvk_nk* nks, int n_opt, int back_mirror,
                           const double* xoff_nm, double lam_nm, int bg_mask,
                           pvk_cplx* cN, pvk_cplx* cth,
                           pvk_cplx* cA, pvk_cplx* cB,
                           double* qbuf, double* g_field, double phi) {
    const double TWO_PI = 6.283185307179586476925286766559;
    int m = n_opt;

    for (int j = 0; j < m; j++) {
        double n = 1.0, k = 0.0;
        if (!nk_lookup_clamped(&nks[j], lam_nm, &n, &k)) {
            double eps = dev->layers[j].eps_r;
            n = (eps > 1.0) ? sqrt(eps) : 1.0;
            k = 0.0;
        }
        if (!(n > 0.0) || !isfinite(n)) n = 1.0;
        if (!(k >= 0.0) || !isfinite(k)) k = 0.0;
        cN[j]  = cx(n, k);
        cth[j] = cx(TWO_PI * n * dev->layers[j].thick_nm / lam_nm,
                    TWO_PI * k * dev->layers[j].thick_nm / lam_nm);
    }

    /* 背边界：A/B 是该层左边界处的前向/后向波幅 */
    if (back_mirror) {
        cA[m - 1] = cx(1.0, 0.0);
        double pr = 2.0 * cth[m - 1].re, pi = 2.0 * cth[m - 1].im;
        double damp = exp(-pi);
        cB[m - 1] = cx(-cos(pr) * damp, -sin(pr) * damp);   /* -e^{2iθ} */
    } else {
        cA[m - 1] = cx(1.0, 0.0);
        double pr = 2.0 * cth[m - 1].re, pi = 2.0 * cth[m - 1].im;
        double damp = exp(-pi);
        pvk_cplx e2 = cx(cos(pr) * damp, sin(pr) * damp);   /* e^{2iθ} */
        pvk_cplx rl = cx_div(cx_sub(cN[m - 1], cx(1.0, 0.0)),
                             cx_add(cN[m - 1], cx(1.0, 0.0)));
        cB[m - 1] = cx_mul(e2, rl);
    }

    /* 从背面向正面递推界面波幅 */
    for (int j = m - 2; j >= 0; j--) {
        pvk_cplx s = cx_add(cA[j + 1], cB[j + 1]);
        pvk_cplx dd = cx_mul(cx_div(cN[j + 1], cN[j]),
                             cx_sub(cA[j + 1], cB[j + 1]));
        pvk_cplx u = cx_scale(cx_add(s, dd), 0.5);
        pvk_cplx v = cx_scale(cx_sub(s, dd), 0.5);
        /* A_j = u·e^{-iθ}, B_j = v·e^{+iθ}（θ 的虚部为衰减，指数有界钳制） */
        double hi = cth[j].im < 300.0 ? cth[j].im : 300.0;
        double er = exp(hi), em = exp(-hi);
        pvk_cplx eip = cx(cos(cth[j].re) * em,  sin(cth[j].re) * em);
        pvk_cplx ein = cx(cos(cth[j].re) * er, -sin(cth[j].re) * er);
        cA[j] = cx_mul(u, ein);
        cB[j] = cx_mul(v, eip);
    }

    /* 定标：空气侧入射波幅 = 1 */
    pvk_cplx s0 = cx_add(cA[0], cB[0]);
    pvk_cplx d0 = cx_mul(cN[0], cx_sub(cA[0], cB[0]));
    pvk_cplx den = cx_add(s0, d0);
    double den2 = cx_abs2(den);
    if (!(den2 > 0.0)) return 0.0;
    double sr = 2.0 * den.re / den2, si = -2.0 * den.im / den2;  /* 2/den */
    for (int j = 0; j < m; j++) {
        cA[j] = cx(cA[j].re * sr - cA[j].im * si, cA[j].re * si + cA[j].im * sr);
        cB[j] = cx(cB[j].re * sr - cB[j].im * si, cB[j].re * si + cB[j].im * sr);
    }

    pvk_cplx r = cx_sub(cx_add(cA[0], cB[0]), cx(1.0, 0.0));
    double R = cx_abs2(r);
    double A_abs;
    if (back_mirror) {
        A_abs = 1.0 - R;
    } else if (cth[m - 1].im > 20.0) {
        A_abs = 1.0 - R;                 /* 末层已把透射衰减殆尽 */
    } else {
        double er = exp(-cth[m - 1].im), eg = exp(cth[m - 1].im);
        pvk_cplx ep = cx(cos(cth[m - 1].re) * er,  sin(cth[m - 1].re) * er);
        pvk_cplx em = cx(cos(cth[m - 1].re) * eg, -sin(cth[m - 1].re) * eg);
        pvk_cplx t = cx_add(cx_mul(cA[m - 1], ep), cx_mul(cB[m - 1], em));
        A_abs = 1.0 - R - cx_abs2(t);
    }
    if (!(A_abs > 0.0) || !isfinite(A_abs)) A_abs = 0.0;
    if (A_abs > 1.0) A_abs = 1.0;

    /* 逐节点相对吸收密度 q = n·k·|E|² */
    int n_node = ctx->mesh.n_node;
    for (int i = 0; i < n_node; i++) {
        qbuf[i] = 0.0;
        int li = ctx->mesh.node_layer[i];
        if (li < 0 || li >= m) continue;
        double nn = cN[li].re, kk = cN[li].im;
        if (!(kk > 0.0)) continue;
        double x_nm = ctx->mesh.x[i] * 1e7 - xoff_nm[li];
        if (x_nm < 0.0) x_nm = 0.0;
        double ph = TWO_PI * nn * x_nm / lam_nm;
        double ex = TWO_PI * kk * x_nm / lam_nm;
        if (ex > 690.0) ex = 690.0;            /* 防 exp 溢出 */
        double ep_r = cos(ph) * exp(-ex), ep_i = sin(ph) * exp(-ex);
        double em_r = cos(ph) * exp(ex),  em_i = -sin(ph) * exp(ex);
        pvk_cplx E = cx_add(cx_mul(cA[li], cx(ep_r, ep_i)),
                            cx_mul(cB[li], cx(em_r, em_i)));
        qbuf[i] = nn * kk * cx_abs2(E);
    }

    double qsum = 0.0;
    for (int i = 0; i + 1 < n_node; i++) {
        qsum += 0.5 * (qbuf[i] + qbuf[i + 1]) * ctx->mesh.dx[i];
    }
    if (!(qsum > 0.0) || !isfinite(qsum)) return A_abs;

    double fac = A_abs / qsum;
    double E_ph = PVK_HC_EVNM / lam_nm;
    for (int i = 0; i < n_node; i++) {
        int li = ctx->mesh.node_layer[i];
        if (li < 0 || li >= m) continue;
        const pvk_layer* L = &dev->layers[li];
        if (!L->is_gen || L->is_contact) continue;
        if (bg_mask && L->Eg_eV > 0.0 && E_ph <= L->Eg_eV) continue;
        g_field[i] += phi * fac * qbuf[i];
    }
    return A_abs;
}

/* 边 (i,i+1) 所在层的 α。规则见 solver.c：节点落在界面上时归属左侧层，
 * 因此两边层号不同时，这条边属于右节点的层。
 * nk 表命中且 λ 在范围内时用 4πk/λ，否则退回唯象模型。 */
static double alpha_of(const pvk_device* dev, int li, double E_eV,
                       double lam_nm, const pvk_nk* nk_table, int bg_mask) {
    if (li < 0 || li >= dev->n_layers) return 0.0;
    const pvk_layer* L = &dev->layers[li];
    if (!L->is_gen || L->is_contact || L->thick_nm <= 0.0) return 0.0;

    double k_val;
    if (nk_lookup(nk_table, lam_nm, NULL, &k_val)) {
        /* bandgap.mask：实测 nk 表普遍带亚带隙吸收尾巴（Urbach/缺陷），
         * 直接 4πk/λ 会让 Eg=1.61 eV 的吸收层一直吸到 1.1 µm，
         * Jsc 冲出带隙上限（复刻卡实测 28.18，理论上限 ~24.5）。
         * 声明了 Eg 且要求掩模时按 Eg 截断。 */
        if (bg_mask && L->Eg_eV > 0.0 && E_eV <= L->Eg_eV) return 0.0;
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

    /* --- 相干薄膜光学（opt-in）：确定参与传播的光学栈 ---
     * 尾部「接触层且无 nk」（复刻卡里的 Au）不传播，其界面当完美背镜；
     * 其余层按 nk 复折射率逐层求解。默认关闭 → 非相干路径逐位不变。 */
    int coherent = (opt->coherent != 0) && (nks != NULL) && (nk_layers > 0);
    int n_opt = dev->n_layers;
    int back_mirror = 0;
    pvk_cplx* cN   = NULL;
    pvk_cplx* cth  = NULL;
    pvk_cplx* cA   = NULL;
    pvk_cplx* cB   = NULL;
    double*   qbuf = NULL;
    double*   xoff = NULL;
    if (coherent) {
        while (n_opt > 1 && nks[n_opt - 1].count == 0
               && dev->layers[n_opt - 1].is_contact) {
            n_opt--;
        }
        back_mirror = (n_opt < dev->n_layers) ? 1 : 0;
        cN   = (pvk_cplx*)pvk__malloc(sizeof(pvk_cplx) * (size_t)n_opt);
        cth  = (pvk_cplx*)pvk__malloc(sizeof(pvk_cplx) * (size_t)n_opt);
        cA   = (pvk_cplx*)pvk__malloc(sizeof(pvk_cplx) * (size_t)n_opt);
        cB   = (pvk_cplx*)pvk__malloc(sizeof(pvk_cplx) * (size_t)n_opt);
        qbuf = (double*)pvk__malloc(sizeof(double) * (size_t)n_node);
        xoff = (double*)pvk__malloc(sizeof(double) * (size_t)n_opt);
        if (!cN || !cth || !cA || !cB || !qbuf || !xoff) {
            pvk__free(cN); pvk__free(cth); pvk__free(cA);
            pvk__free(cB); pvk__free(qbuf); pvk__free(xoff);
            cN = cth = cA = cB = NULL; qbuf = xoff = NULL;
            coherent = 0;
        } else {
            double cur = 0.0;
            for (int j = 0; j < n_opt; j++) {
                xoff[j] = cur;
                cur += dev->layers[j].thick_nm;
            }
            pvk__log(ctx, PVK_LOG_INFO,
                     back_mirror
                     ? "pvk_optics_compute: coherent TMM, perfect back mirror"
                     : "pvk_optics_compute: coherent TMM, air back boundary");
        }
    }

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

        if (coherent) {
            double A = coherent_one(ctx, dev, nks, n_opt, back_mirror, xoff,
                                    lam, opt->bandgap_mask,
                                    cN, cth, cA, cB, qbuf, g_field, phi);
            if (out_absorb_spectrum) out_absorb_spectrum[w] = A;
            continue;
        }

        /* 逐节点 α */
        for (int i = 0; i < n_node; i++) {
            int li = ctx->mesh.node_layer[i];
            alpha[i] = alpha_of(dev, li, E, lam,
                                (nks && li >= 0 && li < dev->n_layers) ? &nks[li] : NULL,
                                opt->bandgap_mask);
        }

        if (nks && nk_layers > 0) {
            /* --- 非相干多层光学：逐界面菲涅尔 + 多次往返 ---
             * tau[] 这里改存折射率 n（无 nk 的层用 sqrt(eps)）。 */
            for (int i = 0; i < n_node; i++) {
                int li = ctx->mesh.node_layer[i];
                double nn;
                if (!nk_lookup_n_clamped((li >= 0 && li < dev->n_layers)
                                         ? &nks[li] : NULL, lam, &nn)) {
                    double eps = (li >= 0 && li < dev->n_layers)
                                 ? dev->layers[li].eps_r : 1.0;
                    nn = (eps > 1.0) ? sqrt(eps) : 1.0;
                }
                tau[i] = fmax(nn, 1e-3);
            }

            /* 前表面（空气→首层）与内反射（首层→空气）的菲涅尔反射率。
             * 之前每一程结束都乘背电极反射率，导致“内部光从前表面反射”
             * 被当成金属镜面（R=0.5）——既非物理，也把多次往返的能量
             * 高估/低估混在一起。 */
            double r_front = fresnel_r(1.0, tau[0]);

            /* 背接触（末层）反射率。有 nk 时按复折射率、从相邻半导体
             * 侧入射计算：R = ((n_s-n_m)²+k_m²)/((n_s+n_m)²+k_m²)。
             * 没有 nk（复刻卡只写了 Au region，但 nk.dir 里没有 Au.txt）
             * 时按金属典型值 0.95；旧的 0.5 无物理依据，且会把亚带隙
             * 光（吸收长度 ~1 µm，靠背反射二次吸收）低估掉约 2 mA/cm²。 */
            int last = dev->n_layers - 1;
            double r_back = 0.95;
            double n_met, k_met;
            if (last >= 0 && nk_lookup(&nks[last], lam, &n_met, &k_met) && n_met > 0.0) {
                double n_s = 1.0;
                for (int i = n_node - 1; i >= 0; i--) {
                    int li = ctx->mesh.node_layer[i];
                    if (li != last) { n_s = tau[i]; break; }
                }
                double a1 = (n_s - n_met) * (n_s - n_met) + k_met * k_met;
                double a2 = (n_s + n_met) * (n_s + n_met) + k_met * k_met;
                r_back = a1 / a2;
            }

            double absorbed_total = 0.0;
            double I = phi * (1.0 - r_front);
            for (int pass = 1; pass <= 200 && I > 1e-9 * phi; pass++) {
                int fwd = (pass % 2) == 1;
                int i0 = fwd ? 0 : n_node - 1;
                int st = fwd ? 1 : -1;
                int prev = i0;
                for (int k = 1; k < n_node; k++) {
                    int i = i0 + st * k;
                    int j = prev;
                    /* 背电极界面不在这里扣透射：金属反射已由 r_back 统一
                     * 计入，否则会把复折射率反射算两遍（且符号相反）。 */
                    int li_i = ctx->mesh.node_layer[i];
                    int li_j = ctx->mesh.node_layer[j];
                    int back_iface = (li_i == last || li_j == last);
                    if (!back_iface && fabs(tau[i] - tau[j]) > 1e-6) {
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
                /* 正向到背电极 → 乘金属反射率；反向到前表面 → 乘前表面
                 * 反射率。两者不可混用。 */
                I *= fwd ? r_back : r_front;
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
    pvk__free(cN);
    pvk__free(cth);
    pvk__free(cA);
    pvk__free(cB);
    pvk__free(qbuf);
    pvk__free(xoff);

    if (nks) {
        for (int li = 0; li < dev->n_layers; li++) nk_free(&nks[li]);
        pvk__free(nks);
    }
    return PVK_OK;
}
