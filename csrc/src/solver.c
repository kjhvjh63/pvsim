/*
 * solver.c — 一维漂移扩散 + 泊松，阻尼 Newton 求解。
 *
 * 未知量（每个节点 3 个，全部无量纲化：电位 / Vt）：
 *   u[3i+0] = ψ / Vt          静电势
 *   u[3i+1] = φn / Vt         电子准费米势
 *   u[3i+2] = φp / Vt         空穴准费米势
 *
 * 方程组：
 *   Poisson : d/dx(ε dψ/dx) = -q (p - n + Nd - Na)
 *   电子连续: dJn/dx =  q (R - G),  SG (Scharfetter-Gummel) 离散
 *   空穴连续: dJp/dx = -q (R - G)
 *
 * 载流子浓度（含异质结亲和势差）：
 *   n = Nc exp((ψ + χ + U_ref + φn) / Vt)
 *   p = Nv exp(-(ψ + χ + Eg + U_ref + φp) / Vt)
 *   U_ref 由阴极（x=0，层 0）的电中性平衡态定出，保证 V=0 时是平衡解。
 *
 * 边界与建模约定（v1）：
 *   - 只有真正的金属／透明电极层（is_contact）钉死 ψ/φn/φp 作为 Ohmic
 *     边界；ETL、HTL、吸收层全部按完整双载流子漂移扩散求解。
 *     （电极层内部少数载流子会掉到 1e-40 以下，连续性方程离散后是 0=0，
 *     数值上不可解，所以整层钉死。）
 *   - x=0 是阴极（layer 0，电子侧），x=L 是阳极（最后一层，空穴侧）。
 *     与 Python 侧 “J 为 anode→cathode、光照短路 J<0” 的约定一致：
 *     对外报的电流是内建 +x 电流取负。
 *   - Ohmic 接触；SCHOTTKY 暂按 Ohmic 处理。
 *
 * 异质结界面（两处关键处理，缺一则暗态无解）：
 *   1) SG 的 Bernoulli 自变量用“广义带边差”而不是单纯的静电势差：
 *        δn = Δψ/Vt + Δχ/Vt + ln(Nc_R/Nc_L)
 *        δp = Δψ/Vt + Δχ/Vt + ΔEg/Vt + ln(Nv_L/Nv_R)
 *      这样零电流条件恰好是界面两侧准费米能级连续（细致平衡），
 *      即界面热电子发射在 vn→∞ 的极限；少了带边项会多出 exp(Δχ/Vt)
 *      （最大 e^96）的残余通量。
 *   2) 电极层整层改用相邻半导体的 ε/Nc/Nv/χ/Eg（理想 Ohmic 接触：
 *      电极只提供电位与费米能级）。否则电极材料（Au: χ=4.7, Eg=1.0）
 *      与 HTL（Spiro: χ=2.5, Eg=3.0）之间 2.5 eV 的人为带边阶跃会要求
 *      半导体侧的载流子密度达到 1e38 才可能零电流，暗态平衡不存在。
 *
 * 数值：
 *   - 数值雅可比（前向差分）+ 稠密 LU（行均衡 + 部分选主元 + 对角正则）。
 *   - 连续性方程按当地载流子密度归一化，避免异质结界面处量级坍缩。
 *   - 线搜索用残差 2-范数。
 *   - 冷启动：从 equilibrium_guess 的平衡态 ψ（φn=φp=0）出发，先在几个
 *     小正偏压下直接解完整方程；只有直接解失败才退到产生率 homotopy
 *     （亮态按光强爬坡，暗态用吸收层里的合成产生率压到 0）。
 *     起点确定后按 ≤0.25 V 的电压 continuation 推到目标偏压。
 *   - 平衡态 ψ 本身用「带边 + 掺杂 homotopy」求：homo=0 时全器件本征、
 *     同一带结构（各层电中性电位都是 ψ=0，初值即精确解），再逐步把
 *     掺杂和 χ/Eg 差异拉到真实值，每步失败就折半步长重试。
 *
 * 符号：所有报告的电流（JN/JP/JTOTAL/info.J）都是“阳极→阴极”（-x）方向，
 *       单位 mA/cm²。
 *
 * SG 的 Bernoulli 函数 B(x) = x/(e^x - 1)，用稳定分支求值。
 */

#include "pvk_internal.h"
#include "pvk/mesh.h"
#include "pvk/optics.h"
#include "pvk/solver.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#define PVK_Q_C     1.602176634e-19
#define PVK_EPS0    8.8541878128e-14   /* F/cm */
#define PVK_KB_EV   8.617333262e-5     /* eV/K */

/* 单个求解点的模型参数（按节点展开） */
typedef struct {
    int     n;
    double  Vt;
    double  U_ref;      /* 参考准费米势 (eV) */
    double  V_app;      /* 准费米劈裂 (V) */
    double  psi_a;      /* 阳极电位 BC (V) */

    const double* x;    /* [n] cm */
    const double* h;    /* [n-1] cm */
    const int*    node_layer;

    double* eps;        /* F/cm */
    double* chi;        /* eV */
    double* Eg;
    double* Nc;
    double* Nv;
    double* mun;
    double* mup;
    double* taun;
    double* taup;
    double* Nd;
    double* Na;
    double* G;          /* cm⁻³ s⁻¹ */
    double* ni;
    double* chis;       /* chi/Vt + U_ref/Vt */
    double* egs;        /* Eg/Vt */
    double* hc;         /* 控制体宽度 */
    int*    bc;         /* 0=求解, 1=阴极型 Dirichlet, 2=阳极型 Dirichlet */
    int     eq_minority;/* 1 = 少数载流子退化时用 φ_min=φ_maj 兜底 */

    double  u_bc[2];
    double  wn_bc[2];
    double  wp_bc[2];
} pvk_model;

/* ─────────────────────────────────────────────────────────────────────
 * 工具
 * ───────────────────────────────────────────────────────────────────── */

/* B(x) = x / (e^x - 1)，稳定求值 */
static double bern(double x) {
    if (x > 1e-8) {
        if (x > 700.0) return 0.0;
        return x / (exp(x) - 1.0);
    }
    if (x < -1e-8) {
        double y = -x;
        if (y > 700.0) return y;
        return y / (1.0 - exp(-y));
    }
    return 1.0 - 0.5 * x + x * x / 12.0;
}

static double clampd(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/* 边 (i-1,i) 的属性取自右节点的层：节点落在界面上时归属左侧层，
 * 所以层号不同时这条边实际位于右节点的层里。 */
static void node_dens(const pvk_model* m, int i, const double* u,
                      double* nout, double* pout) {
    int n = m->n;
    double en = u[i] + m->chis[i] + u[n + i];
    double ep = -(u[i] + m->chis[i] + m->egs[i] + u[2*n + i]);
    en = clampd(en, -280.0, 280.0);
    ep = clampd(ep, -280.0, 280.0);
    *nout = m->Nc[i] * exp(en);
    *pout = m->Nv[i] * exp(ep);
}

/* ─────────────────────────────────────────────────────────────────────
 * 残差 —— 返回按“最大项”归一化的相对残差 ∞-范数
 * ───────────────────────────────────────────────────────────────────── */
static double residual(const pvk_model* m, const double* u, double* r,
                       double* nbuf, double* pbuf, double* rnet) {
    const int n = m->n;

    for (int i = 0; i < n; i++) {
        node_dens(m, i, u, &nbuf[i], &pbuf[i]);
        if (!isfinite(nbuf[i])) nbuf[i] = 1e300;
        if (!isfinite(pbuf[i])) pbuf[i] = 1e300;
    }

    /* SRH 净复合 R（正=复合） */
    for (int i = 0; i < n; i++) {
        double ni = m->ni[i];
        double np = nbuf[i] * pbuf[i];
        if (!isfinite(np)) np = 1e300;
        double den = m->taup[i] * (nbuf[i] + ni) + m->taun[i] * (pbuf[i] + ni);
        if (den > 0.0 && isfinite(den)) {
            rnet[i] = (np - ni * ni) / den;
        } else {
            rnet[i] = 0.0;
        }
        if (!isfinite(rnet[i])) rnet[i] = 0.0;
    }

    double rmax = 0.0;
    double rsum2 = 0.0;

    for (int i = 0; i < n; i++) {
        if (m->bc[i]) {
            int e = m->bc[i] - 1;
            r[i]           = u[i]           - m->u_bc[e];
            r[n + i]       = u[n + i]       - m->wn_bc[e];
            r[2 * n + i]   = u[2 * n + i]   - m->wp_bc[e];
            double a = fabs(r[i]);
            if (fabs(r[n+i]) > a) a = fabs(r[n+i]);
            if (fabs(r[2*n+i]) > a) a = fabs(r[2*n+i]);
            if (a > rmax) rmax = a;
            rsum2 += r[i]*r[i] + r[n+i]*r[n+i] + r[2*n+i]*r[2*n+i];
            continue;
        }

        double hL = m->h[i-1];
        double hR = m->h[i];
        double hc = m->hc[i];

        /* ---- Poisson ---- */
        double fluxR = m->eps[i+1] * (u[i+1] - u[i]) * m->Vt / hR;
        double fluxL = m->eps[i]   * (u[i]   - u[i-1]) * m->Vt / hL;
        double fterm = (fluxR - fluxL) / (PVK_Q_C * hc);
        double cterm = pbuf[i] - nbuf[i] + m->Nd[i] - m->Na[i];
        r[i] = fterm + cterm;
        {
            double den = fabs(fterm);
            if (fabs(cterm) > den) den = fabs(cterm);
            if (den < 1.0) den = 1.0;
            double a = fabs(r[i]) / den;
            if (a > rmax) rmax = a;
            rsum2 += a * a;
        }

        /* ---- 电子连续 ----
         * 按当地载流子密度归一化：异质结界面两侧可差几十个数量级，
         * 固定尺度会让该行残差塌到 1e-30 或涨到 1e30。
         *
         * 异质结界面：SG 的 Bernoulli 自变量必须是“广义带边差”，
         * 只用电势差 Δψ 时不满足界面细致平衡。推导（等准费米 ⇒ 零电流）：
         *   n_R/n_L = (Nc_R/Nc_L)·exp(Δψ/Vt + Δχ/Vt)
         *   零电流条件 n_R·B(δ) = n_L·B(-δ)，而 B(-δ)/B(δ) = e^δ
         *   ⇒ e^δ = n_R/n_L ⇒ δn = Δψ/Vt + Δχ/Vt + ln(Nc_R/Nc_L)
         * 少这一项时零电流条件会多出 exp(Δχ/Vt)（最大 e^96）的残余通量，
         * 暗态在界面处根本无解 —— 这就是 Spiro/Au、TiO2/钙钛矿节点的奇异源。
         * 等效地：δn 就是“带边 + kT·lnNc”这个广义能级之差除以 Vt。 */
        double duR = u[i+1] - u[i];
        double duL = u[i]   - u[i-1];
        double dnR = duR + (m->chis[i+1] - m->chis[i]) + log(m->Nc[i+1] / m->Nc[i]);
        double dnL = duL + (m->chis[i]   - m->chis[i-1]) + log(m->Nc[i] / m->Nc[i-1]);
        /* 空穴与之对称（Ev = Ec - Eg，注意 p 的指数里是 (Ev-Efp)/Vt，
         * 所以零电流条件 p_L/p_R = e^δp 给出的带边项符号与电子相反）：
         *   δp = Δψ/Vt + Δχ/Vt + ΔEg/Vt + ln(Nv_L/Nv_R) */
        double dpR = duR + (m->chis[i+1] - m->chis[i])
                     + (m->egs[i+1] - m->egs[i]) + log(m->Nv[i] / m->Nv[i+1]);
        double dpL = duL + (m->chis[i]   - m->chis[i-1])
                     + (m->egs[i]   - m->egs[i-1]) + log(m->Nv[i-1] / m->Nv[i]);
        double jnR = m->mun[i+1] * m->Vt / hR *
                     (nbuf[i+1] * bern(dnR) - nbuf[i] * bern(-dnR));
        double jnL = m->mun[i] * m->Vt / hL *
                     (nbuf[i] * bern(dnL) - nbuf[i-1] * bern(-dnL));
        double src = rnet[i] - m->G[i];

        /* ---- 空穴连续 ---- */
        double jpR = m->mup[i+1] * m->Vt / hR *
                     (pbuf[i] * bern(dpR) - pbuf[i+1] * bern(-dpR));
        double jpL = m->mup[i] * m->Vt / hL *
                     (pbuf[i-1] * bern(dpL) - pbuf[i] * bern(-dpL));

        if (m->eq_minority) {
            /* 少数载流子退化时换成 φn=φp。两种触发条件：
             *  - 相对：比另一种低 6 个数量级以上；
             *  - 绝对：比本层 ni 还低 20 个数量级（深耗尽区两者都极小）。 */
            int e_deg = (nbuf[i] < 1e-6 * pbuf[i]) || (nbuf[i] < 1e-20 * m->ni[i]);
            int p_deg = (pbuf[i] < 1e-6 * nbuf[i]) || (pbuf[i] < 1e-20 * m->ni[i]);
            if (e_deg && p_deg) p_deg = 0;   /* 两条都换会变成同一个方程 */
            if (e_deg) r[n + i] = u[n + i] - u[2*n + i];
            if (p_deg) r[2 * n + i] = u[2*n + i] - u[n + i];
            if (!e_deg) {
                double den = nbuf[i] > 1e-280 ? nbuf[i] : 1e-280;
                r[n + i] = ((jnR - jnL) / hc - src) / den;
            }
            if (!p_deg) {
                double den = pbuf[i] > 1e-280 ? pbuf[i] : 1e-280;
                r[2 * n + i] = ((jpR - jpL) / hc + src) / den;
            }
        } else {
            double den_n = nbuf[i] > 1e-280 ? nbuf[i] : 1e-280;
            r[n + i] = ((jnR - jnL) / hc - src) / den_n;
            double den_p = pbuf[i] > 1e-280 ? pbuf[i] : 1e-280;
            r[2 * n + i] = ((jpR - jpL) / hc + src) / den_p;
        }

        {
            double sc = fabs(r[n+i]);
            if (fabs(r[2*n+i]) > sc) sc = fabs(r[2*n+i]);
            if (sc > rmax) rmax = sc;
            rsum2 += r[n+i]*r[n+i] + r[2*n+i]*r[2*n+i];
        }
    }
    if (!isfinite(rmax) || !isfinite(rsum2)) return 1e300;
    /* 2-范数：Newton 方向是它的下降方向（∞-范数不保证），线搜索用它 */
    return sqrt(rsum2 / (double)(3 * n));
}

/* ─────────────────────────────────────────────────────────────────────
 * 稠密 LU（部分选主元 + 行均衡）
 * ───────────────────────────────────────────────────────────────────── */
static int dense_solve(double* A, double* b, int nn) {
    for (int i = 0; i < nn; i++) {
        double* row = A + (size_t)i * nn;
        double s = 0.0;
        for (int j = 0; j < nn; j++) {
            double v = fabs(row[j]);
            if (v > s) s = v;
        }
        if (!(s > 1e-290) || !isfinite(s)) {
            if (getenv("PVK_DEBUG")) fprintf(stderr, "[pvk] row %d degenerate s=%g (freeze)\n", i, s);
            /* 整行全零：该方程在这点上不约束任何变量（载流子已到
             * 数值下溢区），把它换成 du_i = 0，即冻结该未知量。 */
            for (int j = 0; j < nn; j++) row[j] = 0.0;
            row[i] = 1.0;
            b[i] = 0.0;
            continue;
        }
        double inv = 1.0 / s;
        for (int j = 0; j < nn; j++) row[j] *= inv;
        b[i] *= inv;
    }

    /* 对角加一点正则：少数载流子在强掺杂层里处于 ~1e-40 量级时，对应的
     * 准费米未知量在残差里几乎不出现，矩阵数值上是奇异的。加 1e-10 的
     * 对角项（相对于行均衡后的 O(1) 尺度）只影响这些零空间方向，给出
     * 一个极小的步长，不改变可解方向。 */
    for (int i = 0; i < nn; i++) A[(size_t)i * nn + i] += 1e-10;

    for (int k = 0; k < nn; k++) {
        int piv = k;
        double best = fabs(A[(size_t)k * nn + k]);
        for (int i = k + 1; i < nn; i++) {
            double v = fabs(A[(size_t)i * nn + k]);
            if (v > best) { best = v; piv = i; }
        }
        if (!(best > 1e-14) || !isfinite(best)) {
            if (getenv("PVK_DEBUG")) fprintf(stderr, "[pvk] pivot k=%d best=%g\n", k, best);
            return -1;
        }
        if (piv != k) {
            double* ra = A + (size_t)k * nn;
            double* rb = A + (size_t)piv * nn;
            for (int j = 0; j < nn; j++) { double t = ra[j]; ra[j] = rb[j]; rb[j] = t; }
            double t = b[k]; b[k] = b[piv]; b[piv] = t;
        }
        double akk = A[(size_t)k * nn + k];
        double* rk = A + (size_t)k * nn;
        for (int i = k + 1; i < nn; i++) {
            double* ri = A + (size_t)i * nn;
            double f = ri[k] / akk;
            if (f == 0.0) continue;
            ri[k] = 0.0;
            for (int j = k + 1; j < nn; j++) ri[j] -= f * rk[j];
            b[i] -= f * b[k];
        }
    }

    for (int i = nn - 1; i >= 0; i--) {
        double s = b[i];
        double* ri = A + (size_t)i * nn;
        for (int j = i + 1; j < nn; j++) s -= ri[j] * b[j];
        b[i] = s / ri[i];
    }
    for (int i = 0; i < nn; i++) {
        if (!isfinite(b[i])) return -1;
    }
    return 0;
}

/* ─────────────────────────────────────────────────────────────────────
 * 模型构建 / 释放
 * ───────────────────────────────────────────────────────────────────── */
static void model_free(pvk_model* m) {
    if (!m) return;
    pvk__free(m->eps);  pvk__free(m->chi); pvk__free(m->Eg);
    pvk__free(m->Nc);   pvk__free(m->Nv);
    pvk__free(m->mun);  pvk__free(m->mup);
    pvk__free(m->taun); pvk__free(m->taup);
    pvk__free(m->Nd);   pvk__free(m->Na);
    pvk__free(m->G);    pvk__free(m->ni);
    pvk__free(m->chis); pvk__free(m->egs);
    pvk__free(m->hc);
    pvk__free(m->bc);
    memset(m, 0, sizeof(*m));
}

/* homo=0 时把某层参数从参考值线性拉到 homo=1 的真实值。
 * 只用于平衡态求解的带边 homotopy（见 equilibrium_guess）。 */
static double homo_lerp(double v, double ref, double t) {
    return ref + t * (v - ref);
}

static int model_init(pvk_ctx* ctx, const pvk_device* dev, pvk_model* m,
                      double V_app, int eq_minority, double homo) {
    memset(m, 0, sizeof(*m));
    m->eq_minority = eq_minority;
    int n = ctx->mesh.n_node;
    if (n < 3) return PVK_EAPI;

    m->n = n;
    m->Vt = PVK_KB_EV * (dev->T_K > 0 ? dev->T_K : 300.0);
    m->V_app = V_app;
    m->x = ctx->mesh.x;
    m->h = ctx->mesh.dx;
    m->node_layer = ctx->mesh.node_layer;

    double Vt = m->Vt;
    const pvk_layer* C = &dev->layers[0];
    const pvk_layer* A = &dev->layers[dev->n_layers - 1];

    /* 带边 homotopy 的参考点：窄带隙（吸收）层的 χ/Eg。
     * homo=1（正式求解）时下面所有插值都退化成真实参数。 */
    int iref = 0;
    for (int i = 1; i < dev->n_layers; i++)
        if (dev->layers[i].Eg_eV < dev->layers[iref].Eg_eV) iref = i;
    double chi_ref = dev->layers[iref].chi_eV;
    double eg_ref  = dev->layers[iref].Eg_eV;

    /* U_ref：让 x=0 处 ψ=0、φ=0 时满足阴极层的电中性。
     * homo 同时缩放掺杂：homo=0 时整个器件是本征、同一带结构，
     * 各层电中性电位全是 ψ=0（含两个电极的边界值），初值即精确解。 */
    double chi_C = homo_lerp(C->chi_eV, chi_ref, homo);
    double eg_C  = homo_lerp(C->Eg_eV,  eg_ref,  homo);
    double nc_C  = C->Nc_cm3 > 0 ? C->Nc_cm3 : 1.0;
    double nv_C  = C->Nv_cm3 > 0 ? C->Nv_cm3 : 1.0;
    double Nd_C  = homo * C->Nd_cm3;
    double Na_C  = homo * C->Na_cm3;
    double ni_c  = sqrt(nc_C * nv_C) * exp(-eg_C / (2.0 * Vt));
    if (Nd_C > 0.0 && Nd_C >= Na_C) {
        m->U_ref = Vt * log(Nd_C / nc_C) - chi_C;
    } else if (Na_C > 0.0) {
        m->U_ref = -chi_C - eg_C - Vt * log(Na_C / nv_C);
    } else {
        m->U_ref = Vt * log(ni_c / nc_C) - chi_C;
    }

    /* 阳极 ψ 的 Dirichlet 值（电中性 + 准费米劈裂）。
     * 注意电化学势符号：阳极静电势比阴极高 V_app 时，电子的准费米势
     * 下降 q·V_app —— 所以 φ = -V_app，ψ 上则是 +V_app。 */
    double chi_A = homo_lerp(A->chi_eV, chi_ref, homo);
    double eg_A  = homo_lerp(A->Eg_eV,  eg_ref,  homo);
    double Nd_A  = homo * A->Nd_cm3;
    double Na_A  = homo * A->Na_cm3;
    if (Na_A > 0.0 && Na_A >= Nd_A) {
        double nv = A->Nv_cm3 > 0 ? A->Nv_cm3 : 1.0;
        m->psi_a = -chi_A - eg_A - m->U_ref + V_app
                   - Vt * log(Na_A / nv);
    } else if (Nd_A > 0.0) {
        double nc = A->Nc_cm3 > 0 ? A->Nc_cm3 : 1.0;
        m->psi_a = Vt * log(Nd_A / nc) - chi_A - m->U_ref + V_app;
    } else {
        double nc = A->Nc_cm3 > 0 ? A->Nc_cm3 : 1.0;
        double nv = A->Nv_cm3 > 0 ? A->Nv_cm3 : 1.0;
        double ni_A = sqrt(nc * nv) * exp(-eg_A / (2.0 * Vt));
        m->psi_a = Vt * log(ni_A / nc) - chi_A - m->U_ref + V_app;
    }

    size_t dn = sizeof(double) * (size_t)n;
    m->eps  = (double*)pvk__malloc(dn);
    m->chi  = (double*)pvk__malloc(dn);
    m->Eg   = (double*)pvk__malloc(dn);
    m->Nc   = (double*)pvk__malloc(dn);
    m->Nv   = (double*)pvk__malloc(dn);
    m->mun  = (double*)pvk__malloc(dn);
    m->mup  = (double*)pvk__malloc(dn);
    m->taun = (double*)pvk__malloc(dn);
    m->taup = (double*)pvk__malloc(dn);
    m->Nd   = (double*)pvk__malloc(dn);
    m->Na   = (double*)pvk__malloc(dn);
    m->G    = (double*)pvk__malloc(dn);
    m->ni   = (double*)pvk__malloc(dn);
    m->chis = (double*)pvk__malloc(dn);
    m->egs  = (double*)pvk__malloc(dn);
    m->hc   = (double*)pvk__malloc(dn);
    m->bc   = (int*)pvk__malloc(sizeof(int) * (size_t)n);
    if (!m->eps || !m->chi || !m->Eg || !m->Nc || !m->Nv || !m->mun ||
        !m->mup || !m->taun || !m->taup || !m->Nd || !m->Na || !m->G ||
        !m->ni || !m->chis || !m->egs || !m->hc || !m->bc) {
        model_free(m);
        return PVK_EMEM;
    }

    const double* gopt = ctx->state.fields[PVK_FIELD_G_OPT];

    for (int i = 0; i < n; i++) {
        int li = ctx->mesh.node_layer[i];
        if (li < 0 || li >= dev->n_layers) li = 0;
        const pvk_layer* L = &dev->layers[li];

        double chi_i = homo_lerp(L->chi_eV, chi_ref, homo);
        double eg_i  = homo_lerp(L->Eg_eV,  eg_ref,  homo);

        m->eps[i]  = PVK_EPS0 * (L->eps_r > 0 ? L->eps_r : 1.0);
        m->chi[i]  = chi_i;
        m->Eg[i]   = eg_i;
        m->Nc[i]   = L->Nc_cm3 > 0 ? L->Nc_cm3 : 1.0;
        m->Nv[i]   = L->Nv_cm3 > 0 ? L->Nv_cm3 : 1.0;
        m->mun[i]  = L->mun_cm2Vs > 0 ? L->mun_cm2Vs : 1e-6;
        m->mup[i]  = L->mup_cm2Vs > 0 ? L->mup_cm2Vs : 1e-6;
        m->taun[i] = L->taun_s > 0 ? L->taun_s : 1e-12;
        m->taup[i] = L->taup_s > 0 ? L->taup_s : 1e-12;
        m->Nd[i]   = homo * L->Nd_cm3;
        m->Na[i]   = homo * L->Na_cm3;
        m->ni[i]   = sqrt(m->Nc[i] * m->Nv[i]) * exp(-eg_i / (2.0 * Vt));
        m->chis[i] = chi_i / Vt + m->U_ref / Vt;
        m->egs[i]  = eg_i / Vt;
        m->G[i]    = gopt ? fmax(gopt[i], 0.0) : 0.0;

        /* 电极层（is_contact）当作理想电极处理：整层钉在电极电位上，
         * 不参与有限载流子的漂移扩散求解。否则少数载流子会掉到
         * 1e-40 以下，连续性方程在数值上退化成零方程。 */
        if (i == 0) {
            m->bc[i] = 1;
        } else if (i == n - 1) {
            m->bc[i] = 2;
        } else if (dev->layers[li].is_contact) {
            m->bc[i] = (li * 2 < dev->n_layers) ? 1 : 2;
        } else {
            m->bc[i] = 0;
        }

        if (i == 0) {
            m->hc[i] = 0.5 * m->h[0];
        } else if (i == n - 1) {
            m->hc[i] = 0.5 * m->h[n-2];
        } else {
            m->hc[i] = 0.5 * (m->h[i-1] + m->h[i]);
        }
    }

    /* 理想 Ohmic 接触：整个电极层改用相邻半导体的 ε/Nc/Nv/χ/Eg
     * （电极只提供电位和费米能级）。
     *
     * 否则界面两侧带边会按电极材料（Au: χ=4.7, Eg=1.0；Spiro: χ=2.5,
     * Eg=3.0）算出 2.5 eV 的“能带阶跃”，细致平衡要求半导体一侧的
     * 载流子密度是 1e38 量级才可能零电流 —— 暗态平衡方程根本无解，
     * 紧邻电极的那个自由节点的连续性行就成了整个雅可比的奇异块。
     * 电极节点整层钉死，这些密度只用于界面边上算通量，不影响 BC 行；
     * 整层一起改是为了让电极内部的边也自洽（否则层内会出现虚假电流）。 */
    for (int i = 0; i + 1 < n; i++) {
        if (m->bc[i] == m->bc[i+1]) continue;
        int jsemi = m->bc[i] ? i + 1 : i;          /* 半导体侧节点 */
        int lcont = m->bc[i] ? m->node_layer[i] : m->node_layer[i+1];
        for (int k = 0; k < n; k++) {
            if (!m->bc[k] || m->node_layer[k] != lcont) continue;
            m->eps[k]  = m->eps[jsemi];
            m->chi[k]  = m->chi[jsemi];
            m->Eg[k]   = m->Eg[jsemi];
            m->Nc[k]   = m->Nc[jsemi];
            m->Nv[k]   = m->Nv[jsemi];
            m->ni[k]   = m->ni[jsemi];
            m->chis[k] = m->chis[jsemi];
            m->egs[k]  = m->egs[jsemi];
        }
    }

    m->u_bc[0]  = 0.0;
    m->u_bc[1]  = m->psi_a / Vt;
    m->wn_bc[0] = 0.0;
    m->wn_bc[1] = -V_app / Vt;
    m->wp_bc[0] = 0.0;
    m->wp_bc[1] = -V_app / Vt;

    return PVK_OK;
}

/* ─────────────────────────────────────────────────────────────────────
 * Newton
 * ───────────────────────────────────────────────────────────────────── */
static int solve_newton(pvk_model* m, double* u,
                        const pvk_solver_opts* opts, pvk_solve_info* out) {
    const int n = m->n;
    const int nn = 3 * n;

    double* r      = (double*)pvk__malloc(sizeof(double) * nn);
    double* rtmp   = (double*)pvk__malloc(sizeof(double) * nn);
    double* utmp   = (double*)pvk__malloc(sizeof(double) * nn);
    double* du     = (double*)pvk__malloc(sizeof(double) * nn);
    double* A      = (double*)pvk__malloc(sizeof(double) * (size_t)nn * nn);
    double* nbuf   = (double*)pvk__malloc(sizeof(double) * n);
    double* pbuf   = (double*)pvk__malloc(sizeof(double) * n);
    double* rnet   = (double*)pvk__malloc(sizeof(double) * n);
    double* nb2    = (double*)pvk__malloc(sizeof(double) * n);
    double* pb2    = (double*)pvk__malloc(sizeof(double) * n);
    double* rn2    = (double*)pvk__malloc(sizeof(double) * n);

    if (!r || !rtmp || !utmp || !du || !A || !nbuf || !pbuf || !rnet ||
        !nb2 || !pb2 || !rn2) {
        pvk__free(r); pvk__free(rtmp); pvk__free(utmp); pvk__free(du);
        pvk__free(A); pvk__free(nbuf); pvk__free(pbuf); pvk__free(rnet);
        pvk__free(nb2); pvk__free(pb2); pvk__free(rn2);
        return PVK_EMEM;
    }

    int maxit = opts && opts->max_iter > 0 ? opts->max_iter : 60;
    double tol = opts && opts->res_tol > 0 ? opts->res_tol : 1e-8;
    double alpha0 = opts && opts->damping > 0 ? opts->damping : 1.0;
    if (alpha0 > 1.0) alpha0 = 1.0;

    double rn = residual(m, u, r, nbuf, pbuf, rnet);
    int converged = 0;
    int iters = 0;
    int status = PVK_OK;

    for (int it = 0; it < maxit; it++) {
        if (rn < tol) { converged = 1; break; }

        /* ---- 数值雅可比（前向差分） ---- */
        for (int j = 0; j < nn; j++) {
            double save = u[j];
            double hstep = 1e-7 * (fabs(save) > 1.0 ? fabs(save) : 1.0);
            u[j] = save + hstep;
            residual(m, u, rtmp, nb2, pb2, rn2);
            u[j] = save;
            double inv = 1.0 / hstep;
            for (int i = 0; i < nn; i++) {
                A[(size_t)i * nn + j] = (rtmp[i] - r[i]) * inv;
            }
        }

        for (int i = 0; i < nn; i++) du[i] = -r[i];

        if (dense_solve(A, du, nn) != 0) {
            if (getenv("PVK_DEBUG")) fprintf(stderr, "[pvk] dense_solve failed at it=%d\n", it);
            status = PVK_ECONV;
            break;
        }

        double dumax = 0.0;
        for (int i = 0; i < nn; i++) {
            /* 步长限幅：病态方向上 Newton 步可能爆到 1e9，先截到 40（≈1 V） */
            if (du[i] > 40.0) du[i] = 40.0;
            if (du[i] < -40.0) du[i] = -40.0;
            double v = fabs(du[i]);
            if (v > dumax) dumax = v;
        }

        /* ---- 回溯线搜索 ---- */
        double alpha = alpha0;
        double rn_new = rn;
        int accepted = 0;
        for (int t = 0; t < 16; t++) {
            for (int i = 0; i < nn; i++) utmp[i] = u[i] + alpha * du[i];
            rn_new = residual(m, utmp, rtmp, nb2, pb2, rn2);
            if (isfinite(rn_new) && rn_new <= rn * (1.0 - 1e-4 * alpha)) {
                accepted = 1;
                break;
            }
            if (isfinite(rn_new) && rn_new < tol) { accepted = 1; break; }
            alpha *= 0.5;
        }
        if (!accepted) {
            if (getenv("PVK_DEBUG")) fprintf(stderr, "[pvk] line search failed at it=%d rn=%.3e\n", it, rn);
            status = PVK_ECONV;
            break;
        }

        memcpy(u, utmp, sizeof(double) * nn);
        memcpy(r, rtmp, sizeof(double) * nn);
        rn = rn_new;
        iters = it + 1;

        if (alpha * dumax < 1e-7) { converged = 1; break; }
    }

    out->iterations = iters;
    out->residual = rn;
    out->converged = converged ? 1 : 0;

    pvk__free(r); pvk__free(rtmp); pvk__free(utmp); pvk__free(du);
    pvk__free(A); pvk__free(nbuf); pvk__free(pbuf); pvk__free(rnet);
    pvk__free(nb2); pvk__free(pb2); pvk__free(rn2);
    return status;
}

/* ─────────────────────────────────────────────────────────────────────
 * 平衡态初值：V=0、无光、φn=φp=0，只解非线性 Poisson
 *
 * 未知量只有 ψ/Vt（n 个），没有连续性方程。它本身仍是强非线性的
 * （电荷按 exp(ψ/Vt) 变），所以不能直接一步 Newton 从电中性解到真解，
 * 要沿带边 homotopy 走（见 equilibrium_guess）。
 * ───────────────────────────────────────────────────────────────────── */
static void eq_dens(const pvk_model* m, int i, double u, double* nout, double* pout) {
    double en = clampd(u + m->chis[i], -280.0, 280.0);
    double ep = clampd(-(u + m->chis[i] + m->egs[i]), -280.0, 280.0);
    *nout = m->Nc[i] * exp(en);
    *pout = m->Nv[i] * exp(ep);
}

static double eq_residual(const pvk_model* m, const double* u, double* r,
                          double* nb, double* pb) {
    int n = m->n;
    double Vt = m->Vt;
    double rsum2 = 0.0;

    for (int i = 0; i < n; i++) {
        eq_dens(m, i, u[i], &nb[i], &pb[i]);
        if (!isfinite(nb[i])) nb[i] = 1e300;
        if (!isfinite(pb[i])) pb[i] = 1e300;
    }

    for (int i = 0; i < n; i++) {
        if (m->bc[i]) {
            r[i] = u[i] - m->u_bc[m->bc[i] - 1];
            rsum2 += r[i] * r[i];
            continue;
        }
        double hL = m->h[i-1], hR = m->h[i], hc = m->hc[i];
        double fluxR = m->eps[i+1] * (u[i+1] - u[i]) * Vt / hR;
        double fluxL = m->eps[i]   * (u[i]   - u[i-1]) * Vt / hL;
        double ft = (fluxR - fluxL) / (PVK_Q_C * hc);
        double ct = pb[i] - nb[i] + m->Nd[i] - m->Na[i];
        r[i] = ft + ct;
        double den = fabs(ft);
        if (fabs(ct) > den) den = fabs(ct);
        if (den < 1.0) den = 1.0;
        double a = r[i] / den;
        rsum2 += a * a;
    }
    if (!isfinite(rsum2)) return 1e300;
    return sqrt(rsum2 / (double)n);
}

/* 电中性初值：按层解 A·e^u - B·e^-u = Nd - Na */
static void eq_neutral_guess(const pvk_model* m, double* u) {
    int n = m->n;
    for (int i = 0; i < n; i++) {
        if (m->bc[i]) {
            u[i] = m->u_bc[m->bc[i] - 1];
            continue;
        }
        double A_ = m->Nc[i] * exp(clampd(m->chis[i], -280.0, 280.0));
        double B_ = m->Nv[i] * exp(clampd(-(m->chis[i] + m->egs[i]), -280.0, 280.0));
        double D = m->Nd[i] - m->Na[i];
        double disc = D * D + 4.0 * A_ * B_;
        double z = 1.0;
        if (disc > 0.0 && A_ > 0.0) {
            double sq = sqrt(disc);
            z = (D >= 0.0) ? (D + sq) / (2.0 * A_) : (2.0 * B_) / (sq - D);
        }
        if (!(z > 0.0) || !isfinite(z)) z = 1.0;
        u[i] = clampd(log(z), -150.0, 150.0);
    }
}

/* 从给定初值解非线性 Poisson（只解 ψ，φn=φp=0），返回最终归一化残差。 */
static double solve_poisson_newton(const pvk_model* m, double* u,
                                   const pvk_solver_opts* opts) {
    int n = m->n;
    double rn = 1e300;
    double* r  = (double*)pvk__malloc(sizeof(double) * n);
    double* rt = (double*)pvk__malloc(sizeof(double) * n);
    double* ut = (double*)pvk__malloc(sizeof(double) * n);
    double* nb = (double*)pvk__malloc(sizeof(double) * n);
    double* pb = (double*)pvk__malloc(sizeof(double) * n);
    double* nb2 = (double*)pvk__malloc(sizeof(double) * n);
    double* pb2 = (double*)pvk__malloc(sizeof(double) * n);
    double* A  = (double*)pvk__malloc(sizeof(double) * (size_t)n * n);
    double* du = (double*)pvk__malloc(sizeof(double) * n);
    if (!r || !rt || !ut || !nb || !pb || !nb2 || !pb2 || !A || !du) goto done_free;

    double tol = opts && opts->res_tol > 0 ? opts->res_tol : 1e-8;
    rn = eq_residual(m, u, r, nb, pb);

    for (int it = 0; it < 80 && rn > tol; it++) {
        for (int j = 0; j < n; j++) {
            double save = u[j];
            double hstep = 1e-7 * (fabs(save) > 1.0 ? fabs(save) : 1.0);
            u[j] = save + hstep;
            eq_residual(m, u, rt, nb2, pb2);
            u[j] = save;
            double inv = 1.0 / hstep;
            for (int i = 0; i < n; i++) A[(size_t)i * n + j] = (rt[i] - r[i]) * inv;
        }
        for (int i = 0; i < n; i++) du[i] = -r[i];
        if (dense_solve(A, du, n) != 0) {
            if (getenv("PVK_DEBUG")) fprintf(stderr, "[pvk] eq: dense_solve failed it=%d\n", it);
            break;
        }

        /* 步长限幅（≈0.5 V），避免指数型电荷让第一个 Newton 步爆掉 */
        double dumax = 0.0;
        for (int i = 0; i < n; i++) {
            if (m->bc[i]) continue;
            if (du[i] > 20.0) du[i] = 20.0;
            if (du[i] < -20.0) du[i] = -20.0;
            if (fabs(du[i]) > dumax) dumax = fabs(du[i]);
        }

        double alpha = 1.0, rnew = rn;
        int ok = 0;
        for (int t = 0; t < 16; t++) {
            for (int i = 0; i < n; i++) ut[i] = u[i] + alpha * du[i];
            rnew = eq_residual(m, ut, rt, nb2, pb2);
            if (isfinite(rnew) && rnew <= rn * (1.0 - 1e-4 * alpha)) { ok = 1; break; }
            alpha *= 0.5;
        }
        if (!ok) {
            if (getenv("PVK_DEBUG")) fprintf(stderr, "[pvk] eq: line search failed it=%d rn=%.3e\n", it, rn);
            break;
        }
        memcpy(u, ut, sizeof(double) * n);
        for (int i = 0; i < n; i++) r[i] = rt[i];
        rn = rnew;
        /* 注意：不能用 du[0] 判停 —— 节点 0 是钉死的 Dirichlet 行，
         * du[0] 恒为 0，会让平衡态 Newton 只走一步（这正是暗态初值
         * 一直不收敛的根因之一）。用自由节点的最大步长。 */
        if (alpha * dumax < 1e-10) break;
    }
    if (getenv("PVK_DEBUG"))
        fprintf(stderr, "[pvk] solve_equilibrium: rn=%.3e\n", rn);

done_free:
    pvk__free(r); pvk__free(rt); pvk__free(ut);
    pvk__free(nb); pvk__free(pb); pvk__free(nb2); pvk__free(pb2);
    pvk__free(A); pvk__free(du);
    return rn;
}

/* 平衡态 ψ：沿“带边 homotopy”解非线性 Poisson。
 *
 * 直接拿各层电中性当起点时，层与层之间 ψ 要跳 0.9~1.4 V（TiO2/Spiro
 * 与吸收层的带边差），Poisson 在界面节点的残差是电荷项的 1e20 量级，
 * 指数型电荷让 Newton 第一步就踏进不可解区、线搜索必然失败。
 *
 * 做法：homo=0 时把所有层的 χ/Eg 拉平到吸收层的值 —— 这时各层电中性
 * 算出的 ψ 完全相同，初值就是精确解，Poisson 一步收敛；再把带边差按
 * homo 逐步打开，每步用上一步的 ψ 作起点。带边差是解析参数，解随它
 * 连续演化，最后 homo=1 落回真实参数。 */
static void equilibrium_guess(pvk_ctx* ctx, const pvk_device* dev,
                              pvk_model* m, double* u, const pvk_solver_opts* opts) {
    /* 掺杂按数量级变化（1e20 → 5e19 …）比带边差（~1 eV）陡得多，
     * 所以起点附近的对数刻度要密。每步若解不动就把步长折半重试。 */
    static const double hs[] = {0.0, 1e-6, 1e-4, 1e-3, 0.01, 0.05,
                                0.15, 0.35, 0.6, 0.8, 1.0};
    const int nh = (int)(sizeof(hs) / sizeof(hs[0]));
    const int n  = m->n;
    pvk_model lm;
    memset(&lm, 0, sizeof(lm));

    double* usave = (double*)pvk__malloc(sizeof(double) * n);
    int did = 0;
    double t_done = 0.0;

    for (int k = 0; k < nh && usave; k++) {
        double t_try = hs[k];
        int ok = 0;
        for (int tries = 0; tries < 12; tries++) {
            if (model_init(ctx, dev, &lm, m->V_app, 0, t_try) != PVK_OK) break;
            if (did) memcpy(u, usave, sizeof(double) * n);
            else     eq_neutral_guess(&lm, u);
            /* psi_a 随 homo 变，钉死节点每次重试都重置到本步边界值 */
            for (int i = 0; i < n; i++)
                if (lm.bc[i]) u[i] = lm.u_bc[lm.bc[i] - 1];

            double rn = solve_poisson_newton(&lm, u, opts);
            model_free(&lm);

            /* 这里是“初值的初值”，不要求解到收敛：残差降到 1e-3 就够，
             * 让 homotopy 能一路走到 homo=1（真实参数），剩下的精度
             * 交给完整 3n 方程的 Newton。 */
            if (rn < 1e-3) {
                ok = 1;
                did = 1;
                t_done = t_try;
                memcpy(usave, u, sizeof(double) * n);
                break;
            }
            /* 失败：在上一个成功点与当前点之间折半 */
            t_try = 0.5 * (t_done + t_try);
            if (t_try - t_done < 1e-8 * (1.0 - t_done)) break;
        }
        if (!ok) break;
    }
    if (!did) eq_neutral_guess(m, u);
    pvk__free(usave);
    if (getenv("PVK_DEBUG"))
        fprintf(stderr, "[pvk] equilibrium homotopy: t=%.4g done=%d\n", t_done, did);

    /* φn=φp=0（钉死节点取边界值，否则 V_app≠0 时会把阳极的
     * 准费米劈裂抹掉）。 */
    for (int i = 0; i < n; i++) {
        int e = m->bc[i] ? m->bc[i] - 1 : -1;
        u[n + i]   = (e >= 0) ? m->wn_bc[e] : 0.0;
        u[2*n + i] = (e >= 0) ? m->wp_bc[e] : 0.0;
    }
}

/* ─────────────────────────────────────────────────────────────────────
 * 初始猜测
 * ───────────────────────────────────────────────────────────────────── */
static void initial_guess(pvk_ctx* ctx, const pvk_device* dev, pvk_model* m,
                          double* u, int warm) {
    const int n = m->n;
    double Vt = m->Vt;
    double L = m->x[n-1] > 0 ? m->x[n-1] : 1.0;

    if (warm) {
        const double* ppsi = ctx->state.fields[PVK_FIELD_PSI];
        const double* pfin = ctx->state.fields[PVK_FIELD_PHIN];
        const double* pfp  = ctx->state.fields[PVK_FIELD_PHIP];
        if (ppsi && pfin && pfp) {
            double dV = m->V_app - ctx->last_V_applied;
            for (int i = 0; i < n; i++) {
                if (m->bc[i]) continue;
                double s = m->x[i] / L;
                u[i]         = ppsi[i] / Vt + dV * s / Vt;
                u[n + i]     = (pfin[i] - m->U_ref) / Vt - dV * s / Vt;
                u[2*n + i]   = (pfp[i]  - m->U_ref) / Vt - dV * s / Vt;
            }
            for (int i = 0; i < n; i++) {
                if (!m->bc[i]) continue;
                int e = m->bc[i] - 1;
                u[i]       = m->u_bc[e];
                u[n + i]   = m->wn_bc[e];
                u[2*n + i] = m->wp_bc[e];
            }
            return;
        }
    }

    /* 冷启动：先用非线性 Poisson 解出平衡态 ψ（带边 homotopy），
     * 再令 φn=φp=0。局域电中性初值在有 ~1 V 内建电势的器件里离真解
     * 差几十个 Vt，暗态（无光生载流子抬高密度）Newton 一步就能把残差
     * 推到 1e25 以上。 */
    equilibrium_guess(ctx, dev, m, u, &ctx->solver_opts);
    /* 电极层所有节点钉死 */
    for (int i = 0; i < n; i++) {
        if (!m->bc[i]) continue;
        int e = m->bc[i] - 1;
        u[i]         = m->u_bc[e];
        u[n + i]     = m->wn_bc[e];
        u[2*n + i]   = m->wp_bc[e];
    }
}

/* ─────────────────────────────────────────────────────────────────────
 * 场量回填
 * ───────────────────────────────────────────────────────────────────── */
static void fill_fields(pvk_ctx* ctx, pvk_model* m, const double* u,
                        double* J_out) {
    const int n = m->n;
    double Vt = m->Vt;

    double* psi  = ctx->state.fields[PVK_FIELD_PSI];
    double* phin = ctx->state.fields[PVK_FIELD_PHIN];
    double* phip = ctx->state.fields[PVK_FIELD_PHIP];
    double* nd   = ctx->state.fields[PVK_FIELD_N];
    double* pd   = ctx->state.fields[PVK_FIELD_P];
    double* ef   = ctx->state.fields[PVK_FIELD_EFIELD];
    double* jn   = ctx->state.fields[PVK_FIELD_JN];
    double* jp   = ctx->state.fields[PVK_FIELD_JP];
    double* jt   = ctx->state.fields[PVK_FIELD_JTOTAL];
    double* rnet = ctx->state.fields[PVK_FIELD_R_NET];
    double* cb   = ctx->state.fields[PVK_FIELD_CB];
    double* vb   = ctx->state.fields[PVK_FIELD_VB];
    double* xf   = ctx->state.fields[PVK_FIELD_X];

    double* nbuf  = (double*)pvk__malloc(sizeof(double) * n);
    double* pbuf  = (double*)pvk__malloc(sizeof(double) * n);
    double* eedge = (double*)pvk__malloc(sizeof(double) * (n - 1));
    double* jne   = (double*)pvk__malloc(sizeof(double) * (n - 1));
    double* jpe   = (double*)pvk__malloc(sizeof(double) * (n - 1));

    if (!nbuf || !pbuf || !eedge || !jne || !jpe) {
        pvk__free(nbuf); pvk__free(pbuf); pvk__free(eedge);
        pvk__free(jne); pvk__free(jpe);
        if (J_out) *J_out = 0.0;
        return;
    }

    for (int i = 0; i < n; i++) {
        node_dens(m, i, u, &nbuf[i], &pbuf[i]);
        double psi_i = u[i] * Vt;
        if (psi)  psi[i]  = psi_i;
        if (phin) phin[i] = m->U_ref + u[n + i] * Vt;
        if (phip) phip[i] = m->U_ref + u[2*n + i] * Vt;
        if (nd)   nd[i]   = nbuf[i];
        if (pd)   pd[i]   = pbuf[i];
        if (cb)   cb[i]   = -psi_i - m->chi[i];
        if (vb)   vb[i]   = -psi_i - m->chi[i] - m->Eg[i];
        if (xf)   xf[i]   = m->x[i] * 1e7;   /* nm */
        if (rnet) {
            double ni = m->ni[i];
            double np = nbuf[i] * pbuf[i];
            double den = m->taup[i] * (nbuf[i] + ni) + m->taun[i] * (pbuf[i] + ni);
            rnet[i] = (den > 0 ? (np - ni * ni) / den : 0.0) - m->G[i];
        }
    }

    /* 边上的电场与电流；对外符号取“阳极→阴极”（-x）。
     * Bernoulli 的自变量与 residual() 中一致：用广义带边差（含 Δχ、ΔEg
     * 和 lnNc），否则界面处算出的电流和实际求解的方程不是同一个量。 */
    for (int i = 0; i < n - 1; i++) {
        double hR = m->h[i];
        double duR = u[i+1] - u[i];
        double dn  = duR + (m->chis[i+1] - m->chis[i]) + log(m->Nc[i+1] / m->Nc[i]);
        double dp  = duR + (m->chis[i+1] - m->chis[i])
                     + (m->egs[i+1] - m->egs[i]) + log(m->Nv[i] / m->Nv[i+1]);
        eedge[i] = -duR * Vt / hR;
        double jn_e = m->mun[i+1] * Vt / hR *
                      (nbuf[i+1] * bern(dn) - nbuf[i] * bern(-dn));
        double jp_e = m->mup[i+1] * Vt / hR *
                      (pbuf[i] * bern(dp) - pbuf[i+1] * bern(-dp));
        jne[i] = -jn_e * PVK_Q_C * 1000.0;   /* mA/cm² */
        jpe[i] = -jp_e * PVK_Q_C * 1000.0;
    }

    double jsum = 0.0;
    for (int i = 0; i < n - 1; i++) jsum += jne[i] + jpe[i];
    if (J_out) *J_out = (n > 1) ? jsum / (double)(n - 1) : 0.0;

    for (int i = 0; i < n; i++) {
        int eL = (i > 0) ? i - 1 : 0;
        int eR = (i < n - 1) ? i : n - 2;
        if (ef) ef[i] = 0.5 * (eedge[eL] + eedge[eR]);
        if (jn) jn[i] = 0.5 * (jne[eL] + jne[eR]);
        if (jp) jp[i] = 0.5 * (jpe[eL] + jpe[eR]);
        if (jt) jt[i] = 0.5 * ((jne[eL] + jpe[eL]) + (jne[eR] + jpe[eR]));
    }

    pvk__free(nbuf); pvk__free(pbuf); pvk__free(eedge);
    pvk__free(jne); pvk__free(jpe);
}

/* ─────────────────────────────────────────────────────────────────────
 * mesh 复用判断
 * ───────────────────────────────────────────────────────────────────── */
static int mesh_matches(const pvk_ctx* ctx, const pvk_device* dev) {
    if (!ctx->mesh.sig_valid || ctx->mesh.n_node < 3) return 0;
    if (ctx->mesh.sig_n_layers != dev->n_layers) return 0;
    if (ctx->mesh.sig_nodes != dev->nodes) return 0;
    double total = 0.0;
    for (int i = 0; i < dev->n_layers; i++) total += dev->layers[i].thick_nm;
    return fabs(total - ctx->mesh.sig_total_nm) < 1e-9;
}

/* ─────────────────────────────────────────────────────────────────────
 * solver.h exports
 * ───────────────────────────────────────────────────────────────────── */

PVK_API int pvk_set_solver_opts(pvk_ctx* ctx, const pvk_solver_opts* opts) {
    if (!ctx || !opts) return PVK_EINVAL;
    ctx->solver_opts = *opts;
    pvk__log(ctx, PVK_LOG_INFO, "pvk_set_solver_opts applied");
    return PVK_OK;
}

/* 内部：复制 device/optics/bias 到 ctx（避免 Python 侧 free 掉后悬空指针） */
static void pvk__cache_inputs(pvk_ctx* ctx,
                              const pvk_device* dev,
                              const pvk_optics_in* opt,
                              const pvk_bias* bias) {
    if (!ctx) return;

    pvk_device cached = *dev;
    cached.layers = NULL;
    int n = dev->n_layers;
    if (n > 0) {
        pvk_layer* arr = (pvk_layer*)pvk__malloc(sizeof(pvk_layer) * n);
        if (arr) {
            for (int i = 0; i < n; i++) {
                arr[i] = dev->layers[i];
                arr[i].name     = pvk__strdup(dev->layers[i].name);
                arr[i].material = pvk__strdup(dev->layers[i].material);
            }
            cached.layers = arr;
        }
    }
    pvk__release_cached_device(ctx);
    ctx->last_device = cached;

    /* wl_nm/flux 借用调用方的指针，不 free；nk_dir 要 strdup */
    pvk_optics_in oc = *opt;
    oc.nk_dir = pvk__strdup(opt->nk_dir);
    pvk__free((void*)ctx->last_optics.nk_dir);
    ctx->last_optics = oc;

    ctx->last_bias = *bias;
    ctx->inputs_valid = 1;
}

/* 把当前解写回 PSI/PHIN/PHIP 场，供下一步 continuation 当 warm start。 */
static void store_guess(pvk_ctx* ctx, const pvk_model* m, const double* u) {
    int n = m->n;
    double* psi  = ctx->state.fields[PVK_FIELD_PSI];
    double* phin = ctx->state.fields[PVK_FIELD_PHIN];
    double* phip = ctx->state.fields[PVK_FIELD_PHIP];
    if (!psi || !phin || !phip) return;
    for (int i = 0; i < n; i++) {
        psi[i]  = u[i] * m->Vt;
        phin[i] = m->U_ref + u[n + i] * m->Vt;
        phip[i] = m->U_ref + u[2*n + i] * m->Vt;
    }
}

PVK_API int pvk_solve_dc(pvk_ctx* ctx,
                         const pvk_device* dev,
                         const pvk_optics_in* opt,
                         const pvk_bias* bias,
                         pvk_solve_info* out_info) {
    if (!ctx || !dev || !opt || !bias || !out_info) return PVK_EINVAL;
    if (dev->n_layers <= 0 || dev->layers == NULL) return PVK_EINVAL;

    if (!mesh_matches(ctx, dev)) {
        int rc = pvk_build_mesh(ctx, dev);
        if (rc != PVK_OK) return rc;
    }
    pvk__cache_inputs(ctx, dev, opt, bias);

    /* 光学：每次重算产生率剖面（intensity=0 时 G 全零） */
    if (opt->n_wl > 0 && opt->wl_nm && opt->flux) {
        int rc = pvk_optics_compute(ctx, dev, opt, NULL, NULL);
        if (rc != PVK_OK) return rc;
    } else {
        double* g = ctx->state.fields[PVK_FIELD_G_OPT];
        if (g) memset(g, 0, sizeof(double) * ctx->mesh.n_node);
    }

    int n = ctx->mesh.n_node;
    double V_app = bias->anode_V - bias->cathode_V;

    double* u = (double*)pvk__malloc(sizeof(double) * 3 * n);
    if (!u) return PVK_EMEM;

    /* 上一次收敛解的快照：本次若不收敛，用它恢复场量，
     * 避免把失败解留给下一个偏压点当 warm start。 */
    double* snap = NULL;
    if (ctx->have_solution) {
        const double* a = ctx->state.fields[PVK_FIELD_PSI];
        const double* b = ctx->state.fields[PVK_FIELD_PHIN];
        const double* c = ctx->state.fields[PVK_FIELD_PHIP];
        if (a && b && c) {
            snap = (double*)pvk__malloc(sizeof(double) * 3 * n);
            if (snap) {
                memcpy(snap,     a, sizeof(double) * n);
                memcpy(snap + n, b, sizeof(double) * n);
                memcpy(snap + 2*n, c, sizeof(double) * n);
            }
        }
    }

    pvk_model m;
    memset(&m, 0, sizeof(m));

    /* 冷启动：依次在几个小正偏压下试，第一个收敛的作为起点，再从它
     * continuation 到目标偏压。
     *
     * 初值来自 equilibrium_guess（沿带边+掺杂 homotopy 解出的平衡态 ψ，
     * φn=φp=0），这个初值本身已经满足 Poisson，正常器件直接解就收敛。
     * 只有在直接解失败时才退到产生率 homotopy：
     *  - 亮态：光强 0.01→1 爬坡；
     *  - 暗态（G≡0）：没有光可爬坡，改用吸收层里的合成产生率
     *    1e20 cm⁻³s⁻¹ 按十进位压到 0（界面少数载流子会塌到 1e-30，
     *    连续性行退化成 0=0，高注入能把密度抬回可解区）。
     * 每一步都先解完整双载流子方程，不收敛才用 φ_min=φ_maj 兜底。 */
    if (!ctx->have_solution) {
        static const double seeds[] = {0.15, 0.25, 0.35, 0.10, 0.05};
        static const double ramp[]  = {0.01, 0.03, 0.1, 0.25, 0.45, 0.7, 1.0};
        static const double daux[]  = {1.0, 0.3, 0.1, 0.03, 0.01, 3e-3, 1e-3,
                                       3e-4, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-10,
                                       1e-12, 0.0};
        const int nseed = (int)(sizeof(seeds) / sizeof(seeds[0]));
        const int nramp = (int)(sizeof(ramp) / sizeof(ramp[0]));
        const int ndaux = (int)(sizeof(daux) / sizeof(daux[0]));
        int seeded = 0;

        int dark = 1;
        for (int i = 0; i < n; i++) if (ctx->state.fields[PVK_FIELD_G_OPT] &&
                                       ctx->state.fields[PVK_FIELD_G_OPT][i] > 0.0) dark = 0;

        for (int si = 0; si < nseed && !seeded; si++) {
            double V_seed = seeds[si];

            for (int md = 0; md < 2 && !seeded; md++) {
                model_free(&m);
                memset(&m, 0, sizeof(m));
                if (model_init(ctx, dev, &m, V_seed, md, 1.0) != PVK_OK) break;

                initial_guess(ctx, dev, &m, u, 0);

                const double* gr = ctx->state.fields[PVK_FIELD_G_OPT];
                double* ubak = (double*)pvk__malloc(sizeof(double) * 3 * n);
                if (!ubak) break;
                memcpy(ubak, u, sizeof(double) * 3 * n);

                pvk_solve_info rinfo;
                int ok;

                /* 1) 直接用真实产生率解一次（暗态就是 G≡0）。初值来自
                 *    平衡态 homotopy，正常器件这一步就收敛。 */
                for (int i = 0; i < n; i++) m.G[i] = gr ? gr[i] : 0.0;
                memset(&rinfo, 0, sizeof(rinfo));
                rinfo.n = n;
                solve_newton(&m, u, &ctx->solver_opts, &rinfo);
                ok = rinfo.converged;
                if (!ok && rinfo.residual < 1e-3) ok = 1;
                if (getenv("PVK_DEBUG"))
                    fprintf(stderr, "[pvk] seed V=%.2f direct mode=%d conv=%d it=%d r=%.2e\n",
                            V_seed, md, rinfo.converged, rinfo.iterations, rinfo.residual);

                /* 2) 直接解不行才走产生率 homotopy 兜底。暗态没有光可爬坡，
                 *    改用吸收层里的合成产生率 1e20 逐十倍压到 0。 */
                if (!ok) {
                    memcpy(u, ubak, sizeof(double) * 3 * n);
                    double* gs = (double*)pvk__malloc(sizeof(double) * n);
                    if (!gs) { pvk__free(ubak); break; }
                    if (dark) {
                        int ia = 0;
                        for (int i = 1; i < n; i++)
                            if (m.Eg[i] < m.Eg[ia]) ia = i;
                        int la = m.node_layer[ia];
                        for (int i = 0; i < n; i++)
                            gs[i] = (m.node_layer[i] == la) ? 1e20 : 0.0;
                    } else {
                        for (int i = 0; i < n; i++) gs[i] = gr ? gr[i] : 0.0;
                    }
                    for (int k = 0; k < (dark ? ndaux : nramp); k++) {
                        for (int i = 0; i < n; i++)
                            m.G[i] = gs[i] * (dark ? daux[k] : ramp[k]);
                        memset(&rinfo, 0, sizeof(rinfo));
                        rinfo.n = n;
                        solve_newton(&m, u, &ctx->solver_opts, &rinfo);
                        ok = rinfo.converged;
                        if (getenv("PVK_DEBUG"))
                            fprintf(stderr, "[pvk] seed V=%.2f G*%.3f mode=%d conv=%d it=%d r=%.2e\n",
                                    V_seed, dark ? daux[k] : ramp[k], md, rinfo.converged,
                                    rinfo.iterations, rinfo.residual);
                        if (!ok) {
                            /* 残差已经很小但再压不动：作为 seed 足够 */
                            if (rinfo.residual < 1e-3) ok = 1;
                            break;
                        }
                    }
                    pvk__free(gs);
                    /* 恢复真实产生率（暗态下是 0，不能把合成 1e20 留下） */
                    for (int i = 0; i < n; i++) m.G[i] = gr ? gr[i] : 0.0;
                }
                pvk__free(ubak);

                if (ok) {
                    store_guess(ctx, &m, u);
                    ctx->have_solution = 1;
                    ctx->last_V_applied = V_seed;
                    ctx->last_J = 0.0;
                    seeded = 1;
                }
            }
        }

        model_free(&m);
        memset(&m, 0, sizeof(m));
    }

    int warm0 = ctx->have_solution;

    double V_prev = ctx->last_V_applied;

    pvk_solve_info info;
    memset(&info, 0, sizeof(info));
    info.n = n;
    info.v_applied = V_app;

    /* 自适应电压 continuation：从当前电压按步长走向目标，失败就把步长
     * 减半重试，成功后逐步放大（上限 0.25 V）。这样单点直接跳到 1.8 V
     * 也能走稳，而正常 0.05 V 扫描时只会走一步。 */
    double V_cur = V_prev;
    double step  = (V_app >= V_cur) ? 0.2 : -0.2;
    int done = 0;
    pvk_solve_info sub;
    memset(&sub, 0, sizeof(sub));
    sub.n = n;
    sub.v_applied = V_app;

    for (int attempt = 1; attempt <= 400 && !done; attempt++) {
        double Vk;
        if (fabs(V_app - V_cur) < 1e-9) {
            Vk = V_app;
        } else {
            Vk = V_cur + step;
            /* 不越过目标 */
            if ((V_app - Vk) * step <= 0.0) Vk = V_app;
            if (fabs(Vk - V_cur) < 1e-12) Vk = V_app;
        }

        /* 每个偏压点都先解完整双载流子方程；只有它不收敛时才退到
         * φ_min=φ_maj 的兜底（深耗尽区少数载流子掉到 1e-30 以下，
         * 那几行在数值上已经是 0=0）。不因为上一个点用了兜底就一直用。 */
        int good = 0;

        for (int md = 0; md < 2 && !good; md++) {

            model_free(&m);
            memset(&m, 0, sizeof(m));
            int rc2 = model_init(ctx, dev, &m, Vk, md, 1.0);
            if (rc2 != PVK_OK) {
                model_free(&m);
                pvk__free(u);
                pvk__free(snap);
                return rc2;
            }

            initial_guess(ctx, dev, &m, u, (attempt == 1 && md == 0) ? warm0 : 1);

            pvk_solve_info trial;
            memset(&trial, 0, sizeof(trial));
            trial.n = n;
            solve_newton(&m, u, &ctx->solver_opts, &trial);

            int ok = trial.converged;
            if (ok) {
                for (int i = 0; i < 3 * n; i++) {
                    if (!isfinite(u[i])) { ok = 0; break; }
                }
            }
            if (ok) {
                sub = trial;
                sub.n = n;
                sub.v_applied = V_app;
                good = 1;
                if (getenv("PVK_DEBUG"))
                    fprintf(stderr, "[pvk] V=%.3f mode=%d ok it=%d r=%.2e\n",
                            Vk, md, trial.iterations, trial.residual);
            }
        }

        if (!good) {
            step *= 0.5;
            if (fabs(step) < 0.005) break;
            continue;
        }

        V_cur = Vk;
        if (fabs(V_app - Vk) < 1e-9) {
            done = 1;
        } else {
            store_guess(ctx, &m, u);
            ctx->have_solution = 1;
            ctx->last_V_applied = Vk;
            /* 逐步放大步长，但不超过 0.25 V */
            if (fabs(step) < 0.25) {
                step *= 1.5;
                if (fabs(step) > 0.25) step = (step > 0 ? 0.25 : -0.25);
            }
        }
    }

    info = sub;
    info.n = n;
    info.v_applied = V_app;
    if (!done) info.converged = 0;

    double J = 0.0;
    if (info.converged) {
        fill_fields(ctx, &m, u, &J);
        if (!isfinite(J) || fabs(J) > 1e5) {
            /* 解出来的电流不物理（发散状态但残差判据被绕过），按失败处理 */
            info.converged = 0;
            J = NAN;
        } else {
            ctx->have_solution = 1;
            ctx->last_V_applied = V_app;
            ctx->last_J = J;
        }
    }
    if (!info.converged && snap) {
        /* 回滚到上一次收敛解，并沿用它的电流 */
        memcpy(ctx->state.fields[PVK_FIELD_PSI],  snap,     sizeof(double) * n);
        memcpy(ctx->state.fields[PVK_FIELD_PHIN], snap + n, sizeof(double) * n);
        memcpy(ctx->state.fields[PVK_FIELD_PHIP], snap + 2*n, sizeof(double) * n);
        J = ctx->last_J;
    } else if (!info.converged) {
        /* 没有任何可用解：写回平衡态（数值有界），电流报 NaN，
         * 绝不要把发散状态的 1e40 量级 dump 出去。 */
        equilibrium_guess(ctx, dev, &m, u, &ctx->solver_opts);
        fill_fields(ctx, &m, u, &J);
        J = NAN;
    }

    info.J_total_mA_cm2 = J;

    *out_info = info;

    model_free(&m);
    pvk__free(u);
    pvk__free(snap);

    if (info.converged) {
        char msg[160];
        snprintf(msg, sizeof(msg),
                 "pvk_solve_dc: converged, J(mA/cm2)=%.4g, iters=%d",
                 info.J_total_mA_cm2, info.iterations);
        pvk__log(ctx, PVK_LOG_INFO, msg);
    } else {
        pvk__set_error(ctx, PVK_ECONV,
                       "Newton not converged: V=%.4f, residual=%.3g, iters=%d",
                       V_app, info.residual, info.iterations);
    }
    return PVK_OK;
}

PVK_API int pvk_sweep_dc(pvk_ctx* ctx,
                         const pvk_device* dev,
                         const pvk_optics_in* opt,
                         const pvk_bias* bias,
                         double v_start,
                         double v_stop,
                         int    n_points,
                         pvk_point_cb cb,
                         void* cb_user,
                         pvk_solve_info* out_infos_out) {
    if (!ctx || !dev || !opt || !bias || n_points < 1) return PVK_EINVAL;
    if (n_points > 1 && v_start > v_stop) return PVK_EINVAL;

    for (int k = 0; k < n_points; k++) {
        double frac = (n_points == 1) ? 0.0 : (double)k / (double)(n_points - 1);
        double v_applied = v_start + frac * (v_stop - v_start);

        /* 扫描的是阳极（x=L，p 侧），阴极电位保持不变 */
        pvk_bias b = *bias;
        b.anode_V = v_applied;

        pvk_solve_info info;
        memset(&info, 0, sizeof(info));

        int rc = pvk_solve_dc(ctx, dev, opt, &b, &info);
        if (rc != PVK_OK) return rc;

        info.v_applied = v_applied;
        if (out_infos_out) out_infos_out[k] = info;

        if (cb) {
            int cb_rc = cb(cb_user, &info);
            if (cb_rc != 0) return PVK_EAPI;
        }
    }

    pvk__log(ctx, PVK_LOG_INFO, "pvk_sweep_dc: sweep complete");
    return PVK_OK;
}
