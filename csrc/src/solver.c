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
    double* brad;       /* 辐射复合系数 B，cm³/s（0=关闭；R_rad=B(np-ni²)） */
    double* Nd;
    double* Na;
    double* G;          /* cm⁻³ s⁻¹ */
    double* ni;
    double* chis;       /* chi/Vt + U_ref/Vt */
    double* egs;        /* Eg/Vt */
    double* hc;         /* 控制体宽度 */
    int*    bc;         /* 0=求解, 1=阴极型 Dirichlet, 2=阳极型 Dirichlet */
    int     eq_minority;/* 1 = 少数载流子退化时用 φ_min=φ_maj 兜底 */
    /* 接触层（隐式）：0=正常, 1=阴极侧接触层, 2=阳极侧接触层。
     * 这些节点的连续性行换成 φ 的边界方程：多数载流子钉在电极费米能级
     * （无限复合），少子沿层平到体内（零电流，选择性接触）。Poisson 仍求解。 */
    unsigned char* cpl;
    /* 接触层每个节点的多数载流子：1=电子, 2=空穴。
     * 选择性接触：多数载流子的 φ 钉在电极费米能级上（无限复合），
     * 少子的 φ 沿接触层平到体内（∂φ/∂x=0 ⇒ J=0，即少子被阻挡）。
     * 若两端都钉死两者，TiO2/钙钛矿这类有 ~1.9 eV 价带台阶的界面会在
     * 正偏下把空穴灌到 1e19 量级，接触层的少子通量发散。 */
    unsigned char* cpl_maj;
    /* 接触层掩码一旦在冷启动（平衡态）确定就冻结：否则偏压变化会让
     * 掩码随 ψ 漂移，方程组在不同偏压间切换，结果不可复现。 */
    int cpl_frozen;
    /* 本次 mark_contact_layer 里"被释放"（上一步还钉在接触层）的节点数。
     * >0 说明方程集刚变过一次、初值离真解较远，调用方应给 Newton 更多步。 */
    int cpl_released;

    double  u_bc[2];
    double  wn_bc[2];
    double  wp_bc[2];
} pvk_model;

/* ─────────────────────────────────────────────────────────────────────
 * 工具
 * ───────────────────────────────────────────────────────────────────── */

/* B(x) = x / (e^x - 1)，稳定求值 */
static double bern(double x) {
    /* Bernoulli 函数 B(x) = x/(e^x-1)。用 expm1 而不是 exp(x)-1：
     * |x| 略大于 1e-8 时 exp(x)-1 的绝对误差是 ~eps*e^x ≈ 1e-16，
     * 相对误差就是 1e-16/|x| ≈ 1e-8；而 SG 通量是两个 n·B(±Δ) 大项
     * 相消，这个 1e-8 直接成为 div J 的相对误差地板（低光时它比
     * |J|/hc 还大，0.1 mW 的 1e-9 精度要求正被它卡住）。
     * expm1 内部做补偿求和，相对误差回到 eps。数学上与原式完全相同。 */
    if (x > 1e-8) {
        if (x > 700.0) return 0.0;
        return x / expm1(x);
    }
    if (x < -1e-8) {
        double y = -x;
        if (y > 700.0) return y;
        return y / (-expm1(-y));
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

/* 连续方程行的归一尺度：max(本载流子边通量, 总电流, 全器件源项)/hc。
 *
 * 只除以当地载流子密度 n 时，接触积累层/异质结附近 n 可达 1e16–1e20，
 * SG 通量（~n·Vt/h）的 double 舍入噪声折算成相对残差就是 O(1)–O(10)，
 * Newton 永远压不下去（暗态 refine>0 停在 rel≈0.15 的根因）。改用
 * “电流守恒”的相对量纲：|div J - src| 与 |J|/hc 比较，舍入噪声只有
 * ~1e-10，物理不平衡仍是 O(1)。缩放因子恒正，不改变方程的根。
 *
 * 源项取**全器件** max(|R|,|G|,ni/τeff)（不是逐行）：高注入/正偏大电流区
 * 少数载流子行的 SG 一阶截断误差本身就是 O(R)，逐行尺度会把它判成不
 * 收敛（800 nm 卡 0.62–2.0 V，rel≈0.1 停滞）。ni/τeff 是暗态零偏
 * （R≡G≡0）时唯一非零的量纲。
 *
 * 总电流项是关键：中性区里电流几乎全由多数载流子承担，少数载流子行的
 * 自通量可以比它小十几个数量级；只用自通量归一，该行在低光下的截断
 * 误差会被放大成 O(1)（800 nm 0.1 mW 卡在 rel≈0.2，节点 106 的空穴行）。 */
static double cont_scale(double jSelfR, double jSelfL,
                         double jTotR, double jTotL,
                         double hc, double src_scale) {
    double s = fmax(fabs(jSelfR), fabs(jSelfL)) / hc;
    double jt = fmax(fabs(jTotR), fabs(jTotL)) / hc;
    if (jt > s) s = jt;
    if (src_scale > s) s = src_scale;
    if (!(s > 0.0) || !isfinite(s)) return 1.0;
    return s;
}

/* 累积相对残差：平方和 + 计数，最后取 RMS；同时记录峰值位置 */
#define REL_ACC_K(v, k) do { \
        double _v = (v); \
        if (!isfinite(_v)) _v = 1e6; \
        if (_v > 1e6) _v = 1e6; \
        if (_v < -1e6) _v = -1e6; \
        relsum += _v * _v; \
        relcnt++; \
        if (fabs(_v) > relpeak) { relpeak = fabs(_v); reli = i; relkind = (k); } \
    } while (0)

/* ─────────────────────────────────────────────────────────────────────
 * 残差 —— 返回按“最大项”归一化的相对残差 ∞-范数
 * ───────────────────────────────────────────────────────────────────── */
static double residual(const pvk_model* m, const double* u, double* r,
                       double* nbuf, double* pbuf, double* rnet,
                       double* rel_out, double* hidden_out) {
    const int n = m->n;

    for (int i = 0; i < n; i++) {
        node_dens(m, i, u, &nbuf[i], &pbuf[i]);
        if (!isfinite(nbuf[i])) nbuf[i] = 1e300;
        if (!isfinite(pbuf[i])) pbuf[i] = 1e300;
    }

    /* SRH 净复合 R（正=复合），叠加辐射复合 B(np-ni²)（B=0 时逐位不变） */
    for (int i = 0; i < n; i++) {
        double ni = m->ni[i];
        double np = nbuf[i] * pbuf[i];
        if (!isfinite(np)) np = 1e300;
        double dnp = np - ni * ni;
        double rad = m->brad[i] * dnp;
        if (!isfinite(rad)) rad = 0.0;
        double den = m->taup[i] * (nbuf[i] + ni) + m->taun[i] * (pbuf[i] + ni);
        if (den > 0.0 && isfinite(den)) {
            rnet[i] = dnp / den + rad;
        } else {
            rnet[i] = rad;
        }
        if (!isfinite(rnet[i])) rnet[i] = 0.0;
    }

    double rmax = 0.0;
    double rsum2 = 0.0;
    int rmax_i = -1, rmax_k = 0;   /* 最大缩放残差的位置，仅 PVK_DEBUG 诊断用 */
    double relsum = 0.0;   /* 相对残差平方和（判收敛用） */
    int    relcnt = 0;
    double relpeak = 0.0;  /* 峰值相对残差及其位置（诊断用） */
    int    reli = -1, relkind = -1;
    /* 被 eq_minority 替换掉的行不再约束物理连续性；这里记录这些行
     * "按真实方程算"的最大相对不平衡度。塌缩解（全层钉扎）靠删方程
     * 让 rel/rn 同时假收敛，但隐藏行的物理不平衡仍是 O(1)，用它识破。 */
    double hidden = 0.0;
    int hid_node = -1, hid_kind = 0;      /* 隐藏行峰值位置（PVK_DEBUG 诊断） */
    double hid_num = 0.0, hid_scale = 0.0;
    /* 被排除的两端界面行按真实方程算的最大不平衡度。只做诊断：它随 1/h
     * 增长（实测 101→397 为 1.3e-3→6.1e-2），拿来当收敛门槛会把正常点
     * 全部拒掉，所以不参与判据。 */
    double iface_diag = 0.0;

    /* 全器件源项尺度：max over nodes of max(|R|,|G|,ni/τeff)，见 cont_scale。
     * 取 R 是为了暗态正偏时也有物理量纲（中性区 R 比 ni/τeff 大十几个
     * 数量级）；保留 ni/τeff 是因为暗态零偏（平衡态）R≡G≡0，完全没有
     * 下限时 cont_scale 会退回 1.0，舍入噪声把 rel 撑到 O(1)。 */
    double smax = 0.0;
    for (int i = 0; i < n; i++) {
        double t = 1.0 / (1.0 / m->taun[i] + 1.0 / m->taup[i]);
        double s = fmax(fabs(rnet[i]), fabs(m->G[i]));
        if (m->ni[i] / t > s) s = m->ni[i] / t;
        if (s > smax) smax = s;
    }

    /* 两端紧邻接触层的**两个**自由节点的连续性行不参与判据。
     * 理想 Ohmic 金属侧的载流子密度与体内相差十几个数量级，中间积累层
     * 只有 ~0.2 nm，实用网格解析不了；再往外一个节点仍落在同一层边界层里
     * （1.9 eV 价带台阶把空穴堆在头几个节点上，正偏 1.3 V 时 p≈9e18）。
     * 这些行的通量不平衡随 1/h 增长（实测 101→397：1.3e-3→6.1e-2），但
     * 不代表体内方程没解到位：同一偏压下 J 在 101/201/401 上只差 1e-3 级，
     * 而失败边界反而随加密往下掉——说明卡住的是判据而不是解。
     * 注意：这不影响 1500nm 的假收敛防线，那道防线用的是 eq_minority
     * 替换行上的 hid，与这里的界面行无关。 */
    int first_free = -1, first_free2 = -1, last_free = -1, last_free2 = -1;
    if (rel_out) {
        int cnt = 0;
        for (int i = 0; i < n && cnt < 2; i++) {
            if (m->bc[i] || m->cpl[i]) continue;
            if (cnt == 0) first_free = i; else first_free2 = i;
            cnt++;
        }
        cnt = 0;
        for (int i = n - 1; i >= 0 && cnt < 2; i--) {
            if (m->bc[i] || m->cpl[i]) continue;
            if (cnt == 0) last_free = i; else last_free2 = i;
            cnt++;
        }
    }

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
            REL_ACC_K(a, 5);
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
            /* 收敛判据用的相对残差下限取 1e10 cm⁻³：界面附近 n、p 都
             * 极小（1e-17 / 1e5 量级）时，绝对残差 ~1e4 cm⁻³ 只是离散
             * 误差的噪声，却会让 |r|/max(|f|,|c|) 停在 1e-2 以上，把
             * 已经解到位的解判成未收敛。 */
            { double denm = den < 1e10 ? 1e10 : den;
              REL_ACC_K(fabs(r[i]) / denm, 0); }
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
        int iface_row = (i == first_free || i == first_free2 ||
                         i == last_free || i == last_free2);

        /* ---- 空穴连续 ---- */
        double jpR = m->mup[i+1] * m->Vt / hR *
                     (pbuf[i] * bern(dpR) - pbuf[i+1] * bern(-dpR));
        double jpL = m->mup[i] * m->Vt / hL *
                     (pbuf[i-1] * bern(dpL) - pbuf[i] * bern(-dpL));

        /* ---- 接触层隐式处理 ----
         * 电极旁的积累层德拜长度可到 0.1 nm，网格（哪怕 refine=2 的
         * 0.4 nm）解析不了。与其让连续性方程去追这个边界层（通量
         * 噪声把相对残差抬到 O(1)，Newton 永远停不下来），不如把这一层
         * 归入接触：**多数**载流子的 φ 钉在电极费米能级（无限复合），
         * **少数**载流子的 φ 沿接触层平到体内（∂φ/∂x=0 ⇔ J=0），即
         * 选择性接触。少子若也钉死，TiO2/钙钛矿这种 ~1.9 eV 价带台阶
         * 的界面会被当透明接触，正偏下把空穴灌到 1e19 且界面通量发散。
         * 载流子密度仍由 Poisson 与 ψ 自洽给出。Poisson 行保留。 */
        if (m->cpl[i]) {
            int e = m->cpl[i] - 1;
            int in = (m->cpl[i] == 1) ? i + 1 : i - 1;   /* 朝器件内部 */
            int has_in = (in >= 0 && in < n);
            int maj = m->cpl_maj ? m->cpl_maj[i] : 0;
            if (maj == 1) {          /* 电子多数：钉 φn，空穴零电流 */
                r[n + i]     = u[n + i] - m->wn_bc[e];
                r[2 * n + i] = has_in ? u[2*n + i] - u[2*n + in]
                                      : u[2*n + i] - m->wp_bc[e];
            } else if (maj == 2) {   /* 空穴多数：钉 φp，电子零电流 */
                r[2 * n + i] = u[2*n + i] - m->wp_bc[e];
                r[n + i]     = has_in ? u[n + i] - u[n + in]
                                      : u[n + i] - m->wn_bc[e];
            } else {
                r[n + i]     = u[n + i]     - m->wn_bc[e];
                r[2 * n + i] = u[2*n + i]   - m->wp_bc[e];
            }
            double sc = fabs(r[n+i]);
            if (fabs(r[2*n+i]) > sc) sc = fabs(r[2*n+i]);
            if (sc > rmax) rmax = sc;
            rsum2 += r[n+i]*r[n+i] + r[2*n+i]*r[2*n+i];
            REL_ACC_K(r[n + i], 9);
            REL_ACC_K(r[2*n + i], 10);
            continue;
        }

        if (m->eq_minority) {
            /* 少数载流子退化时换成 φn=φp。两种触发条件：
             *  - 相对：比另一种低 6 个数量级以上；
             *  - 绝对：比本层 ni 还低 20 个数量级（深耗尽区两者都极小）。 */
            int e_deg = (nbuf[i] < 1e-6 * pbuf[i]) || (nbuf[i] < 1e-20 * m->ni[i]);
            int p_deg = (pbuf[i] < 1e-6 * nbuf[i]) || (pbuf[i] < 1e-20 * m->ni[i]);
            if (e_deg && p_deg) p_deg = 0;   /* 两条都换会变成同一个方程 */
            if (e_deg) {
                r[n + i] = u[n + i] - u[2*n + i];
                REL_ACC_K(r[n + i], 3);
                if (hidden_out && !iface_row) {
                    double sc = cont_scale(jnR, jnL, jnR + jpR, jnL + jpL, hc, smax);
                    double hp = fabs((jnR - jnL) / hc - src) / sc;
                    if (!isfinite(hp)) hp = 1e6;
                    if (hp > hidden) {
                        hidden = hp; hid_node = i; hid_kind = 1;
                        hid_num = fabs((jnR - jnL) / hc - src); hid_scale = sc;
                    }
                }
            }
            if (p_deg) {
                r[2 * n + i] = u[2*n + i] - u[n + i];
                REL_ACC_K(r[2*n + i], 4);
                if (hidden_out && !iface_row) {
                    double sc = cont_scale(jpR, jpL, jnR + jpR, jnL + jpL, hc, smax);
                    double hp = fabs((jpR - jpL) / hc + src) / sc;
                    if (!isfinite(hp)) hp = 1e6;
                    if (hp > hidden) {
                        hidden = hp; hid_node = i; hid_kind = 2;
                        hid_num = fabs((jpR - jpL) / hc + src); hid_scale = sc;
                    }
                }
            }
            if (!e_deg) {
                double den = nbuf[i] > 1e-280 ? nbuf[i] : 1e-280;
                r[n + i] = ((jnR - jnL) / hc - src) / den;
                { double rc = fabs((jnR - jnL) / hc - src)
                                  / cont_scale(jnR, jnL, jnR + jpR, jnL + jpL, hc, smax);
                  if (!iface_row) REL_ACC_K(rc, 1); }
            }
            if (!p_deg) {
                double den = pbuf[i] > 1e-280 ? pbuf[i] : 1e-280;
                r[2 * n + i] = ((jpR - jpL) / hc + src) / den;
                { double rc = fabs((jpR - jpL) / hc + src)
                                  / cont_scale(jpR, jpL, jnR + jpR, jnL + jpL, hc, smax);
                  if (!iface_row) REL_ACC_K(rc, 2); }
            }
        } else {
            double den_n = nbuf[i] > 1e-280 ? nbuf[i] : 1e-280;
            r[n + i] = ((jnR - jnL) / hc - src) / den_n;
            { double rc = fabs((jnR - jnL) / hc - src)
                              / cont_scale(jnR, jnL, jnR + jpR, jnL + jpL, hc, smax);
              if (!iface_row) REL_ACC_K(rc, 1); }
            double den_p = pbuf[i] > 1e-280 ? pbuf[i] : 1e-280;
            r[2 * n + i] = ((jpR - jpL) / hc + src) / den_p;
            { double rc = fabs((jpR - jpL) / hc + src)
                              / cont_scale(jpR, jpL, jnR + jpR, jnL + jpL, hc, smax);
              if (!iface_row) REL_ACC_K(rc, 2); }
        }

        /* 被排除的界面行：只记诊断值（归一尺度与 REL_ACC 相同） */
        if (iface_row) {
            double rcn = fabs((jnR - jnL) / hc - src)
                       / cont_scale(jnR, jnL, jnR + jpR, jnL + jpL, hc, smax);
            double rcp = fabs((jpR - jpL) / hc + src)
                       / cont_scale(jpR, jpL, jnR + jpR, jnL + jpL, hc, smax);
            double rc = fmax(rcn, rcp);
            if (!isfinite(rc)) rc = 1e6;
            if (rc > iface_diag) iface_diag = rc;
        }

        {
            double scn = fabs(r[n+i]), scp = fabs(r[2*n+i]);
            double sc = scn > scp ? scn : scp;
            if (sc > rmax) { rmax = sc; rmax_i = i; rmax_k = (scn >= scp) ? 1 : 2; }
            rsum2 += r[n+i]*r[n+i] + r[2*n+i]*r[2*n+i];
        }
    }
    if (!isfinite(rmax) || !isfinite(rsum2)) {
        if (rel_out) *rel_out = 1e300;
        if (hidden_out) *hidden_out = 1e300;
        return 1e300;
    }
    if (rel_out) {
        *rel_out = relcnt > 0 ? sqrt(relsum / (double)relcnt) : 0.0;
        if (getenv("PVK_DEBUG") && relpeak > 1e-1 && reli >= 0) {
            /* kind: 0=Poisson 1=电子连续 2=空穴连续 3/4=φn=φp 5=边界 */
            fprintf(stderr,
                    "[pvk] rel peak=%.3e node=%d layer=%d kind=%d x=%.2f nm "
                    "n=%.3g p=%.3g\n",
                    relpeak, reli, m->node_layer[reli], relkind,
                    m->x[reli] * 1e7, nbuf[reli], pbuf[reli]);
        }
        /* 两端界面行（已排除在判据外）的不平衡度，仅诊断用 */
        if (getenv("PVK_DEBUG") && iface_diag > 1e-2)
            fprintf(stderr, "[pvk] iface rows (excluded): imbalance=%.2e\n",
                    iface_diag);
    }
    if (hidden_out) *hidden_out = hidden;
    if (getenv("PVK_DEBUG") && hidden > 1e-2 && hid_node >= 0) {
        fprintf(stderr,
                "[pvk] hid=%.3e node=%d layer=%d kind=%d x=%.2f nm "
                "n=%.3g p=%.3g num=%.3e scale=%.3e\n",
                hidden, hid_node, m->node_layer[hid_node], hid_kind,
                m->x[hid_node] * 1e7, nbuf[hid_node], pbuf[hid_node],
                hid_num, hid_scale);
    }
    if (getenv("PVK_DEBUG") && rmax > 1e-1 && rmax_i >= 0) {
        fprintf(stderr,
                "[pvk] rmax=%.3e node=%d layer=%d kind=%d x=%.2f nm n=%.3g p=%.3g\n",
                rmax, rmax_i, m->node_layer[rmax_i], rmax_k,
                m->x[rmax_i] * 1e7, nbuf[rmax_i], pbuf[rmax_i]);
    }
    /* 2-范数：Newton 方向是它的下降方向（∞-范数不保证），线搜索用它 */
    return sqrt(rsum2 / (double)(3 * n));
}

/* ─────────────────────────────────────────────────────────────────────
 * 局部残差（单节点三行）
 *
 * 数值雅可比只需要被扰动节点 ±1 的残差行，但 residual() 每次都扫全器件
 * （O(n)），3n 列的代价就是 O(n²)。这里把单节点的三行公式抽出来，
 * 配合预计算的 nbuf/pbuf/rnet 复用，使整块雅可比的代价降到 O(n)。
 * 公式必须与 residual() 逐项一致（含 bc / 接触层 / 少子退化分支）。
 * ───────────────────────────────────────────────────────────────────── */
static void residual_local(const pvk_model* m, const double* u, int i,
                           const double* nbuf, const double* pbuf,
                           const double* rnet, double out[3]) {
    const int n = m->n;

    if (m->bc[i]) {
        int e = m->bc[i] - 1;
        out[0] = u[i]         - m->u_bc[e];
        out[1] = u[n + i]     - m->wn_bc[e];
        out[2] = u[2 * n + i] - m->wp_bc[e];
        return;
    }

    double hL = m->h[i-1];
    double hR = m->h[i];
    double hc = m->hc[i];

    double fluxR = m->eps[i+1] * (u[i+1] - u[i]) * m->Vt / hR;
    double fluxL = m->eps[i]   * (u[i]   - u[i-1]) * m->Vt / hL;
    double fterm = (fluxR - fluxL) / (PVK_Q_C * hc);
    out[0] = fterm + (pbuf[i] - nbuf[i] + m->Nd[i] - m->Na[i]);

    if (m->cpl[i]) {
        int e = m->cpl[i] - 1;
        int in = (m->cpl[i] == 1) ? i + 1 : i - 1;
        int has_in = (in >= 0 && in < n);
        int maj = m->cpl_maj ? m->cpl_maj[i] : 0;
        if (maj == 1) {
            out[1] = u[n + i] - m->wn_bc[e];
            out[2] = has_in ? u[2*n + i] - u[2*n + in]
                            : u[2*n + i] - m->wp_bc[e];
        } else if (maj == 2) {
            out[2] = u[2*n + i] - m->wp_bc[e];
            out[1] = has_in ? u[n + i] - u[n + in]
                            : u[n + i] - m->wn_bc[e];
        } else {
            out[1] = u[n + i]     - m->wn_bc[e];
            out[2] = u[2 * n + i] - m->wp_bc[e];
        }
        return;
    }

    double duR = u[i+1] - u[i];
    double duL = u[i]   - u[i-1];
    double dnR = duR + (m->chis[i+1] - m->chis[i]) + log(m->Nc[i+1] / m->Nc[i]);
    double dnL = duL + (m->chis[i]   - m->chis[i-1]) + log(m->Nc[i] / m->Nc[i-1]);
    double dpR = duR + (m->chis[i+1] - m->chis[i])
                 + (m->egs[i+1] - m->egs[i]) + log(m->Nv[i] / m->Nv[i+1]);
    double dpL = duL + (m->chis[i]   - m->chis[i-1])
                 + (m->egs[i]   - m->egs[i-1]) + log(m->Nv[i-1] / m->Nv[i]);
    double jnR = m->mun[i+1] * m->Vt / hR *
                 (nbuf[i+1] * bern(dnR) - nbuf[i] * bern(-dnR));
    double jnL = m->mun[i] * m->Vt / hL *
                 (nbuf[i] * bern(dnL) - nbuf[i-1] * bern(-dnL));
    double jpR = m->mup[i+1] * m->Vt / hR *
                 (pbuf[i] * bern(dpR) - pbuf[i+1] * bern(-dpR));
    double jpL = m->mup[i] * m->Vt / hL *
                 (pbuf[i-1] * bern(dpL) - pbuf[i] * bern(-dpL));
    double src = rnet[i] - m->G[i];

    if (m->eq_minority) {
        int e_deg = (nbuf[i] < 1e-6 * pbuf[i]) || (nbuf[i] < 1e-20 * m->ni[i]);
        int p_deg = (pbuf[i] < 1e-6 * nbuf[i]) || (pbuf[i] < 1e-20 * m->ni[i]);
        if (e_deg && p_deg) p_deg = 0;
        if (!e_deg) {
            double den = nbuf[i] > 1e-280 ? nbuf[i] : 1e-280;
            out[1] = ((jnR - jnL) / hc - src) / den;
        } else {
            out[1] = u[n + i] - u[2 * n + i];
        }
        if (!p_deg) {
            double den = pbuf[i] > 1e-280 ? pbuf[i] : 1e-280;
            out[2] = ((jpR - jpL) / hc + src) / den;
        } else {
            out[2] = u[2 * n + i] - u[n + i];
        }
    } else {
        double den_n = nbuf[i] > 1e-280 ? nbuf[i] : 1e-280;
        double den_p = pbuf[i] > 1e-280 ? pbuf[i] : 1e-280;
        out[1] = ((jnR - jnL) / hc - src) / den_n;
        out[2] = ((jpR - jpL) / hc + src) / den_p;
    }
}

/* ─────────────────────────────────────────────────────────────────────
 * 带状 LU（部分选主元）
 *
 * 残差按（方程类型 t, 节点 i）排列、未知量按（变量类型 s, 节点 j）排列时，
 * 每个方程只耦合 j = i-1,i,i+1。把未知量和方程都重排成“节点主序”
 * （r = 3i+t, c = 3j+s）后，矩阵是带宽 5 的带状矩阵：
 *   |r-c| = |3(i-j)+(t-s)| <= 3+2 = 5
 * 带状 LU（下带宽 5、上带宽 5）配合部分选主元，复杂度 O(n)，且主元
 * 可以跨节点交换 —— 这是块 Thomas 做不到的（它对近奇异对角块不稳）。
 *
 * 选主元后上带宽会增长到 下带宽+上带宽 = 10（标准结论），所以按
 * BL=5、BU=10 存储，共 16 条对角线。A(r,c) 存在 ab[r*16 + (c-r+5)]。
 * ───────────────────────────────────────────────────────────────────── */
#define PVK_BW_L 5
#define PVK_BW_U 10
#define PVK_BW_W (PVK_BW_L + PVK_BW_U + 1)

/* 行均衡 + 对角正则 + 带状 LU + 回代；b 既作右端也作解。 */
static int band_solve(int nn, double* ab, double* b) {
    const int w = PVK_BW_W;

    for (int i = 0; i < nn; i++) {
        double* row = ab + (size_t)i * w;
        double s = 0.0;
        for (int k = 0; k < w; k++) {
            double v = fabs(row[k]);
            if (v > s) s = v;
        }
        if (!(s > 1e-290) || !isfinite(s)) {
            if (getenv("PVK_DEBUG"))
                fprintf(stderr, "[pvk] band row %d degenerate (freeze)\n", i);
            for (int k = 0; k < w; k++) row[k] = 0.0;
            row[PVK_BW_L] = 1.0;
            b[i] = 0.0;
            continue;
        }
        double inv = 1.0 / s;
        for (int k = 0; k < w; k++) row[k] *= inv;
        b[i] *= inv;
    }
    for (int i = 0; i < nn; i++) ab[(size_t)i * w + PVK_BW_L] += 1e-10;

    for (int k = 0; k < nn; k++) {
        int rmax = k + PVK_BW_L;
        if (rmax >= nn) rmax = nn - 1;
        int piv = k;
        double best = fabs(ab[(size_t)k * w + PVK_BW_L]);
        for (int r = k + 1; r <= rmax; r++) {
            double v = fabs(ab[(size_t)r * w + (k - r + PVK_BW_L)]);
            if (v > best) { best = v; piv = r; }
        }
        if (!(best > 1e-14) || !isfinite(best)) return -1;
        if (piv != k) {
            int c0 = piv - PVK_BW_L; if (c0 < 0) c0 = 0;
            int c1 = k + PVK_BW_U;   if (c1 >= nn) c1 = nn - 1;
            for (int c = c0; c <= c1; c++) {
                double* rk = ab + (size_t)k   * w + (c - k   + PVK_BW_L);
                double* rp = ab + (size_t)piv * w + (c - piv + PVK_BW_L);
                double t = *rk; *rk = *rp; *rp = t;
            }
            double t = b[k]; b[k] = b[piv]; b[piv] = t;
        }
        double akk = ab[(size_t)k * w + PVK_BW_L];
        for (int i = k + 1; i <= rmax; i++) {
            double* ri = ab + (size_t)i * w;
            double f = ri[k - i + PVK_BW_L] / akk;
            if (f == 0.0) continue;
            ri[k - i + PVK_BW_L] = 0.0;
            int cmax = k + PVK_BW_U; if (cmax >= nn) cmax = nn - 1;
            if (i + PVK_BW_U < cmax) cmax = i + PVK_BW_U;
            for (int c = k + 1; c <= cmax; c++)
                ri[c - i + PVK_BW_L] -= f * ab[(size_t)k * w + (c - k + PVK_BW_L)];
            b[i] -= f * b[k];
        }
    }

    for (int i = nn - 1; i >= 0; i--) {
        double s = b[i];
        int cmax = i + PVK_BW_U; if (cmax >= nn) cmax = nn - 1;
        for (int c = i + 1; c <= cmax; c++)
            s -= ab[(size_t)i * w + (c - i + PVK_BW_L)] * b[c];
        double aii = ab[(size_t)i * w + PVK_BW_L];
        if (!(fabs(aii) > 1e-300) || !isfinite(aii)) return -1;
        b[i] = s / aii;
    }
    for (int i = 0; i < nn; i++) if (!isfinite(b[i])) return -1;
    return 0;
}

/* ─────────────────────────────────────────────────────────────────────
 * 三对角 LU（部分选主元）
 *
 * 平衡态 Poisson 只有 ψ 一个未知量，每个方程只耦合 i-1,i,i+1。
 * 部分选主元会让上带宽增长到 2，所以按 BL=1、BU=2 存 4 条对角线：
 *   ab[i*4 + (c - i + 1)]
 * 复杂度 O(n)，替代原来 dense_solve 的 O(n^3)。行均衡 / 对角正则 /
 * 奇异行冻结的语义与 band_solve 保持一致。
 * ───────────────────────────────────────────────────────────────────── */
#define PVK_TRI_W 4
static int tri_solve(int nn, double* ab, double* b) {
    const int w = PVK_TRI_W;

    for (int i = 0; i < nn; i++) {
        double* row = ab + (size_t)i * w;
        double s = 0.0;
        for (int k = 0; k < w; k++) {
            double v = fabs(row[k]);
            if (v > s) s = v;
        }
        if (!(s > 1e-290) || !isfinite(s)) {
            if (getenv("PVK_DEBUG"))
                fprintf(stderr, "[pvk] tri row %d degenerate (freeze)\n", i);
            for (int k = 0; k < w; k++) row[k] = 0.0;
            row[1] = 1.0;
            b[i] = 0.0;
            continue;
        }
        double inv = 1.0 / s;
        for (int k = 0; k < w; k++) row[k] *= inv;
        b[i] *= inv;
    }
    for (int i = 0; i < nn; i++) ab[(size_t)i * w + 1] += 1e-10;

    for (int k = 0; k < nn; k++) {
        int rmax = k + 1;
        if (rmax >= nn) rmax = nn - 1;
        int piv = k;
        double best = fabs(ab[(size_t)k * w + 1]);
        for (int r = k + 1; r <= rmax; r++) {
            double v = fabs(ab[(size_t)r * w + (k - r + 1)]);
            if (v > best) { best = v; piv = r; }
        }
        if (!(best > 1e-14) || !isfinite(best)) return -1;
        if (piv != k) {
            /* 列 c<k 在消元后已显式置零，只需交换 c>=k 的部分 */
            int c0 = piv - 1;
            if (c0 < 0) c0 = 0;
            int c1 = k + 2;
            if (c1 >= nn) c1 = nn - 1;
            for (int c = c0; c <= c1; c++) {
                double* rk = ab + (size_t)k   * w + (c - k   + 1);
                double* rp = ab + (size_t)piv * w + (c - piv + 1);
                double t = *rk; *rk = *rp; *rp = t;
            }
            double t = b[k]; b[k] = b[piv]; b[piv] = t;
        }
        double akk = ab[(size_t)k * w + 1];
        int i = k + 1;
        if (i >= nn) continue;
        double f = ab[(size_t)i * w + (k - i + 1)] / akk;
        if (f != 0.0) {
            ab[(size_t)i * w + (k - i + 1)] = 0.0;
            int cmax = k + 2;
            if (cmax >= nn) cmax = nn - 1;
            if (i + 2 < cmax) cmax = i + 2;
            for (int c = k + 1; c <= cmax; c++)
                ab[(size_t)i * w + (c - i + 1)] -=
                    f * ab[(size_t)k * w + (c - k + 1)];
            b[i] -= f * b[k];
        }
    }

    for (int i = nn - 1; i >= 0; i--) {
        double s = b[i];
        int cmax = i + 2;
        if (cmax >= nn) cmax = nn - 1;
        for (int c = i + 1; c <= cmax; c++)
            s -= ab[(size_t)i * w + (c - i + 1)] * b[c];
        double aii = ab[(size_t)i * w + 1];
        if (!(fabs(aii) > 1e-300) || !isfinite(aii)) return -1;
        b[i] = s / aii;
    }
    for (int i = 0; i < nn; i++) if (!isfinite(b[i])) return -1;
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
    pvk__free(m->taun); pvk__free(m->taup); pvk__free(m->brad);
    pvk__free(m->Nd);   pvk__free(m->Na);
    pvk__free(m->G);    pvk__free(m->ni);
    pvk__free(m->chis); pvk__free(m->egs);
    pvk__free(m->hc);
    pvk__free(m->bc);
    pvk__free(m->cpl);
    pvk__free(m->cpl_maj);
    memset(m, 0, sizeof(*m));
}

/* homo=0 时把某层参数从参考值线性拉到 homo=1 的真实值。
 * 只用于平衡态求解的带边 homotopy（见 equilibrium_guess）。 */
static double homo_lerp(double v, double ref, double t) {
    return ref + t * (v - ref);
}

/* 某一层电中性时的 ψ/Vt（不含 U_ref）。
 * 解 A·e^z - B·e^-z = Nd - Na（A=Nc e^{χ/Vt}, B=Nv e^{-(χ+Eg)/Vt}）。
 *
 * 电极边界条件必须用它，而不是按“本征 / n 型 / p 型”分支套公式：
 * homotopy 里掺杂按 homo·Nd/Na 缩放，homo→0+ 时 Na→0，p 型分支的
 * -Vt·log(Na/Nv) 会跳到 +∞ 再掉回本征值，pin 住的接触电位在第一步
 * 就跳 0.2 V 以上。refine>0 时接触旁的网格只有 0.1 nm，这个跳变
 * 直接让 Poisson 的 Newton 在 it=0 线搜索失败。用统一的电中性解，
 * 电位随 homo 连续变化。
 *
 * 实现上用「本征点 + asinh」：ni = sqrt(Nc·Nv)·e^{-Eg/2Vt} 常常小到
 * 下溢（TiO2 的 Eg=3.2 eV 时 ni~1e-6），直接解二次式会丢精度，
 * 因此在 log 空间算 rho = (Nd-Na)/(2·ni)。 */
static double neutral_u(double Nc, double Nv, double chi, double eg,
                        double Nd, double Na, double Vt) {
    if (!(Nc > 0.0) || !(Nv > 0.0) || !(Vt > 0.0)) return 0.0;
    double ln_ni = 0.5 * (log(Nc) + log(Nv)) - eg / (2.0 * Vt);
    double u_i   = 0.5 * log(Nv / Nc) - chi / Vt - eg / (2.0 * Vt);
    double D = Nd - Na;
    if (D == 0.0 || !isfinite(ln_ni)) return u_i;

    /* delta = asinh(rho)，rho = D/(2·ni)，在 log 空间算模长 */
    double lrho = log(fabs(D)) - 0.6931471805599453 - ln_ni;
    double delta;
    if (lrho > 30.0)       delta = lrho + 0.6931471805599453;
    else if (lrho < -30.0) delta = exp(lrho);
    else                   delta = asinh(exp(lrho));
    return u_i + (D > 0.0 ? delta : -delta);
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

    /* 带边 homotopy 的参考点：全器件带隙最小的一层（含电极）。
     * 这是历史沿用的路径：参考点带隙越窄、本征载流子越少（Au 的
     * Eg=1.0 eV 时 ni~1e11），t=0 时器件近似“空”的，Poisson 近 Laplace，
     * 对初值偏压最不敏感。homo=1 时所有插值退化成真实参数。
     * 只改参考点会改变整条 homotopy 路径（exp 型 Poisson 有多解），
     * 亮态冷启动对路径敏感，所以这里保持原样。 */
    int iref = 0;
    for (int i = 1; i < dev->n_layers; i++)
        if (dev->layers[i].Eg_eV < dev->layers[iref].Eg_eV) iref = i;
    double chi_ref = dev->layers[iref].chi_eV;
    double eg_ref  = dev->layers[iref].Eg_eV;

    /* 电极层的带参数在下面会被换成相邻半导体的（理想 Ohmic 接触），
     * 所以边界电位也必须用相邻半导体的 Nc/Nv/χ/Eg 来算，只保留电极
     * 自己的掺杂。否则 pin 住的电位和接触节点的带参数差 ~1.5 eV
     * （Au 4.7/1.0 vs Spiro 2.5/3.0），粗网格把这台阶摊在 16 nm 上还能
     * 忍，interface.refine>0 时首格 0.1 nm，电场 1e8 V/cm，Poisson 的
     * Newton 在 t≈0.4 就停滞。 */
    int lic = 0;
    while (lic + 1 < dev->n_layers && dev->layers[lic].is_contact) lic++;
    int lia = dev->n_layers - 1;
    while (lia > 0 && dev->layers[lia].is_contact) lia--;
    const pvk_layer* Cb = &dev->layers[lic];
    const pvk_layer* Ab = &dev->layers[lia];

    /* U_ref：让 x=0 处 ψ=0、φ=0 时满足阴极层的电中性。
     * homo 同时缩放掺杂：homo=0 时整个器件是本征、同一带结构，
     * 各层电中性电位全是 ψ=0（含两个电极的边界值），初值即精确解。 */
    double chi_C = homo_lerp(Cb->chi_eV, chi_ref, homo);
    double eg_C  = homo_lerp(Cb->Eg_eV,  eg_ref,  homo);
    double nc_C  = Cb->Nc_cm3 > 0 ? Cb->Nc_cm3 : 1.0;
    double nv_C  = Cb->Nv_cm3 > 0 ? Cb->Nv_cm3 : 1.0;
    double Nd_C  = homo * C->Nd_cm3;
    double Na_C  = homo * C->Na_cm3;
    double u_neu_C = neutral_u(nc_C, nv_C, chi_C, eg_C, Nd_C, Na_C, Vt);
    m->U_ref = Vt * u_neu_C;

    /* 阳极 ψ 的 Dirichlet 值（电中性 + 准费米劈裂）。
     * 注意电化学势符号：阳极静电势比阴极高 V_app 时，电子的准费米势
     * 下降 q·V_app —— 所以 φ = -V_app，ψ 上则是 +V_app。 */
    double chi_A = homo_lerp(Ab->chi_eV, chi_ref, homo);
    double eg_A  = homo_lerp(Ab->Eg_eV,  eg_ref,  homo);
    double nc_A  = Ab->Nc_cm3 > 0 ? Ab->Nc_cm3 : 1.0;
    double nv_A  = Ab->Nv_cm3 > 0 ? Ab->Nv_cm3 : 1.0;
    double Nd_A  = homo * A->Nd_cm3;
    double Na_A  = homo * A->Na_cm3;
    double u_neu_A = neutral_u(nc_A, nv_A, chi_A, eg_A, Nd_A, Na_A, Vt);
    m->psi_a = Vt * (u_neu_A - u_neu_C) + V_app;

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
    m->brad = (double*)pvk__malloc(dn);
    m->Nd   = (double*)pvk__malloc(dn);
    m->Na   = (double*)pvk__malloc(dn);
    m->G    = (double*)pvk__malloc(dn);
    m->ni   = (double*)pvk__malloc(dn);
    m->chis = (double*)pvk__malloc(dn);
    m->egs  = (double*)pvk__malloc(dn);
    m->hc   = (double*)pvk__malloc(dn);
    m->bc   = (int*)pvk__malloc(sizeof(int) * (size_t)n);
    m->cpl  = (unsigned char*)pvk__malloc(sizeof(unsigned char) * (size_t)n);
    m->cpl_maj = (unsigned char*)pvk__malloc(sizeof(unsigned char) * (size_t)n);
    if (!m->eps || !m->chi || !m->Eg || !m->Nc || !m->Nv || !m->mun ||
        !m->mup || !m->taun || !m->taup || !m->brad || !m->Nd || !m->Na || !m->G ||
        !m->ni || !m->chis || !m->egs || !m->hc || !m->bc || !m->cpl ||
        !m->cpl_maj) {
        model_free(m);
        return PVK_EMEM;
    }
    memset(m->cpl, 0, sizeof(unsigned char) * (size_t)n);
    memset(m->cpl_maj, 0, sizeof(unsigned char) * (size_t)n);
    m->cpl_frozen = 0;

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
        m->brad[i] = L->b_rad_cm3_s > 0 ? L->b_rad_cm3_s : 0.0;
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
        if (getenv("PVK_DEBUG_MODEL"))
            fprintf(stderr, "[pvk] copy: i=%d bc=%d/%d jsemi=%d lcont=%d chi_semi=%.3f\n",
                    i, m->bc[i], m->bc[i+1], jsemi, lcont, m->chi[jsemi]);
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

    /* 理想 Ohmic 接触的边界条件施加在紧邻电极的那个半导体节点上。
     * 电极层整层仍钉死，但它与半导体之间的亚纳米积累层不再参与离散：
     * 1e20 的金属接触贴着本征层时德拜长度只有 ~0.2 nm，要求网格解析
     * 这层会让暗态连续性残差按 1/h 增长（101→397 从 4.5 涨到 436）。
     *
     * 前提：转换后该层必须还留有自由节点。否则整个薄层会被并入电极
     * （1500nm 吸收层时 40nm TiO2 在 101 节点网格下只剩 1 个节点），
     * mark_contact_layer 的首个自由节点就落进吸收层，"限本层"防线
     * 失效，吸收层的积累尾巴被整段钉扎成电极——产生载流子没有输运
     * 方程，亮态 Jsc 从 -21 塌到 -0.07 还被宽松判据报成收敛。
     * 留不下第二个节点时不转换，该节点由 cpl 掩码按积累层处理，
     * φ 钉扎效果等价，但吸收层不会再被卷进标记起点。 */
    {
        int first_free = -1, last_free = -1;
        for (int i = 0; i < n; i++) if (!m->bc[i]) { first_free = i; break; }
        for (int i = n - 1; i >= 0; i--) if (!m->bc[i]) { last_free = i; break; }
        if (first_free > 0 &&
            first_free + 1 < n && !m->bc[first_free + 1] &&
            m->node_layer[first_free + 1] == m->node_layer[first_free])
            m->bc[first_free] = m->bc[first_free - 1];
        if (last_free >= 0 && last_free < n - 1 &&
            last_free - 1 >= 0 && !m->bc[last_free - 1] &&
            m->node_layer[last_free - 1] == m->node_layer[last_free])
            m->bc[last_free] = m->bc[last_free + 1];
    }

    m->u_bc[0]  = 0.0;
    m->u_bc[1]  = m->psi_a / Vt;
    m->wn_bc[0] = 0.0;
    m->wn_bc[1] = -V_app / Vt;
    m->wp_bc[0] = 0.0;
    m->wp_bc[1] = -V_app / Vt;

    if (getenv("PVK_DEBUG") && getenv("PVK_DEBUG_MODEL")) {
        fprintf(stderr, "[pvk] model homo=%g V=%g n=%d u_bc=%.4f/%.4f\n",
                homo, V_app, n, m->u_bc[0], m->u_bc[1]);
        for (int i = 0; i < n; i++) {
            if (m->bc[i] || (i + 1 < n && m->bc[i+1])) {
                fprintf(stderr,
                        "[pvk]   node=%d layer=%d bc=%d x=%.3f chi=%.3f eg=%.3f "
                        "Nc=%.2e Nd=%.2e\n",
                        i, m->node_layer[i], m->bc[i], m->x[i] * 1e7,
                        m->chi[i], m->Eg[i], m->Nc[i], m->Nd[i]);
            }
        }
    }

    return PVK_OK;
}

/* ─────────────────────────────────────────────────────────────────────
 * Newton
 * ───────────────────────────────────────────────────────────────────── */

/* 收敛失败诊断：打印违反最大的 3 行（仅 PVK_DEBUG=1）。 */
static void dump_worst_rows(const pvk_model* m, const double* r,
                            const double* u, int nn) {
    int n = m->n;
    int picked[3] = { -1, -1, -1 };
    for (int k = 0; k < 3; k++) {
        int worst = -1;
        double wv = -1.0;
        for (int i = 0; i < nn; i++) {
            int skip = 0;
            for (int q = 0; q < k; q++) if (picked[q] == i) { skip = 1; break; }
            if (skip) continue;
            double v = fabs(r[i]);
            if (v > wv) { wv = v; worst = i; }
        }
        if (worst < 0) break;
        picked[k] = worst;
        int t = worst / n, node = worst % n;
        double nv = 0.0, pv = 0.0;
        node_dens(m, node, u, &nv, &pv);
        fprintf(stderr, "[pvk]   worst %s node=%d x=%.4g nm |r|=%.3e n=%.3e p=%.3e\n",
                t == 0 ? "psi " : (t == 1 ? "phin" : "phip"),
                node, m->x[node] * 1e7, wv, nv, pv);
    }
}

static int solve_newton(pvk_model* m, double* u,
                        const pvk_solver_opts* opts, pvk_solve_info* out) {
    const int n = m->n;
    const int nn = 3 * n;

    double* r      = (double*)pvk__malloc(sizeof(double) * nn);
    double* rtmp   = (double*)pvk__malloc(sizeof(double) * nn);
    double* utmp   = (double*)pvk__malloc(sizeof(double) * nn);
    double* du     = (double*)pvk__malloc(sizeof(double) * nn);
    /* 带状 LU：方程/未知量按节点主序（3i+t），带宽 5，
     * 存储宽度 BL+BU=15（选主元后上带宽增长，见 band_solve） */
    double* band   = (double*)pvk__malloc(sizeof(double) * PVK_BW_W * (size_t)nn);
    double* bvec   = (double*)pvk__malloc(sizeof(double) * nn);
    double* nbuf   = (double*)pvk__malloc(sizeof(double) * n);
    double* pbuf   = (double*)pvk__malloc(sizeof(double) * n);
    double* rnet   = (double*)pvk__malloc(sizeof(double) * n);
    double* nb2    = (double*)pvk__malloc(sizeof(double) * n);
    double* pb2    = (double*)pvk__malloc(sizeof(double) * n);
    double* rn2    = (double*)pvk__malloc(sizeof(double) * n);

    if (!r || !rtmp || !utmp || !du || !band || !bvec ||
        !nbuf || !pbuf || !rnet || !nb2 || !pb2 || !rn2) {
        pvk__free(r); pvk__free(rtmp); pvk__free(utmp); pvk__free(du);
        pvk__free(band); pvk__free(bvec);
        pvk__free(nbuf); pvk__free(pbuf); pvk__free(rnet);
        pvk__free(nb2); pvk__free(pb2); pvk__free(rn2);
        return PVK_EMEM;
    }

    int maxit = opts && opts->max_iter > 0 ? opts->max_iter : 60;
    double tol = opts && opts->res_tol > 0 ? opts->res_tol : 1e-8;
    double alpha0 = opts && opts->damping > 0 ? opts->damping : 1.0;
    if (alpha0 > 1.0) alpha0 = 1.0;

    double rn = residual(m, u, r, nbuf, pbuf, rnet, NULL, NULL);
    int converged = 0;
    int iters = 0;
    int status = PVK_OK;
    /* 预算：默认 maxit+20；只要残差仍在明显下降，就每 20 步再续 20 步，
     * 上限 4×maxit。
     *
     * 高偏压（>Voc）处 2-范数以 ~3%/步线性下降，200 步只够从 1e13 降到
     * ~1e3；旧逻辑（仅当 rel<1e-2 且 >2%/步时才多给 20 步）在 220 步判
     * 失败，随后触发 continuation 折半 + 冷启动重试风暴 —— 401/801 节点
     * 的 J-V 里有 76%~94% 的时间花在这上面。实测 maxit=800 时 401/801
     * 的 1.34~1.40 V 全部首次尝试即收敛（401@1.40 用 699 步），且 J 与
     * 细网格一致到 0.3%，说明解本身没问题、只是预算不够。 */
    int maxit_ext = maxit + 20;
    int maxit_cap = maxit * 4;
    if (maxit_cap < maxit_ext) maxit_cap = maxit_ext;
    double rn_chk = rn;

    for (int it = 0; it < maxit_ext; it++) {
        if (rn < tol) { converged = 1; break; }

        if (getenv("PVK_DEBUG"))
            fprintf(stderr, "[pvk] it=%d rn=%.6e\n", it, rn);

        /* ---- 数值雅可比（前向差分，只算被扰动节点 ±1 的三行） ----
         * 先按当前 u 预计算密度与 SRH（与 residual 完全一致），
         * 扰动 (s,jn) 只改变 jn 自身的 n/p/R，邻居行通过数组复用。 */
        for (int i = 0; i < n; i++) {
            node_dens(m, i, u, &nbuf[i], &pbuf[i]);
            if (!isfinite(nbuf[i])) nbuf[i] = 1e300;
            if (!isfinite(pbuf[i])) pbuf[i] = 1e300;
            double ni = m->ni[i];
            double np = nbuf[i] * pbuf[i];
            if (!isfinite(np)) np = 1e300;
            double dnp = np - ni * ni;
            double rad = m->brad[i] * dnp;
            if (!isfinite(rad)) rad = 0.0;
            double den = m->taup[i] * (nbuf[i] + ni) + m->taun[i] * (pbuf[i] + ni);
            rnet[i] = (den > 0.0 && isfinite(den) ? dnp / den : 0.0) + rad;
            if (!isfinite(rnet[i])) rnet[i] = 0.0;
        }

        memset(band, 0, sizeof(double) * PVK_BW_W * (size_t)nn);
        for (int jn = 0; jn < n; jn++) {
            for (int s = 0; s < 3; s++) {
                int j = s * n + jn;
                double save = u[j];
                double hstep = 1e-7 * (fabs(save) > 1.0 ? fabs(save) : 1.0);
                u[j] = save + hstep;

                double sn = nbuf[jn], sp = pbuf[jn], sr = rnet[jn];
                node_dens(m, jn, u, &nbuf[jn], &pbuf[jn]);
                if (!isfinite(nbuf[jn])) nbuf[jn] = 1e300;
                if (!isfinite(pbuf[jn])) pbuf[jn] = 1e300;
                {
                    double ni = m->ni[jn];
                    double np = nbuf[jn] * pbuf[jn];
                    if (!isfinite(np)) np = 1e300;
                    double dnp = np - ni * ni;
                    double rad = m->brad[jn] * dnp;
                    if (!isfinite(rad)) rad = 0.0;
                    double den = m->taup[jn] * (nbuf[jn] + ni)
                               + m->taun[jn] * (pbuf[jn] + ni);
                    rnet[jn] = (den > 0.0 && isfinite(den) ? dnp / den : 0.0) + rad;
                    if (!isfinite(rnet[jn])) rnet[jn] = 0.0;
                }

                double inv = 1.0 / hstep;
                for (int i = jn - 1; i <= jn + 1; i++) {
                    if (i < 0 || i >= n) continue;
                    double vloc[3];
                    residual_local(m, u, i, nbuf, pbuf, rnet, vloc);
                    for (int t = 0; t < 3; t++) {
                        int row = 3 * i + t;
                        int col = 3 * jn + s;
                        band[(size_t)row * PVK_BW_W + (col - row + PVK_BW_L)] =
                            (vloc[t] - r[t * n + i]) * inv;
                    }
                }
                u[j] = save;
                nbuf[jn] = sn; pbuf[jn] = sp; rnet[jn] = sr;
            }
        }

        /* b = -r（节点主序：3i+t） */
        for (int i = 0; i < n; i++)
            for (int t = 0; t < 3; t++)
                bvec[3 * i + t] = -r[t * n + i];

        if (band_solve(nn, band, bvec) != 0) {
            if (getenv("PVK_DEBUG"))
                fprintf(stderr, "[pvk] band solve failed at it=%d\n", it);
            status = PVK_ECONV;
            break;
        }
        for (int i = 0; i < n; i++)
            for (int s = 0; s < 3; s++)
                du[s * n + i] = bvec[3 * i + s];

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
            rn_new = residual(m, utmp, rtmp, nb2, pb2, rn2, NULL, NULL);
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

        /* 注：低光/暗态下 SG 通量的 double 舍入给残差设了 ~1e15 cm-3 s-1 的
         * 地板（缩放后 1e3~1e4），rn 到不了 tol。已试过两种放行方式：
         *   * 抬高 smax 到 1e-3*(净掺杂/tau)：1 mW/cm2 的 Jsc 会错 1.9 倍；
         *   * 步长判据 dumax<1e-6：1 mW/cm2 正确，但 0.1/0.01 mW/cm2 会停在
         *     错误解上（Jsc 偏小 1.5 倍、Voc 反而更高）。
         * 两者都不可接受，已回退。根因是源项尺度取了全器件最大值（暗态
         * 只剩 ni/τ）；现在改成 max(|R|,|G|) 并用总电流通量 |Jn+Jp|/hc
         * 参与归一（见 cont_scale）。
         * 另外试过"终端电流连续 2 次迭代相对变化 <1e-4"作为第二判据：
         * 端电流取自接触层旁的交界边，SG 一阶截断误差给它的噪声地板就是
         * 1e-5~1e-4（1 mW 的 V=0 实测 1.4e-5），正常收敛点会被大面积拒掉
         * （800 nm 1 sun 从 80/101 掉到 44/101），不可用。
         *
         * 0.1/0.01 mW（800 nm 卡）仍是 0/101。已试过并证伪的路径，勿重复：
         *   * 中心差分雅可比（h=1e-6，O(h²)）：1 sun 80→79 且 Voc 变 nan；
         *   * 电势步长限幅（ψ 0.25 V / φn,φp 0.13 V）：1 sun 80→67；
         *   * 种子加 V=0.02/0（从平衡态起步）、亮态 100×/10× 光强加码
         *     （后者能解到 10×，但压回 1× 仍失败）、rn 阈值放到 1e4。
         * 停滞点固定在中性区**多数**载流子行（p=4e16，n~1e-5），需要把
         * div Jp 解到 |Jn+Jp|/hc 的 1e-9 量级才够 1e-3；这是离散精度/
         * 条件数层面的限制，改判据只会放行错解（Jsc 曾错 1.9 倍）。 */

        /* 物理判据提前放行：通量相对残差（局部电流守恒/泊松/边界）已经
         * < 1e-3 时，2-范数可能仍被"按 n 或 p 归一"的少数载流子行撑在
         * O(10–1e3)。高注入下界面处 p 很大而空穴通量更大，这一行本来就
         * 只是噪声（绝对不平衡 < 产生率的 1e-3），再往下是 SG 在异质结
         * 上的一阶截断误差，Newton 压不动 —— 201/refine=2 在 V>1.1
         * 卡满 60 步就是这个原因。
         *
         * 必须同时要求 rn < 1e3：rel 只保证"相对当地通量"守恒，若某个
         * 局部通量被算得极大（错误解），rel 也可以很小。加上这条，低偏压
         * 下 rn~1e19 的病态解不会被误当成收敛（101/refine=1 的 J-V 曾
         * 因此出现 J 符号乱跳）。
         *
         * 不要放宽这个门槛：实测放到 1e6 会让 0.3~1.1 V 的点提前放行
         * （rel<1e-3 但 rn~1e4），J 系统性偏低 0.5%~4%，FF 掉 1.4 个点。
         * 高偏压的收敛问题由上面的预算续期解决 —— 续到 800 步时 401/801
         * 的 1.34~1.40 V 都能把 rn 压到 ~997（刚好在 1e3 以下）。 */
        {
            double rel_now = 0.0, hid_now = 0.0;
            residual(m, u, utmp, nb2, pb2, rn2, &rel_now, &hid_now);
            /* hid_now：被 eq_minority 替换的行按真实方程算的最大不平衡度。
             * 1500nm 亮态的塌缩解（吸收层全层钉扎）rel/rn 双假收敛，
             * 但隐藏行不平衡 ~O(1)（产生率无处泄放），在这里被拦下。 */
            if (rel_now < 1e-3 && rn < 1e3 && hid_now < 1e-2) converged = 1;
            if (converged && getenv("PVK_DEBUG"))
                fprintf(stderr, "[pvk] relaxed accept: rel=%.2e rn=%.2e hid=%.2e\n",
                        rel_now, rn, hid_now);
            if (converged) break;
        }

        /* 步长停滞后再用相对不平衡度确认，避免把"伪收敛"当收敛。
         * 阈值 1e-2 比上面的提前放行（1e-3）宽松：这里 Newton 已经走不动，
         * 只能确认解没有明显不平衡；正常收敛仍靠 rn<tol 或上面的 1e-3。 */
        if (alpha * dumax < 1e-7) {
            double rel_now = 0.0, hid_now = 0.0;
            residual(m, u, utmp, nb2, pb2, rn2, &rel_now, &hid_now);
            converged = (rel_now < 1e-2 && hid_now < 1e-2) ? 1 : 0;
            if (!converged && getenv("PVK_DEBUG"))
                fprintf(stderr, "[pvk] 步长停滞但 rel=%.3e hid=%.3e，判为未收敛\n",
                        rel_now, hid_now);
            break;
        }

        /* 预算续期：每 20 步检查一次，残差至少降 20% 才续。 */
        if (it + 1 >= maxit_ext && maxit_ext < maxit_cap && rn < 0.8 * rn_chk) {
            maxit_ext += 20;
            rn_chk = rn;
        }
    }

    if (!converged && getenv("PVK_DEBUG")) {
        fprintf(stderr, "[pvk] not converged: status=%d iters=%d rn=%.3e\n",
                status, iters, rn);
        if (isfinite(rn)) dump_worst_rows(m, r, u, nn);
    }

    /* 诊断输出：最终解的无量纲相对不平衡度（判据见循环尾部）。 */
    {
        double rel = 1e300, hid = 0.0;
        residual(m, u, r, nbuf, pbuf, rnet, &rel, &hid);
        if (getenv("PVK_DEBUG")) {
            fprintf(stderr, "[pvk] rel_residual=%.3e hid=%.3e (converged=%d)\n",
                    rel, hid, converged);
            if (!isfinite(rel)) {
                for (int i2 = 0; i2 < nn; i2++) {
                    if (!isfinite(r[i2])) {
                        fprintf(stderr,
                                "[pvk] non-finite row %d (node %d, kind %d) r=%g "
                                "u=%.6g %.6g %.6g\n",
                                i2, i2 % n, i2 / n, r[i2],
                                u[i2 % n], u[n + (i2 % n)], u[2*n + (i2 % n)]);
                        break;
                    }
                }
            }
        }
    }

    out->iterations = iters;
    out->residual = rn;
    out->converged = converged ? 1 : 0;

    pvk__free(r); pvk__free(rtmp); pvk__free(utmp); pvk__free(du);
    pvk__free(band); pvk__free(bvec);
    pvk__free(nbuf); pvk__free(pbuf); pvk__free(rnet);
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
                          double* nb, double* pb, double* relrow_out) {
    int n = m->n;
    double Vt = m->Vt;
    double rsum2 = 0.0;
    double relsum2 = 0.0;

    for (int i = 0; i < n; i++) {
        eq_dens(m, i, u[i], &nb[i], &pb[i]);
        if (!isfinite(nb[i])) nb[i] = 1e300;
        if (!isfinite(pb[i])) pb[i] = 1e300;
    }

    /* 线搜索用的 merit：全器件统一尺度 Cref 归一。
     * 逐节点归一化的 den 会随迭代变化，界面节点 den~1e20、体内节点
     * den~1e-5，舍入噪声就能让 RMS 不下降，Newton 方向并非它的下降
     * 方向 —— refine>0 时 homotopy 在 t=0 就卡死就是这个原因。
     * 统一尺度等价于给方程乘常数，不改变 Newton 方向。 */
    double Cref = 1e10;
    for (int i = 0; i < n; i++) {
        double s = m->Nc[i] > m->Nv[i] ? m->Nc[i] : m->Nv[i];
        double d = m->Nd[i] + m->Na[i];
        if (d > s) s = d;
        if (s > Cref) Cref = s;
    }

    for (int i = 0; i < n; i++) {
        if (m->bc[i]) {
            r[i] = u[i] - m->u_bc[m->bc[i] - 1];
            rsum2 += r[i] * r[i];
            relsum2 += r[i] * r[i];
            continue;
        }
        double hL = m->h[i-1], hR = m->h[i], hc = m->hc[i];
        double fluxR = m->eps[i+1] * (u[i+1] - u[i]) * Vt / hR;
        double fluxL = m->eps[i]   * (u[i]   - u[i-1]) * Vt / hL;
        double ft = (fluxR - fluxL) / (PVK_Q_C * hc);
        double ct = pb[i] - nb[i] + m->Nd[i] - m->Na[i];
        r[i] = ft + ct;
        double a = r[i] / Cref;
        rsum2 += a * a;
        /* 逐节点相对残差只做“收敛确认”，不参与线搜索：界面节点的通量
         * 尺度 ~1e20、体内 ~1e-5，用它当 merit 会让线搜索永远不下降。
         * 下限取 1e10 cm⁻³（而不是 1）：绝对残差即使已到机器精度
         * （rn~1e-17 即 ~1e3 cm⁻³），除以下限 1 也会得到 O(1) 的假
         * “不平衡”。1e10 cm⁻³ 折算到 1 nm 网格只有 ~1e-3 V/cm 的场误差。 */
        double den = fabs(ft);
        if (fabs(ct) > den) den = fabs(ct);
        if (den < 1e10) den = 1e10;
        double ar = r[i] / den;
        relsum2 += ar * ar;
    }
    if (relrow_out) {
        *relrow_out = isfinite(relsum2) ? sqrt(relsum2 / (double)n) : 1e300;
    }
    if (!isfinite(rsum2)) return 1e300;
    return sqrt(rsum2 / (double)n);
}

/* 单行 Poisson 残差 —— 与 eq_residual 里的 r[i] 逐项一致。
 * 只依赖 u[i-1..i+1] 与当地 nb[i]/pb[i]，所以数值雅可比扰动 u[j] 时
 * 只有 j-1..j+1 三行会变，邻居的密度数组可以复用（O(n) 组装）。 */
static double eq_row(const pvk_model* m, const double* u, int i,
                     const double* nb, const double* pb) {
    if (m->bc[i]) return u[i] - m->u_bc[m->bc[i] - 1];
    double Vt = m->Vt;
    double hL = m->h[i-1], hR = m->h[i], hc = m->hc[i];
    double fluxR = m->eps[i+1] * (u[i+1] - u[i]) * Vt / hR;
    double fluxL = m->eps[i]   * (u[i]   - u[i-1]) * Vt / hL;
    double ft = (fluxR - fluxL) / (PVK_Q_C * hc);
    double ct = pb[i] - nb[i] + m->Nd[i] - m->Na[i];
    return ft + ct;
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

/* 从给定初值解非线性 Poisson（只解 ψ，φn=φp=0），返回最终归一化残差。
 * relrow_out（可空）返回最后的逐节点相对残差，供 homotopy 判质量。 */
static double solve_poisson_newton(const pvk_model* m, double* u,
                                   const pvk_solver_opts* opts,
                                   double* relrow_out) {
    int n = m->n;
    double rn = 1e300;
    if (relrow_out) *relrow_out = 1e300;
    double* r  = (double*)pvk__malloc(sizeof(double) * n);
    double* rt = (double*)pvk__malloc(sizeof(double) * n);
    double* ut = (double*)pvk__malloc(sizeof(double) * n);
    double* nb = (double*)pvk__malloc(sizeof(double) * n);
    double* pb = (double*)pvk__malloc(sizeof(double) * n);
    double* nb2 = (double*)pvk__malloc(sizeof(double) * n);
    double* pb2 = (double*)pvk__malloc(sizeof(double) * n);
    double* ab = (double*)pvk__malloc(sizeof(double) * PVK_TRI_W * (size_t)n);
    double* du = (double*)pvk__malloc(sizeof(double) * n);
    if (!r || !rt || !ut || !nb || !pb || !nb2 || !pb2 || !ab || !du) goto done_free;

    double tol = opts && opts->res_tol > 0 ? opts->res_tol : 1e-8;
    double rr = 1e300;
    rn = eq_residual(m, u, r, nb, pb, &rr);

    /* 双判据：全局尺度 rn 保证“大项”都解到位，逐节点相对 rr 保证体内
     * 那些绝对值很小、但相对自身量级仍差很多的方程也解到位。只用 rn
     * 会在 1e-8 就放行（此时界面节点的通量还差 ~80%），亮态随后发散；
     * 只用 rr 则线搜索不下降（见 eq_residual 的注释）。 */
    for (int it = 0; it < 80 && (rn > tol || rr > 1e-3); it++) {
        memset(ab, 0, sizeof(double) * PVK_TRI_W * (size_t)n);
        /* 数值雅可比（前向差分）：Poisson 每行只耦合 i-1..i+1，所以
         * 扰动 u[j] 只重算 j-1..j+1 三行，O(n)（旧版每列都重扫全器件
         * 是 O(n^2)，再叠加 dense_solve 的 O(n^3)）。 */
        for (int j = 0; j < n; j++) {
            double save = u[j];
            double sn = nb[j], sp = pb[j];
            double hstep = 1e-7 * (fabs(save) > 1.0 ? fabs(save) : 1.0);
            u[j] = save + hstep;
            eq_dens(m, j, u[j], &nb[j], &pb[j]);
            if (!isfinite(nb[j])) nb[j] = 1e300;
            if (!isfinite(pb[j])) pb[j] = 1e300;
            double inv = 1.0 / hstep;
            for (int i = j - 1; i <= j + 1; i++) {
                if (i < 0 || i >= n) continue;
                ab[(size_t)i * PVK_TRI_W + (j - i + 1)] =
                    (eq_row(m, u, i, nb, pb) - r[i]) * inv;
            }
            u[j] = save;
            nb[j] = sn; pb[j] = sp;
        }
        for (int i = 0; i < n; i++) du[i] = -r[i];
        if (tri_solve(n, ab, du) != 0) {
            if (getenv("PVK_DEBUG")) fprintf(stderr, "[pvk] eq: tri_solve failed it=%d\n", it);
            break;
        }

        /* 步长限幅（≈3 V）：界面处带边差可达 ~2 eV（≈80 Vt），限得太死
         * 会把 Newton 方向截断成非下降方向，线搜索在 it=0 就必然失败。 */
        double dumax = 0.0;
        for (int i = 0; i < n; i++) {
            if (m->bc[i]) continue;
            if (du[i] > 120.0) du[i] = 120.0;
            if (du[i] < -120.0) du[i] = -120.0;
            if (fabs(du[i]) > dumax) dumax = fabs(du[i]);
        }

        double alpha = 1.0, rnew = rn, rrnew = rr;
        int ok = 0;
        for (int t = 0; t < 16; t++) {
            for (int i = 0; i < n; i++) ut[i] = u[i] + alpha * du[i];
            rnew = eq_residual(m, ut, rt, nb2, pb2, &rrnew);
            if (isfinite(rnew) && rnew <= rn * (1.0 - 1e-4 * alpha)) { ok = 1; break; }
            alpha *= 0.5;
        }
        if (!ok) {
            if (getenv("PVK_DEBUG")) {
                fprintf(stderr, "[pvk] eq: line search failed it=%d rn=%.3e dumax=%.3e\n",
                        it, rn, dumax);
                for (int k = 0; k < 3; k++) {
                    int imax = -1; double vmax = 0.0;
                    for (int i = 0; i < n; i++) {
                        double v = fabs(r[i]);
                        if (v > vmax) { vmax = v; imax = i; }
                    }
                    if (imax < 0) break;
                    fprintf(stderr,
                            "[pvk]   worst r=%.3e node=%d layer=%d x=%.3f nm u=%.4f\n",
                            r[imax], imax, m->node_layer[imax],
                            m->x[imax] * 1e7, u[imax]);
                    r[imax] = 0.0;
                }
            }
            break;
        }
        memcpy(u, ut, sizeof(double) * n);
        for (int i = 0; i < n; i++) r[i] = rt[i];
        /* nb2/pb2 是线搜索那次 eq_residual 在 ut 上算的密度，必须同步回来：
         * 下一步的局部雅可比要复用它（旧版每列都重算全器件，所以不需要）。 */
        memcpy(nb, nb2, sizeof(double) * n);
        memcpy(pb, pb2, sizeof(double) * n);
        rn = rnew;
        rr = rrnew;
        /* 注意：不能用 du[0] 判停 —— 节点 0 是钉死的 Dirichlet 行，
         * du[0] 恒为 0，会让平衡态 Newton 只走一步（这正是暗态初值
         * 一直不收敛的根因之一）。用自由节点的最大步长。 */
        if (alpha * dumax < 1e-10) break;
    }
    if (relrow_out) *relrow_out = rr;
    if (getenv("PVK_DEBUG")) {
        if (relrow_out)
            fprintf(stderr, "[pvk] solve_equilibrium: rn=%.3e relrow=%.3e\n",
                    rn, rr);
        else
            fprintf(stderr, "[pvk] solve_equilibrium: rn=%.3e\n", rn);
    }

done_free:
    pvk__free(r); pvk__free(rt); pvk__free(ut);
    pvk__free(nb); pvk__free(pb); pvk__free(nb2); pvk__free(pb2);
    pvk__free(ab); pvk__free(du);
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
            if (did) {
                memcpy(u, usave, sizeof(double) * n);
            } else {
                /* 首个 homotopy 台阶（homo=0：全器件本征、同一带结构）：
                 * 泊松方程的首阶解就是两接触之间的线性电势。旧代码用电
                 * 中性初值把所有自由节点置 ψ=0，等于把 V_app 整个压到紧邻
                 * 接触的第一层网格上 —— 网格越细这个台阶的相对跨度越大，
                 * nodes=797（1.28 nm）时 Newton 在 it=0 就线搜索失败，
                 * homotopy 直接停在 t=0。线性插值让接触附近的电势梯度与
                 * 网格无关。
                 *
                 * 锚点必须取**首末自由节点**：理想 Ohmic 的 Dirichlet 行
                 * 被内移到紧邻电极的半导体节点上，若仍按 x[0]/x[n-1] 插值，
                 * 那个 pin 住的节点与其后第一个自由节点（refine>0 时相距
                 * 0.1 nm）之间会多出一个 V_app 的台阶，Poisson 残差 1e20。 */
                int ifree = 0, ilast = n - 1;
                while (ifree + 1 < n && lm.bc[ifree]) ifree++;
                while (ilast > 0 && lm.bc[ilast]) ilast--;
                double uL = lm.u_bc[0];
                double uR = lm.u_bc[1];
                double x0 = lm.x[ifree], x1 = lm.x[ilast];
                double dx = (x1 > x0) ? (x1 - x0) : 1.0;
                for (int i = 0; i < n; i++) {
                    if (lm.bc[i]) continue;
                    u[i] = uL + (uR - uL) * (lm.x[i] - x0) / dx;
                }
            }
            /* psi_a 随 homo 变，钉死节点每次重试都重置到本步边界值 */
            for (int i = 0; i < n; i++)
                if (lm.bc[i]) u[i] = lm.u_bc[lm.bc[i] - 1];

            double relrow = 1e300;
            double rn = solve_poisson_newton(&lm, u, opts, &relrow);
            model_free(&lm);

            /* 这里是“初值的初值”，不要求解到机器精度，但必须真的是
             * 平衡态：只看全局尺度 rn 会在体内电荷仍差 ~1e17 cm⁻³ 时
             * 就放行，亮态随后在 TiO2/钙钛矿界面处把 n 推到 1e25 发散。
             * 所以再加一道逐节点相对残差（<1e-3）。 */
            if (rn < 1e-3 && relrow < 1e-3) {
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

/* 标记接触层（隐式欧姆接触）。
 *
 * 从紧邻电极的第一个自由节点往里走，只要该层的**平衡**载流子密度
 * （φ=0，只由 ψ 决定）还比 max(ni,1) 高 3 个数量级以上，就把它归入
 * 接触层：这层即电极注入的积累层，厚度由德拜长度决定（~0.1–10 nm），
 * 网格解析不了，连续性方程在这一层没有可靠的离散解。归入接触后，
 * **多数**载流子的 φ 钉在电极值上（无限复合），**少数**载流子的 φ
 * 沿接触层平到体内（∂φ/∂x=0 ⇔ J=0，即被阻挡的选择性接触），ψ 仍由
 * Poisson 求解。
 *
 * 用平衡密度而不是当前密度：亮态下吸收层的 n 可以很高，但它不是
 * 接触层，必须保留完整的双载流子方程。走到层边界就停，避免把 ETL/HTL
 * 之外的吸收层卷进来。
 *
 * 【已知物理后果：亮态理想因子 >2】本模型把整个简并 ETL/HTL 归入
 * 钉扎层后，选择性提取集中在单网格节点完成，钉扎区内 SRH 恒为 0。
 * Voc 处体内 SRH≈总产生仍是主导，但环流电流（=Jsc）把
 * 吸收层边缘撕出随光强增大的 n/p 失衡（1 sun 时前端 p/n≈115），
 * SRH=np/(τ(n+p)) 中 (n+p) 增速超过 e^(V/2Vt)，导致 dVoc/dlnG 的
 * 表观理想因子升到 ~3（0.1–1 sun 段峰值），偏离高注入 SRH 理论值 2。
 * 已验证与网格无关（101/1 与 201/2 差 <0.05），非串联电阻（Voc 处
 * J=0），也非插值伪影（二分法精确求 Voc 同值）。回归锚点见
 * python/scripts/analytic_benchmarks.py 第 7 项。
 *
 * 少子边界改成零电流（选择性接触）后锚点一位数字都没变：钉扎层里少子
 * 密度只有 1e-33 量级，它的 φ 无论钉在电极还是平到体内，对 Poisson
 * 电荷和界面通量的贡献都小到可以忽略——这条边界方程实际是个哑变量。
 * 实测真正的瓶颈在吸收层一侧：1.87 eV 价带台阶把空穴堆在紧邻界面
 * 的头几个自由节点上（正偏 1.3 V 时 p≈9e18），那里 SG 通量的截断误差
 * 让空穴连续性行的相对残差卡在 0.6 左右，Newton 再也压不下去。 */
static void mark_contact_layer(pvk_ctx* ctx, pvk_model* m, double* u, int warm) {
    int n = m->n;
    memset(m->cpl, 0, sizeof(unsigned char) * (size_t)n);

    int i = 0;
    while (i < n && m->bc[i]) i++;
    if (i < n) {
        int lay = m->node_layer[i];
        for (; i < n && !m->bc[i] && m->node_layer[i] == lay; i++) {
            double nn, pp, thr;
            eq_dens(m, i, u[i], &nn, &pp);
            thr = 1e3 * (m->ni[i] > 1.0 ? m->ni[i] : 1.0);
            if (!(fmax(nn, pp) > thr)) break;
            m->cpl[i] = 1;
            m->cpl_maj[i] = (nn >= pp) ? 1 : 2;
        }
    }

    int j = n - 1;
    while (j >= 0 && m->bc[j]) j--;
    if (j >= 0) {
        int lay = m->node_layer[j];
        for (; j >= 0 && !m->bc[j] && m->node_layer[j] == lay; j--) {
            double nn, pp, thr;
            eq_dens(m, j, u[j], &nn, &pp);
            thr = 1e3 * (m->ni[j] > 1.0 ? m->ni[j] : 1.0);
            if (!(fmax(nn, pp) > thr)) break;
            m->cpl[j] = 2;
            m->cpl_maj[j] = (nn >= pp) ? 1 : 2;
        }
    }

    /* 掩码随偏压收缩：上一步还被钉在接触层里、这一步被释放的节点，
     * 其 φn/φp 仍是**电极的**准费米能级（钉扎值）。拿它当暖启动初值，
     * 相当于让这些节点的载流子密度停在平衡值附近，而连续性方程要求的
     * 是偏压下的密度，残差直接爆到 1e63 —— Cs2AgBiBr6 复刻卡在
     * 0.90→0.92 V 阳极掩码从 20 个节点骤降到 6 个，之后整段扫描全
     * 发散就是这个原因。这些节点改从相邻自由节点取 φ（两侧都有就取
     * 平均），ψ 仍用上一步的值。 */
    if (warm && ctx->cpl_prev && ctx->cpl_prev_n == n) {
        for (int k = 0; k < n; k++) {
            if (m->bc[k] || m->cpl[k] || !ctx->cpl_prev[k]) continue;
            /* 只掉一个边界节点（掩码收缩一格）时，电极准费米能级仍是好初值，
             * 保持原行为；成片释放说明发生了整层翻转，才需要重新起算。 */
            int adj_pin = (k > 0 && m->cpl[k-1]) || (k + 1 < n && m->cpl[k+1]);
            int adj_rel = (k > 0 && !m->cpl[k-1] && ctx->cpl_prev[k-1]) ||
                          (k + 1 < n && !m->cpl[k+1] && ctx->cpl_prev[k+1]);
            if (adj_pin && !adj_rel) continue;
            int kl = -1, kr = -1;
            for (int q = k - 1; q >= 0; q--)
                if (!m->bc[q] && !m->cpl[q]) { kl = q; break; }
            for (int q = k + 1; q < n; q++)
                if (!m->bc[q] && !m->cpl[q]) { kr = q; break; }
            if (kl < 0 && kr < 0) continue;
            if (kl >= 0 && kr >= 0) {
                u[n + k]     = 0.5 * (u[n + kl] + u[n + kr]);
                u[2 * n + k] = 0.5 * (u[2*n + kl] + u[2*n + kr]);
            } else {
                int q = (kl >= 0) ? kl : kr;
                u[n + k]     = u[n + q];
                u[2 * n + k] = u[2*n + q];
            }
            m->cpl_released++;
        }
    }

    /* 记下本次掩码，供下一个偏压点识别"被释放"的节点 */
    if (!ctx->cpl_prev || ctx->cpl_prev_n != n) {
        pvk__free(ctx->cpl_prev);
        ctx->cpl_prev = (unsigned char*)pvk__malloc(sizeof(unsigned char) * (size_t)n);
        ctx->cpl_prev_n = ctx->cpl_prev ? n : 0;
    }
    if (ctx->cpl_prev && ctx->cpl_prev_n == n)
        memcpy(ctx->cpl_prev, m->cpl, sizeof(unsigned char) * (size_t)n);

    if (getenv("PVK_DEBUG")) {
        int c1 = 0, c2 = 0, k0 = -1, k1 = -1, k2 = -1, k3 = -1;
        for (int k = 0; k < n; k++) {
            if (m->cpl[k] == 1) { c1++; if (k0 < 0) k0 = k; k1 = k; }
            else if (m->cpl[k] == 2) { c2++; if (k2 < 0) k2 = k; k3 = k; }
        }
        fprintf(stderr, "[pvk] contact layer: cathode %d nodes (%.1f..%.1f nm, lay=%d, maj=%d), "
                        "anode %d nodes (%.1f..%.1f nm, lay=%d, maj=%d)\n",
                c1, k0 >= 0 ? m->x[k0]*1e7 : -1.0, k1 >= 0 ? m->x[k1]*1e7 : -1.0,
                k0 >= 0 ? m->node_layer[k0] : -1,
                k1 >= 0 ? m->cpl_maj[k1] : -1,
                c2, k2 >= 0 ? m->x[k2]*1e7 : -1.0, k3 >= 0 ? m->x[k3]*1e7 : -1.0,
                k2 >= 0 ? m->node_layer[k2] : -1,
                k2 >= 0 ? m->cpl_maj[k2] : -1);
    }
}

/* 接触层节点的 φ：多数载流子取电极费米能级（与 residual 中的方程一致），
 * 少子沿接触层平到器件内部（选择性接触，零少子电流）。传播方向必须
 * 从内侧往外，否则链上的值还是上一轮留下的。 */
static void apply_contact_layer_qfl(const pvk_model* m, double* u) {
    int n = m->n;
    for (int i = n - 1; i >= 0; i--) {          /* 阴极侧：内侧 → 外侧 */
        if (m->cpl[i] != 1) continue;
        int maj = m->cpl_maj ? m->cpl_maj[i] : 0;
        if (maj == 2) {
            u[2 * n + i] = m->wp_bc[0];
            u[n + i]     = (i + 1 < n) ? u[n + i + 1] : m->wn_bc[0];
        } else if (maj == 1) {
            u[n + i]     = m->wn_bc[0];
            u[2 * n + i] = (i + 1 < n) ? u[2*n + i + 1] : m->wp_bc[0];
        } else {
            u[n + i]     = m->wn_bc[0];
            u[2 * n + i] = m->wp_bc[0];
        }
    }
    for (int i = 0; i < n; i++) {               /* 阳极侧：内侧 → 外侧 */
        if (m->cpl[i] != 2) continue;
        int maj = m->cpl_maj ? m->cpl_maj[i] : 0;
        if (maj == 1) {
            u[n + i]     = m->wn_bc[1];
            u[2 * n + i] = (i > 0) ? u[2*n + i - 1] : m->wp_bc[1];
        } else if (maj == 2) {
            u[2 * n + i] = m->wp_bc[1];
            u[n + i]     = (i > 0) ? u[n + i - 1] : m->wn_bc[1];
        } else {
            u[n + i]     = m->wn_bc[1];
            u[2 * n + i] = m->wp_bc[1];
        }
    }
}

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
            /* 掩码在首次冷启动时按平衡态冻结，暖启动只套用，不重算。
             * 收缩时被释放的节点由 mark_contact_layer 内部改初值。 */
            if (!m->cpl_frozen) mark_contact_layer(ctx, m, u, 1);
            apply_contact_layer_qfl(m, u);
            return;
        }
    }

    /* 冷启动：先用非线性 Poisson 解出平衡态 ψ（带边 homotopy），
     * 再令 φn=φp=0。局域电中性初值在有 ~1 V 内建电势的器件里离真解
     * 差几十个 Vt，暗态（无光生载流子抬高密度）Newton 一步就能把残差
     * 推到 1e25 以上。 */
    equilibrium_guess(ctx, dev, m, u, &ctx->solver_opts);

    /* 光照下初值必须包含光生载流子。平衡态 homotopy 给的是 np=ni²，
     * 而亮态稳态要求 np ≈ ni² + G·τ；少这一项时少子密度只有 ~ni，
     * 空穴连续性方程第一步就要求它跨十几个数量级，Newton 必然发散。
     *
     * 劈裂必须相对**当地平衡 np**（而不是 ni²）来定，并且只抬少子：
     * 接触积累层里 n0 可比 ni 高十几个数量级，按 ni² 算出的 split
     * （TiO2 里可达 1.4 eV）再均分到两端，会把 n 抬到 1e31 —— 亮态
     * V=0 冷启动的初始残差 1e39 就是这么来的。只抬少子则多数载流子
     * 不动，np 恰好达标。 */
    for (int i = 0; i < n; i++) {
        if (m->bc[i] || !(m->G[i] > 0.0)) continue;
        double ni = m->ni[i];
        if (!(ni > 0.0)) continue;
        double n0, p0;
        node_dens(m, i, u, &n0, &p0);
        double np0 = n0 * p0;
        if (!(np0 > 0.0) || !isfinite(np0)) continue;
        double tau = 1.0 / (1.0 / m->taun[i] + 1.0 / m->taup[i]);
        /* 辐射复合与 SRH 并联：np-ni² = G·τ/(1+B·τ)。漏掉 B·τ 会把初值
         * 的目标 np 抬得过高（劈裂偏大），亮态首步残差反而变大。 */
        double np_t = ni * ni + m->G[i] * tau / (1.0 + m->brad[i] * tau);
        double split = Vt * log(np_t / np0);
        if (!(split > 0.0) || !isfinite(split)) continue;
        if (split > 1.2) split = 1.2;   /* 上限 ≈ 1.2 eV，避免初值过冲 */
        if (n0 >= p0) {
            u[2*n + i] = -split / Vt;   /* n 型：抬空穴（少子） */
        } else {
            u[n + i]   =  split / Vt;   /* p 型：抬电子（少子） */
        }
    }

    /* 电极层所有节点钉死 */
    for (int i = 0; i < n; i++) {
        if (!m->bc[i]) continue;
        int e = m->bc[i] - 1;
        u[i]         = m->u_bc[e];
        u[n + i]     = m->wn_bc[e];
        u[2*n + i]   = m->wp_bc[e];
    }

    mark_contact_layer(ctx, m, u, 0);
    m->cpl_frozen = 1;
    apply_contact_layer_qfl(m, u);
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
            double dnp = np - ni * ni;
            double den = m->taup[i] * (nbuf[i] + ni) + m->taun[i] * (pbuf[i] + ni);
            rnet[i] = (den > 0 ? dnp / den : 0.0) + m->brad[i] * dnp - m->G[i];
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

    /* 终端电流：取「电极层/半导体」交界那两条边上的通量。
     * 电极内部节点全被 Dirichlet 钉死，那里的边电流恒为 0；体内的 J
     * 又会因产生/复合随位置变化（dJn/dx=q(R-G)，dJp/dx=-q(R-G)，但
     * Jn+Jp 守恒）。旧实现把全部 n-1 条边（含两段电极）做无权平均，
     * 得到的既不是终端电流，也随网格/加密方式漂移 —— 这就是同一器件
     * 在 101/201/397/797 上 Jsc 各不相同、一开界面加密就跳到 4.2 的来源。 */
    int iL = -1, iR = -1;
    for (int i = 0; i < n - 1; i++) {
        int pinL = m->bc[i] || m->cpl[i];
        int pinR = m->bc[i+1] || m->cpl[i+1];
        if (pinL && !pinR && iL < 0) iL = i;
        if (!pinL && pinR) iR = i;
    }
    if (J_out) {
        if (iL >= 0 && iR >= 0) {
            double JL = jne[iL] + jpe[iL];
            double JR = jne[iR] + jpe[iR];
            *J_out = 0.5 * (JL + JR);
            /* 收敛解两端的终端电流必须相等；不等说明连续方程没解到位 */
            if (getenv("PVK_DEBUG"))
                fprintf(stderr, "[pvk] terminal J: cathode=%.6g anode=%.6g\n", JL, JR);
        } else if (n > 1) {
            double jsum = 0.0;
            for (int i = 0; i < n - 1; i++) jsum += jne[i] + jpe[i];
            *J_out = jsum / (double)(n - 1);
        } else {
            *J_out = 0.0;
        }
    }

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
    if (fabs(ctx->mesh.sig_refine - dev->interface_refine) > 1e-12) return 0;
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

/* 冷启动重试的递归深度（只为防无限递归；正常最多 1 层）。 */
static int g_retry_depth = 0;

/* 上一次 optics_compute 的输入是否与本次相同（device + optics）。
 * 只做逐字段比较；wl_nm/flux 只比指针（调用方在整个扫描里复用同一组
 * numpy 数组，而解引用 last_optics 里借用的指针可能已经悬空）。 */
static int same_optics_inputs(const pvk_ctx* ctx, const pvk_device* dev,
                              const pvk_optics_in* opt) {
    if (!ctx->inputs_valid || !ctx->optics_valid) return 0;
    const pvk_device* d0 = &ctx->last_device;
    if (d0->layers == NULL || d0->n_layers != dev->n_layers) return 0;
    if (d0->T_K != dev->T_K || d0->nodes != dev->nodes ||
        d0->interface_refine != dev->interface_refine) return 0;
    for (int i = 0; i < dev->n_layers; i++) {
        const pvk_layer* a = &d0->layers[i];
        const pvk_layer* b = &dev->layers[i];
        if (a->thick_nm != b->thick_nm || a->eps_r != b->eps_r ||
            a->Eg_eV != b->Eg_eV || a->chi_eV != b->chi_eV ||
            a->Nc_cm3 != b->Nc_cm3 || a->Nv_cm3 != b->Nv_cm3 ||
            a->mun_cm2Vs != b->mun_cm2Vs || a->mup_cm2Vs != b->mup_cm2Vs ||
            a->taun_s != b->taun_s || a->taup_s != b->taup_s ||
            a->Nd_cm3 != b->Nd_cm3 || a->Na_cm3 != b->Na_cm3 ||
            a->is_contact != b->is_contact || a->is_gen != b->is_gen ||
            a->b_rad_cm3_s != b->b_rad_cm3_s) return 0;
        /* material 决定 n,k 查表 */
        if ((a->material == NULL) != (b->material == NULL)) return 0;
        if (a->material && strcmp(a->material, b->material) != 0) return 0;
    }
    const pvk_optics_in* o0 = &ctx->last_optics;
    if (o0->n_wl != opt->n_wl ||
        o0->intensity_mW_cm2 != opt->intensity_mW_cm2 ||
        o0->bounces != opt->bounces ||
        o0->bandgap_mask != opt->bandgap_mask ||
        o0->coherent != opt->coherent || o0->atlas_ray != opt->atlas_ray)
        return 0;
    if (o0->wl_nm != opt->wl_nm || o0->flux != opt->flux) return 0;
    if ((o0->nk_dir == NULL) != (opt->nk_dir == NULL)) return 0;
    if (o0->nk_dir && strcmp(o0->nk_dir, opt->nk_dir) != 0) return 0;
    return 1;
}

PVK_API int pvk_solve_dc(pvk_ctx* ctx,
                         const pvk_device* dev,
                         const pvk_optics_in* opt,
                         const pvk_bias* bias,
                         pvk_solve_info* out_info) {
    if (!ctx || !dev || !opt || !bias || !out_info) return PVK_EINVAL;
    if (dev->n_layers <= 0 || dev->layers == NULL) return PVK_EINVAL;

    int optics_dirty = !same_optics_inputs(ctx, dev, opt);
    if (!mesh_matches(ctx, dev)) {
        int rc = pvk_build_mesh(ctx, dev);
        if (rc != PVK_OK) return rc;
        optics_dirty = 1;
    }
    pvk__cache_inputs(ctx, dev, opt, bias);

    /* 光学：只在 device/optics 变化时重算（141 点扫描里省 0.6~1.1 s，
     * 201 节点时这一项占端到端 16%）。intensity=0 时 G 全零。 */
    if (opt->n_wl > 0 && opt->wl_nm && opt->flux) {
        if (optics_dirty) {
            int rc = pvk_optics_compute(ctx, dev, opt, NULL, NULL);
            if (rc != PVK_OK) return rc;
        }
        ctx->optics_valid = 1;
    } else {
        double* g = ctx->state.fields[PVK_FIELD_G_OPT];
        if (g) memset(g, 0, sizeof(double) * ctx->mesh.n_node);
        ctx->optics_valid = 1;
    }

    int n = ctx->mesh.n_node;
    double V_app = bias->anode_V - bias->cathode_V;
    /* 分支跳变防护用的"上一点"电流（要在本次求解改写 last_J 之前取） */
    int    had_prev = ctx->have_solution;
    double J_prev   = ctx->last_J;
    double V_prev0  = ctx->last_V_applied;

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
     * 直接解失败时按顺序退：
     *  1) τ homotopy：SRH 寿命 ×1e5 → ×1，先把少子堆到高注入；
     *  2) 产生率 homotopy：亮态光强 0.01→1 爬坡；暗态用吸收层里的
     *     合成产生率 1e20 cm⁻³s⁻¹ 按十进位压到 0。
     * 每一步都先解完整双载流子方程，不收敛才用 φ_min=φ_maj 兜底。 */
    if (!ctx->have_solution) {
        static const double seeds[] = {0.15, 0.25, 0.35, 0.10, 0.05};
        static const double ramp[]  = {1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2,
                                       0.05, 0.15, 0.35, 0.6, 0.85, 1.0};
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

                /* 2) τ homotopy：把 SRH 寿命放大 1e5 倍（复合几乎关闭），
                 *    少子可以从平衡态的 ni 直接堆到高注入水平，再按十进位
                 *    压回真实寿命，每一步用上一步的解做初值。
                 *    亮态直接解失败的根因就是少子停在 1e-29 量级、Newton
                 *    一步要跨几十个数量级；先关掉复合比光强爬坡更对症。 */
                if (!ok) {
                    memcpy(u, ubak, sizeof(double) * 3 * n);
                    static const double tscale[] = {1e5, 1e4, 1e3, 1e2, 10.0, 1.0};
                    const int nts = (int)(sizeof(tscale) / sizeof(tscale[0]));
                    double* tn = (double*)pvk__malloc(sizeof(double) * n);
                    double* tp = (double*)pvk__malloc(sizeof(double) * n);
                    if (tn && tp) {
                        memcpy(tn, m.taun, sizeof(double) * n);
                        memcpy(tp, m.taup, sizeof(double) * n);
                        for (int k = 0; k < nts; k++) {
                            for (int i = 0; i < n; i++) {
                                m.taun[i] = tn[i] * tscale[k];
                                m.taup[i] = tp[i] * tscale[k];
                            }
                            memset(&rinfo, 0, sizeof(rinfo));
                            rinfo.n = n;
                            solve_newton(&m, u, &ctx->solver_opts, &rinfo);
                            ok = rinfo.converged;
                            if (!ok && rinfo.residual < 1e-3) ok = 1;
                            if (getenv("PVK_DEBUG"))
                                fprintf(stderr,
                                        "[pvk] seed V=%.2f tau*%.0e mode=%d conv=%d it=%d r=%.2e\n",
                                        V_seed, tscale[k], md, rinfo.converged,
                                        rinfo.iterations, rinfo.residual);
                            if (!ok) break;
                        }
                        for (int i = 0; i < n; i++) {
                            m.taun[i] = tn[i];
                            m.taup[i] = tp[i];
                        }
                    }
                    pvk__free(tn);
                    pvk__free(tp);
                    if (!ok) memcpy(u, ubak, sizeof(double) * 3 * n);
                }

                /* 3) 产生率 homotopy 兜底。暗态没有光可爬坡，改用吸收层里的
                 *    合成产生率 1e20 逐十倍压到 0。 */
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

            /* 接触层掩码刚收缩过：被释放的节点是从相邻自由节点重新起算的，
             * 初值离真解较远，这一个偏压点需要更多 Newton 步（实测 80 步
             * 不够、150 步左右收敛）。只对这类点放宽预算，正常点不受影响。 */
            pvk_solver_opts opts = ctx->solver_opts;
            if (m.cpl_released > 0 && opts.max_iter > 0) opts.max_iter *= 3;

            pvk_solve_info trial;
            memset(&trial, 0, sizeof(trial));
            trial.n = n;
            solve_newton(&m, u, &opts, &trial);

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
            /* 折半步长只有折到"目标真的变了"才有意义：
             *   0.01 V 的扫描从 1.35 走向 1.36 时，step 0.2..0.0125 都会
             *   被夹到同一个 1.36 V，重跑只是同一个 Newton（实测 rn 轨迹
             *   逐位相同）；折到 0.00625 才出现中间点 1.35625 V，是有用
             *   的一步。这里一次折到出现新目标为止，省掉中间的重复。 */
            double gap = fabs(V_app - V_cur);
            double next = step;
            int moved = 0;
            for (int t = 0; t < 12; t++) {
                next *= 0.5;
                if (fabs(next) < 0.005) break;
                double Vnext;
                if (gap < 1e-9) {
                    Vnext = V_app;
                } else {
                    Vnext = V_cur + next;
                    if ((V_app - Vnext) * next <= 0.0) Vnext = V_app;
                    if (fabs(Vnext - V_cur) < 1e-12) Vnext = V_app;
                }
                if (fabs(Vnext - Vk) >= 1e-12) { moved = 1; break; }
            }
            if (!moved) break;
            step = next;
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

    /* 延续失败：目标点改为冷启动重解一次。高偏压（>Voc）时热启动解可能
     * 落在另一条解支上，步长减半救不回来；而冷启动沿 V=0.15→… 的种子
     * 路径能解到同一个物理解（800 nm 卡 1.30 V：热启动 line search 失败
     * rn~1e12，冷启动 J=+22.8 mA/cm² 与旧版一致）。 */
    int from_retry = 0;
    if (!info.converged && ctx->have_solution && g_retry_depth == 0) {
        g_retry_depth++;
        ctx->have_solution = 0;
        pvk_solve_info r2;
        memset(&r2, 0, sizeof(r2));
        r2.n = n;
        if (pvk_solve_dc(ctx, dev, opt, bias, &r2) == PVK_OK && r2.converged) {
            info = r2;
            info.v_applied = V_app;
            from_retry = 1;
            done = 1;
        }
        if (getenv("PVK_DEBUG"))
            fprintf(stderr, "[pvk] cold retry V=%.3f -> conv=%d it=%d r=%.2e\n",
                    V_app, r2.converged, r2.iterations, r2.residual);
        g_retry_depth--;
    }

    double J = 0.0;
    if (info.converged && from_retry) {
        J = ctx->last_J;          /* 内层已写好 fields 与 last_J */
    } else if (info.converged) {
        fill_fields(ctx, &m, u, &J);
        /* 分支跳变防护：隐式接触层让方程有多条解支（例如吸收层被整层
         * 钉扎、电流几乎为零的"死"解）。继续法从正确解支出发时，若某一步
         * Newton 跑到另一条支上，终端电流会在 0.02 V 内反向并跳变几个
         * Jsc —— Cs2AgBiBr6 复刻卡 1.28→1.30 V 就从 -16 跳到 +22 mA/cm²
         * 并被宽松判据当成收敛。正常 J-V 在 Voc 附近也会反号，但 |ΔJ|
         * 只有几 mA，所以阈值保守地取 10 mA/cm²。但 Voc 附近是指数段，
         * 20 mV 内 |J| 就能涨好几倍（1.28→1.30 实测 -3.06→+22.8），
         * 所以只有 |J| 缩到前一点的 20% 以下（死解特征）才判为跳支。 */
        int branch_jump = had_prev && fabs(V_app - V_prev0) < 0.1 &&
                          isfinite(J_prev) && fabs(J_prev) > 1.0 &&
                          J * J_prev < 0.0 &&
                          fabs(J) < 0.2 * fabs(J_prev) &&
                          fabs(J - J_prev) > 10.0;   /* mA/cm²：只拦大幅跳变 */
        if (!isfinite(J) || fabs(J) > 1e5 || branch_jump) {
            /* 解出来的电流不物理（发散状态但残差判据被绕过），按失败处理 */
            if (branch_jump && getenv("PVK_DEBUG"))
                fprintf(stderr, "[pvk] branch jump at V=%.3f: J %.4g -> %.4g\n",
                        V_app, J_prev, J);
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
