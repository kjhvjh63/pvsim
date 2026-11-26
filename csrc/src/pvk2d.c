/*
 * pvk2d.c — 2D 稳态漂移-扩散原型（M4-4）。
 *
 * 与 perovskite/solvers/fv2d.py 同一离散：
 *   · 有限体积 + Scharfetter–Gummel（Slotboom 变量 Yn=n/ni、Zp=p/ni）；
 *   · 全耦合阻尼 Newton，解析 Jacobian，行均衡 + ILU(0)/BiCGSTAB；
 *   · 顶面局部电极（ψ 钉 + 多子钉 + 少子 Robin SRV）由掩码描述；
 *   · 输运层少子施加局部平衡（Yn·Zp=1）约束。
 *
 * 网格/逐节点参数/掩码由调用方传入，见 pvk2d.h。
 */
#include "pvk2d.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PVK2D_Q   1.602176634e-19
#define PVK2D_KB  1.380649e-23
#define PVK2D_BW  16            /* 每行预分配的最大非零元数 */

typedef struct {
    int    k0, k1;
    double h, area;
    double aE, Cn, Cp;
    double cn, an, bp, ap;
    double Bpd, Bmd, Bpq, Bmq;
} pvk2d_edge;

struct pvk2d_ctx {
    char err[512];

    int nx, ny, N, n3;
    double Vt, width;
    int iL, iR;

    double *x, *y, *dxw, *dyw, *vol;

    double *eps, *chi, *Eg, *Nc, *Nv, *ni, *u_off;
    double *mun, *mup, *taun, *taup, *brad, *Nd, *Na;
    double *psi_neu, *yn_pin, *zp_pin;
    double *p0s, *n0s;
    double top_srv;

    unsigned char *pin_psi, *pin_yn, *pin_zp, *pin_anod;
    unsigned char *slave_n, *slave_p, *surf, *surf_e, *gen;

    double *G;                 /* [N]，可 NULL（暗态） */
    double *psi, *Yn, *Zp;     /* 当前状态 */
    double *psi_bak, *Yn_bak, *Zp_bak;
    int have_state;
    int need_seed;             /* 状态刚由平衡态生成，光照初值尚未预填 */

    pvk2d_edge *ye, *xe;
    int nye, nxe;

    /* 稀疏 Jacobian（固定 CSR 结构） */
    int *rowptr, *col;
    double *val;
    int nnz;

    /* 线性求解工作区 */
    double *srow, *lu, *rhs, *du, *du2;
    int *diagpos;
    double *w1, *w2, *w3, *w4, *w5, *w6;
    double *rtmp, *rbuf;
};

/* ------------------------------------------------------------------ */
/* 工具                                                                */
/* ------------------------------------------------------------------ */

static void pvk2d_seterr(pvk2d_ctx* c, const char* msg) {
    if (!c) return;
    snprintf(c->err, sizeof(c->err), "%s", msg);
}

static void* pvk2d_alloc(size_t n) { return calloc(n ? n : 1, 1); }

static double clampd(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/* B(z) = z/(e^z-1)，与 1D 内核同款 expm1 分支 */
static double bern(double z) {
    if (z > 1e-8) {
        if (z > 700.0) return 0.0;
        return z / expm1(z);
    }
    if (z < -1e-8) {
        double w = -z;
        if (w > 700.0) return w;
        return w / (-expm1(-w));
    }
    return 1.0 - 0.5 * z + z * z / 12.0;
}

/* B'(z) = (e^z-1-z e^z)/(e^z-1)^2，与 fv2d._Bp 同分支 */
static double bernp(double z) {
    if (fabs(z) < 1e-4) return -0.5 + z / 6.0;
    if (z > 50.0) return (1.0 - z) * exp(-z);
    if (z < -50.0) return -1.0;
    {
        double ez = exp(z), em1 = expm1(z);
        return (em1 - z * ez) / (em1 * em1);
    }
}

/* ------------------------------------------------------------------ */
/* 网格与参数                                                          */
/* ------------------------------------------------------------------ */

static void pvk2d_free_arrays(pvk2d_ctx* c) {
    if (!c) return;
#define F(p) do { free(p); (p) = NULL; } while (0)
    F(c->x); F(c->y); F(c->dxw); F(c->dyw); F(c->vol);
    F(c->eps); F(c->chi); F(c->Eg); F(c->Nc); F(c->Nv); F(c->ni); F(c->u_off);
    F(c->mun); F(c->mup); F(c->taun); F(c->taup); F(c->brad); F(c->Nd); F(c->Na);
    F(c->psi_neu); F(c->yn_pin); F(c->zp_pin); F(c->p0s); F(c->n0s);
    F(c->pin_psi); F(c->pin_yn); F(c->pin_zp); F(c->pin_anod);
    F(c->slave_n); F(c->slave_p); F(c->surf); F(c->surf_e); F(c->gen);
    F(c->G); F(c->psi); F(c->Yn); F(c->Zp);
    F(c->psi_bak); F(c->Yn_bak); F(c->Zp_bak);
    F(c->ye); F(c->xe);
    F(c->rowptr); F(c->col); F(c->val);
    F(c->srow); F(c->lu); F(c->diagpos); F(c->rhs); F(c->du); F(c->du2);
    F(c->w1); F(c->w2); F(c->w3); F(c->w4); F(c->w5); F(c->w6);
    F(c->rtmp); F(c->rbuf);
#undef F
}

static int pvk2d_copy_d(pvk2d_ctx* c, double** dst, const double* src, int n) {
    if (!src) { *dst = NULL; return 0; }
    *dst = (double*)pvk2d_alloc(sizeof(double) * (size_t)n);
    if (!*dst) { pvk2d_seterr(c, "out of memory"); return -1; }
    memcpy(*dst, src, sizeof(double) * (size_t)n);
    return 0;
}

static int pvk2d_copy_b(pvk2d_ctx* c, unsigned char** dst,
                        const unsigned char* src, int n) {
    if (!src) { *dst = NULL; return 0; }
    *dst = (unsigned char*)pvk2d_alloc((size_t)n);
    if (!*dst) { pvk2d_seterr(c, "out of memory"); return -1; }
    memcpy(*dst, src, (size_t)n);
    return 0;
}

static int pvk2d_build_pattern(pvk2d_ctx* c) {
    int N = c->N, n3 = c->n3;
    int *cnt = (int*)pvk2d_alloc(sizeof(int) * (size_t)n3);
    int *tmp = (int*)pvk2d_alloc(sizeof(int) * (size_t)n3 * PVK2D_BW);
    int i, j, e, total = 0;
    if (!cnt || !tmp) { free(cnt); free(tmp); return -1; }

#define PAD(row, col) do {                                                   \
        int r_ = (row), c_ = (col);                                          \
        int* tc_ = tmp + (size_t)PVK2D_BW * r_;                              \
        int k_;                                                              \
        for (k_ = 0; k_ < cnt[r_]; ++k_) if (tc_[k_] == c_) break;           \
        if (k_ == cnt[r_] && cnt[r_] < PVK2D_BW) tc_[cnt[r_]++] = c_;        \
    } while (0)

    for (i = 0; i < N; ++i) {
        PAD(i, i);                 PAD(i, N + i);          PAD(i, 2 * N + i);
        PAD(N + i, N + i);         PAD(N + i, 2 * N + i);
        PAD(2 * N + i, 2 * N + i); PAD(2 * N + i, N + i);
    }
    for (e = 0; e < c->nye; ++e) {
        int k0 = c->ye[e].k0, k1 = c->ye[e].k1;
        PAD(k0, k0); PAD(k0, k1); PAD(k1, k0); PAD(k1, k1);
        PAD(N + k0, N + k0); PAD(N + k0, N + k1);
        PAD(N + k1, N + k0); PAD(N + k1, N + k1);
        PAD(N + k0, k0); PAD(N + k0, k1); PAD(N + k1, k0); PAD(N + k1, k1);
        PAD(2 * N + k0, 2 * N + k0); PAD(2 * N + k0, 2 * N + k1);
        PAD(2 * N + k1, 2 * N + k0); PAD(2 * N + k1, 2 * N + k1);
        PAD(2 * N + k0, k0); PAD(2 * N + k0, k1); PAD(2 * N + k1, k0); PAD(2 * N + k1, k1);
    }
    for (e = 0; e < c->nxe; ++e) {
        int k0 = c->xe[e].k0, k1 = c->xe[e].k1;
        PAD(k0, k0); PAD(k0, k1); PAD(k1, k0); PAD(k1, k1);
        PAD(N + k0, N + k0); PAD(N + k0, N + k1);
        PAD(N + k1, N + k0); PAD(N + k1, N + k1);
        PAD(N + k0, k0); PAD(N + k0, k1); PAD(N + k1, k0); PAD(N + k1, k1);
        PAD(2 * N + k0, 2 * N + k0); PAD(2 * N + k0, 2 * N + k1);
        PAD(2 * N + k1, 2 * N + k0); PAD(2 * N + k1, 2 * N + k1);
        PAD(2 * N + k0, k0); PAD(2 * N + k0, k1); PAD(2 * N + k1, k0); PAD(2 * N + k1, k1);
    }
#undef PAD

    c->rowptr = (int*)pvk2d_alloc(sizeof(int) * (size_t)(n3 + 1));
    if (!c->rowptr) { free(cnt); free(tmp); return -1; }
    for (i = 0; i < n3; ++i) { c->rowptr[i] = total; total += cnt[i]; }
    c->rowptr[n3] = total;
    c->nnz = total;
    c->col = (int*)pvk2d_alloc(sizeof(int) * (size_t)total);
    c->val = (double*)pvk2d_alloc(sizeof(double) * (size_t)total);
    if (!c->col || !c->val) { free(cnt); free(tmp); return -1; }
    for (i = 0; i < n3; ++i) {
        int* tc = tmp + (size_t)PVK2D_BW * i;
        int base = c->rowptr[i], m = cnt[i];
        for (j = 1; j < m; ++j) {              /* 插入排序，保证列升序 */
            int key = tc[j], q = j - 1;
            while (q >= 0 && tc[q] > key) { tc[q + 1] = tc[q]; --q; }
            tc[q + 1] = key;
        }
        for (j = 0; j < m; ++j) c->col[base + j] = tc[j];
    }
    free(cnt);
    free(tmp);
    return 0;
}

int pvk2d_set_mesh(pvk2d_ctx* c, const pvk2d_mesh* m) {
    int nx, ny, N, i, j, e;
    if (!c || !m) return PVK_EINVAL;
    if (m->nx < 2 || m->ny < 2) { pvk2d_seterr(c, "nx/ny must be >= 2"); return PVK_EINVAL; }
    pvk2d_free_arrays(c);

    nx = m->nx; ny = m->ny; N = nx * ny;
    c->nx = nx; c->ny = ny; c->N = N; c->n3 = 3 * N;
    c->Vt = PVK2D_KB * (m->T_K > 0 ? m->T_K : 300.0) / PVK2D_Q;
    c->width = m->width_cm > 0 ? m->width_cm : 1.0;
    c->iL = m->iL; c->iR = m->iR;
    c->top_srv = m->top_srv;

    if (pvk2d_copy_d(c, &c->x, m->x_cm, nx) ||
        pvk2d_copy_d(c, &c->y, m->y_cm, ny) ||
        pvk2d_copy_d(c, &c->dxw, m->dxw, nx) ||
        pvk2d_copy_d(c, &c->eps, m->eps, N) ||
        pvk2d_copy_d(c, &c->chi, m->chi, N) ||
        pvk2d_copy_d(c, &c->Eg, m->Eg, N) ||
        pvk2d_copy_d(c, &c->Nc, m->Nc, N) ||
        pvk2d_copy_d(c, &c->Nv, m->Nv, N) ||
        pvk2d_copy_d(c, &c->ni, m->ni, N) ||
        pvk2d_copy_d(c, &c->u_off, m->u_off, N) ||
        pvk2d_copy_d(c, &c->mun, m->mun, N) ||
        pvk2d_copy_d(c, &c->mup, m->mup, N) ||
        pvk2d_copy_d(c, &c->taun, m->taun, N) ||
        pvk2d_copy_d(c, &c->taup, m->taup, N) ||
        pvk2d_copy_d(c, &c->brad, m->brad, N) ||
        pvk2d_copy_d(c, &c->Nd, m->Nd, N) ||
        pvk2d_copy_d(c, &c->Na, m->Na, N) ||
        pvk2d_copy_d(c, &c->vol, m->vol, N) ||
        pvk2d_copy_d(c, &c->psi_neu, m->psi_neu, N) ||
        pvk2d_copy_d(c, &c->yn_pin, m->yn_pin, N) ||
        pvk2d_copy_d(c, &c->zp_pin, m->zp_pin, N) ||
        pvk2d_copy_d(c, &c->p0s, m->p0s, N) ||
        pvk2d_copy_d(c, &c->n0s, m->n0s, N) ||
        pvk2d_copy_b(c, &c->pin_psi, m->pin_psi, N) ||
        pvk2d_copy_b(c, &c->pin_yn, m->pin_yn, N) ||
        pvk2d_copy_b(c, &c->pin_zp, m->pin_zp, N) ||
        pvk2d_copy_b(c, &c->pin_anod, m->pin_anod, N) ||
        pvk2d_copy_b(c, &c->slave_n, m->slave_n, N) ||
        pvk2d_copy_b(c, &c->slave_p, m->slave_p, N) ||
        pvk2d_copy_b(c, &c->surf, m->surf, N) ||
        pvk2d_copy_b(c, &c->surf_e, m->surf_e, N) ||
        pvk2d_copy_b(c, &c->gen, m->gen, N))
        return PVK_EMEM;

    /* 深度控制体高度 */
    c->dyw = (double*)pvk2d_alloc(sizeof(double) * (size_t)ny);
    if (!c->dyw) return PVK_EMEM;
    c->dyw[0] = 0.5 * (c->y[1] - c->y[0]);
    c->dyw[ny - 1] = 0.5 * (c->y[ny - 1] - c->y[ny - 2]);
    for (j = 1; j < ny - 1; ++j)
        c->dyw[j] = 0.5 * (c->y[j + 1] - c->y[j - 1]);

    /* 边表 */
    c->nye = (ny - 1) * nx;
    c->nxe = ny * (nx - 1);
    c->ye = (pvk2d_edge*)pvk2d_alloc(sizeof(pvk2d_edge) * (size_t)(c->nye ? c->nye : 1));
    c->xe = (pvk2d_edge*)pvk2d_alloc(sizeof(pvk2d_edge) * (size_t)(c->nxe ? c->nxe : 1));
    if (!c->ye || !c->xe) return PVK_EMEM;
    e = 0;
    for (j = 0; j < ny - 1; ++j) {
        double h = c->y[j + 1] - c->y[j];
        for (i = 0; i < nx; ++i) {
            c->ye[e].k0 = j * nx + i;
            c->ye[e].k1 = (j + 1) * nx + i;
            c->ye[e].h = h;
            c->ye[e].area = c->dxw[i];
            ++e;
        }
    }
    e = 0;
    for (j = 0; j < ny; ++j) {
        for (i = 0; i < nx - 1; ++i) {
            c->xe[e].k0 = j * nx + i;
            c->xe[e].k1 = j * nx + i + 1;
            c->xe[e].h = c->x[i + 1] - c->x[i];
            c->xe[e].area = c->dyw[j];
            ++e;
        }
    }

    /* 状态与工作区 */
    c->psi = (double*)pvk2d_alloc(sizeof(double) * (size_t)N);
    c->Yn = (double*)pvk2d_alloc(sizeof(double) * (size_t)N);
    c->Zp = (double*)pvk2d_alloc(sizeof(double) * (size_t)N);
    c->psi_bak = (double*)pvk2d_alloc(sizeof(double) * (size_t)N);
    c->Yn_bak = (double*)pvk2d_alloc(sizeof(double) * (size_t)N);
    c->Zp_bak = (double*)pvk2d_alloc(sizeof(double) * (size_t)N);
    c->G = (double*)pvk2d_alloc(sizeof(double) * (size_t)N);
    c->srow = (double*)pvk2d_alloc(sizeof(double) * (size_t)(3 * N));
    c->rhs = (double*)pvk2d_alloc(sizeof(double) * (size_t)(3 * N));
    c->du = (double*)pvk2d_alloc(sizeof(double) * (size_t)(3 * N));
    c->du2 = (double*)pvk2d_alloc(sizeof(double) * (size_t)(3 * N));
    c->rtmp = (double*)pvk2d_alloc(sizeof(double) * (size_t)(3 * N));
    c->rbuf = (double*)pvk2d_alloc(sizeof(double) * (size_t)(3 * N));
    c->w1 = (double*)pvk2d_alloc(sizeof(double) * (size_t)(3 * N));
    c->w2 = (double*)pvk2d_alloc(sizeof(double) * (size_t)(3 * N));
    c->w3 = (double*)pvk2d_alloc(sizeof(double) * (size_t)(3 * N));
    c->w4 = (double*)pvk2d_alloc(sizeof(double) * (size_t)(3 * N));
    c->w5 = (double*)pvk2d_alloc(sizeof(double) * (size_t)(3 * N));
    c->w6 = (double*)pvk2d_alloc(sizeof(double) * (size_t)(3 * N));
    if (!c->psi || !c->Yn || !c->Zp || !c->psi_bak || !c->Yn_bak || !c->Zp_bak ||
        !c->G || !c->srow || !c->rhs || !c->du || !c->du2 || !c->rtmp || !c->rbuf ||
        !c->w1 || !c->w2 || !c->w3 || !c->w4 || !c->w5 || !c->w6)
        return PVK_EMEM;

    if (pvk2d_build_pattern(c) != 0) return PVK_EMEM;

    c->lu = (double*)pvk2d_alloc(sizeof(double) * (size_t)c->nnz);
    c->diagpos = (int*)pvk2d_alloc(sizeof(int) * (size_t)(3 * N));
    if (!c->lu || !c->diagpos) return PVK_EMEM;
    {
        int r, p;
        for (r = 0; r < 3 * N; ++r) {
            int found = -1;
            for (p = c->rowptr[r]; p < c->rowptr[r + 1]; ++p)
                if (c->col[p] == r) { found = p; break; }
            if (found < 0) { pvk2d_seterr(c, "pattern missing diagonal"); return PVK_EAPI; }
            c->diagpos[r] = found;
        }
    }

    c->have_state = 0;
    c->need_seed = 0;
    return PVK_OK;
}

int pvk2d_set_generation(pvk2d_ctx* c, const double* G) {
    if (!c || !c->G) return PVK_EINVAL;
    if (!G) { memset(c->G, 0, sizeof(double) * (size_t)c->N); return PVK_OK; }
    memcpy(c->G, G, sizeof(double) * (size_t)c->N);
    return PVK_OK;
}

/* ------------------------------------------------------------------ */
/* 边系数                                                              */
/* ------------------------------------------------------------------ */

static void pvk2d_edge_coeff(pvk2d_ctx* c, pvk2d_edge* e, const double* psi) {
    int k0 = e->k0, k1 = e->k1;
    double Vt = c->Vt;
    double Wn0 = psi[k0] + c->chi[k0] + Vt * log(c->Nc[k0]);
    double Wn1 = psi[k1] + c->chi[k1] + Vt * log(c->Nc[k1]);
    double Wp0 = psi[k0] + c->chi[k0] + c->Eg[k0] - Vt * log(c->Nv[k0]);
    double Wp1 = psi[k1] + c->chi[k1] + c->Eg[k1] - Vt * log(c->Nv[k1]);
    double dn = (Wn1 - Wn0) / Vt;
    double dp = (Wp1 - Wp0) / Vt;
    double eps_e = 0.5 * (c->eps[k0] + c->eps[k1]);
    double mun_e = 0.5 * (c->mun[k0] + c->mun[k1]);
    double mup_e = 0.5 * (c->mup[k0] + c->mup[k1]);
    e->aE = eps_e * e->area / e->h;
    e->Cn = PVK2D_Q * mun_e * Vt / e->h * e->area;
    e->Cp = PVK2D_Q * mup_e * Vt / e->h * e->area;
    e->cn = e->Cn * bern(dn) * c->ni[k1];
    e->an = e->Cn * bern(-dn) * c->ni[k0];
    e->bp = e->Cp * bern(dp) * c->ni[k0];
    e->ap = e->Cp * bern(-dp) * c->ni[k1];
    e->Bpd = bernp(dn);
    e->Bmd = bernp(-dn);
    e->Bpq = bernp(dp);
    e->Bmq = bernp(-dp);
}

static void pvk2d_compute_edges(pvk2d_ctx* c, const double* psi) {
    int e;
    for (e = 0; e < c->nye; ++e) pvk2d_edge_coeff(c, &c->ye[e], psi);
    for (e = 0; e < c->nxe; ++e) pvk2d_edge_coeff(c, &c->xe[e], psi);
}

/* ------------------------------------------------------------------ */
/* Jacobian 累加                                                       */
/* ------------------------------------------------------------------ */

static void jadd(pvk2d_ctx* c, int row, int col, double v) {
    int p;
    for (p = c->rowptr[row]; p < c->rowptr[row + 1]; ++p) {
        if (c->col[p] == col) { c->val[p] += v; return; }
    }
    if (getenv("PVK2D_DEBUG"))
        fprintf(stderr, "[pvk2d] jadd miss row=%d col=%d\n", row, col);
}

/* 行是否是约束行（钉扎/局部平衡），正常物理项不再进入 */
static int row_is_cons(const pvk2d_ctx* c, int blk, int i) {
    if (blk == 0) return c->pin_psi && c->pin_psi[i];
    if (blk == 1) return (c->pin_yn && c->pin_yn[i]) || (c->slave_n && c->slave_n[i]);
    return (c->pin_zp && c->pin_zp[i]) || (c->slave_p && c->slave_p[i]);
}

/* 组装残差（R）与 Jacobian（with_jac 时）。列已按 S=Yn/Zp 缩放。 */
static void pvk2d_assemble(pvk2d_ctx* c, double V, const double* G,
                           const double* psi, const double* Yn,
                           const double* Zp, double* R, int with_jac) {
    int N = c->N, nx = c->nx, i, e;
    double Vt = c->Vt, q = PVK2D_Q;

    if (with_jac) memset(c->val, 0, sizeof(double) * (size_t)c->nnz);
    memset(R, 0, sizeof(double) * (size_t)(3 * N));
    pvk2d_compute_edges(c, psi);

    for (i = 0; i < N; ++i) {
        double ni = c->ni[i];
        double Y = Yn[i], Z = Zp[i];
        double n = ni * Y, p = ni * Z;
        double D = c->taup[i] * (n + ni) + c->taun[i] * (p + ni);
        double dnp = n * p - ni * ni;
        double Rr, sn, sp;
        double g = G ? G[i] : 0.0;
        if (D < 1e-300) D = 1e-300;
        Rr = dnp / D + c->brad[i] * dnp;
        sn = (p * D - dnp * c->taup[i]) / (D * D) + c->brad[i] * p;
        sp = (n * D - dnp * c->taun[i]) / (D * D) + c->brad[i] * n;

        if (!row_is_cons(c, 0, i)) {
            R[i] = -q * (ni * Z - ni * Y + (c->Nd[i] - c->Na[i])) * c->vol[i];
            if (with_jac) {
                jadd(c, i, N + i, q * ni * c->vol[i]);
                jadd(c, i, 2 * N + i, -q * ni * c->vol[i]);
            }
        }
        if (!row_is_cons(c, 1, i)) {
            R[N + i] = -q * (Rr - g) * c->vol[i];
            if (with_jac) {
                jadd(c, N + i, 2 * N + i, -q * c->vol[i] * sp * ni);
                jadd(c, N + i, N + i, -q * c->vol[i] * sn * ni);
            }
        }
        if (!row_is_cons(c, 2, i)) {
            R[2 * N + i] = -q * (g - Rr) * c->vol[i];
            if (with_jac) {
                jadd(c, 2 * N + i, N + i, q * c->vol[i] * sn * ni);
                jadd(c, 2 * N + i, 2 * N + i, q * c->vol[i] * sp * ni);
            }
        }
    }

    for (e = 0; e < c->nye + c->nxe; ++e) {
        pvk2d_edge* ed = (e < c->nye) ? &c->ye[e] : &c->xe[e - c->nye];
        int k0 = ed->k0, k1 = ed->k1;
        double dpsi = psi[k0] - psi[k1];
        double Jn = ed->cn * Yn[k1] - ed->an * Yn[k0];
        double Jp = ed->bp * Zp[k0] - ed->ap * Zp[k1];
        int c0 = row_is_cons(c, 0, k0), c1 = row_is_cons(c, 0, k1);
        int n0 = row_is_cons(c, 1, k0), n1 = row_is_cons(c, 1, k1);
        int p0 = row_is_cons(c, 2, k0), p1 = row_is_cons(c, 2, k1);

        if (!c0) R[k0] += ed->aE * dpsi;
        if (!c1) R[k1] -= ed->aE * dpsi;
        if (!n0) R[N + k0] += Jn;
        if (!n1) R[N + k1] -= Jn;
        if (!p0) R[2 * N + k0] += Jp;
        if (!p1) R[2 * N + k1] -= Jp;

        if (with_jac) {
            if (!c0) { jadd(c, k0, k1, -ed->aE); jadd(c, k0, k0, ed->aE); }
            if (!c1) { jadd(c, k1, k0, -ed->aE); jadd(c, k1, k1, ed->aE); }
            if (!n0) { jadd(c, N + k0, N + k1, ed->cn); jadd(c, N + k0, N + k0, -ed->an); }
            if (!n1) { jadd(c, N + k1, N + k0, ed->an); jadd(c, N + k1, N + k1, -ed->cn); }
            if (!p0) { jadd(c, 2 * N + k0, 2 * N + k1, -ed->ap); jadd(c, 2 * N + k0, 2 * N + k0, ed->bp); }
            if (!p1) { jadd(c, 2 * N + k1, 2 * N + k0, -ed->bp); jadd(c, 2 * N + k1, 2 * N + k1, ed->ap); }

            {
                double ny1 = c->ni[k1] * Yn[k1], ny0 = c->ni[k0] * Yn[k0];
                double pz0 = c->ni[k0] * Zp[k0], pz1 = c->ni[k1] * Zp[k1];
                double Tn = ed->Cn / Vt * (ny1 * ed->Bpd + ny0 * ed->Bmd);
                double Tp = ed->Cp / Vt * (pz0 * ed->Bpq + pz1 * ed->Bmq);
                if (!n0) { jadd(c, N + k0, k0, -Tn); jadd(c, N + k0, k1, Tn); }
                if (!n1) { jadd(c, N + k1, k0, Tn); jadd(c, N + k1, k1, -Tn); }
                if (!p0) { jadd(c, 2 * N + k0, k0, -Tp); jadd(c, 2 * N + k0, k1, Tp); }
                if (!p1) { jadd(c, 2 * N + k1, k0, Tp); jadd(c, 2 * N + k1, k1, -Tp); }
            }
        }
    }

    /* 顶面有限表面复合（Robin） */
    if (c->surf && c->p0s && c->top_srv > 0) {
        for (i = 0; i < N; ++i) {
            if (!c->surf[i]) continue;
            R[2 * N + i] += q * c->top_srv * (c->ni[i] * Zp[i] - c->p0s[i])
                            * c->dxw[i % nx];
            if (with_jac)
                jadd(c, 2 * N + i, 2 * N + i,
                     q * c->top_srv * c->ni[i] * c->dxw[i % nx]);
        }
    }
    if (c->surf_e && c->n0s && c->top_srv > 0) {
        for (i = 0; i < N; ++i) {
            if (!c->surf_e[i]) continue;
            R[N + i] -= q * c->top_srv * (c->ni[i] * Yn[i] - c->n0s[i])
                        * c->dxw[i % nx];
            if (with_jac)
                jadd(c, N + i, N + i,
                     -q * c->top_srv * c->ni[i] * c->dxw[i % nx]);
        }
    }

    /* 约束行 */
    for (i = 0; i < N; ++i) {
        if (c->pin_psi && c->pin_psi[i]) {
            double pin = c->psi_neu[i] + ((c->pin_anod && c->pin_anod[i]) ? V : 0.0);
            R[i] = psi[i] - pin;
            if (with_jac) jadd(c, i, i, 1.0);
        }
        if (c->pin_yn && c->pin_yn[i]) {
            R[N + i] = Yn[i] - c->yn_pin[i];
            if (with_jac) jadd(c, N + i, N + i, 1.0);   /* 列缩放后再乘 S=Yn */
        }
        if (c->pin_zp && c->pin_zp[i]) {
            R[2 * N + i] = Zp[i] - c->zp_pin[i];
            if (with_jac) jadd(c, 2 * N + i, 2 * N + i, 1.0);
        }
        if (c->slave_n && c->slave_n[i]) {
            R[N + i] = Yn[i] * Zp[i] - 1.0;
            if (with_jac) {
                jadd(c, N + i, N + i, fmax(Zp[i], 1e-300));
                jadd(c, N + i, 2 * N + i, fmax(Yn[i], 1e-300));
            }
        }
        if (c->slave_p && c->slave_p[i]) {
            R[2 * N + i] = Yn[i] * Zp[i] - 1.0;
            if (with_jac) {
                jadd(c, 2 * N + i, 2 * N + i, fmax(Yn[i], 1e-300));
                jadd(c, 2 * N + i, N + i, fmax(Zp[i], 1e-300));
            }
        }
    }
}

/* 列缩放：d 乘以 S[col]（ψ:1，Yn:Yn，Zp:Zp） */
static void pvk2d_scale_cols(pvk2d_ctx* c, const double* Yn, const double* Zp) {
    int N = c->N, r, p;
    for (r = 0; r < 3 * N; ++r) {
        for (p = c->rowptr[r]; p < c->rowptr[r + 1]; ++p) {
            int col = c->col[p];
            double s = 1.0;
            if (col >= N && col < 2 * N) s = fmax(Yn[col - N], 1e-300);
            else if (col >= 2 * N) s = fmax(Zp[col - 2 * N], 1e-300);
            c->val[p] *= s;
        }
    }
}

/* ------------------------------------------------------------------ */
/* ILU(0) + BiCGSTAB                                                   */
/* ------------------------------------------------------------------ */

static int pvk2d_ilu0(pvk2d_ctx* c) {
    int n = c->n3, i, p, q;
    int* diag = (int*)c->diagpos;
    memcpy(c->lu, c->val, sizeof(double) * (size_t)c->nnz);
    for (i = 0; i < n; ++i) {
        for (p = c->rowptr[i]; p < c->rowptr[i + 1]; ++p) {
            int k = c->col[p];
            if (k >= i) break;
            {
                int dk = diag[k];
                double piv = c->lu[dk];
                double m;
                if (fabs(piv) < 1e-300) {
                    piv = (piv >= 0.0 ? 1e-300 : -1e-300);
                    c->lu[dk] = piv;
                }
                m = c->lu[p] / piv;
                c->lu[p] = m;
                for (q = c->rowptr[k]; q < c->rowptr[k + 1]; ++q) {
                    int j = c->col[q];
                    if (j <= k) continue;
                    {
                        int t;
                        for (t = c->rowptr[i]; t < c->rowptr[i + 1]; ++t) {
                            if (c->col[t] == j) { c->lu[t] -= m * c->lu[q]; break; }
                        }
                    }
                }
            }
        }
        if (fabs(c->lu[diag[i]]) < 1e-300)
            c->lu[diag[i]] = (c->lu[diag[i]] >= 0.0 ? 1e-300 : -1e-300);
    }
    return 0;
}

static void pvk2d_ilu_solve(pvk2d_ctx* c, const double* b, double* x) {
    int n = c->n3, i, p;
    int* diag = (int*)c->diagpos;
    memcpy(x, b, sizeof(double) * (size_t)n);
    for (i = 0; i < n; ++i) {
        double s = x[i];
        for (p = c->rowptr[i]; p < c->rowptr[i + 1]; ++p) {
            int j = c->col[p];
            if (j >= i) break;
            s -= c->lu[p] * x[j];
        }
        x[i] = s;
    }
    for (i = n - 1; i >= 0; --i) {
        double s = x[i];
        for (p = diag[i] + 1; p < c->rowptr[i + 1]; ++p)
            s -= c->lu[p] * x[c->col[p]];
        x[i] = s / c->lu[diag[i]];
    }
}

static double dot(const double* a, const double* b, int n) {
    double s = 0.0; int i;
    for (i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

static double norm2(const double* a, int n) { return sqrt(dot(a, a, n)); }

static void matvec(pvk2d_ctx* c, const double* x, double* y) {
    int r, p;
    for (r = 0; r < c->n3; ++r) {
        double s = 0.0;
        for (p = c->rowptr[r]; p < c->rowptr[r + 1]; ++p)
            s += c->val[p] * x[c->col[p]];
        y[r] = s;
    }
}

static int pvk2d_bicgstab(pvk2d_ctx* c, const double* b, double* x, int maxit,
                          double tol, int resume) {
    int n = c->n3, it, i;
    double bn = norm2(b, n);
    double *r = c->w1, *r0 = c->w2, *p = c->w3, *v = c->w4,
           *s = c->w5, *t = c->w6, *ph = c->rtmp, *sh = c->rbuf;
    double rho = 1.0, alpha = 1.0, omega = 1.0;
    if (bn == 0.0) { memset(x, 0, sizeof(double) * (size_t)n); return 0; }
    if (resume) {
        /* 重启：保留当前 x，用它的真实残差作新的影子残差（停滞常见解法） */
        matvec(c, x, r);
        for (i = 0; i < n; ++i) r[i] = b[i] - r[i];
    } else {
        memset(x, 0, sizeof(double) * (size_t)n);
        memcpy(r, b, sizeof(double) * (size_t)n);
    }
    memcpy(r0, r, sizeof(double) * (size_t)n);
    memset(p, 0, sizeof(double) * (size_t)n);
    memset(v, 0, sizeof(double) * (size_t)n);
    for (it = 0; it < maxit; ++it) {
        double rho1 = dot(r0, r, n);
        double beta, rn, den;
        if (!isfinite(rho1) || fabs(rho) < 1e-300) break;
        if (!isfinite(alpha) || fabs(omega) < 1e-300) break;
        beta = (rho1 / rho) * (alpha / omega);
        rho = rho1;
        for (i = 0; i < n; ++i) p[i] = r[i] + beta * (p[i] - omega * v[i]);
        pvk2d_ilu_solve(c, p, ph);
        matvec(c, ph, v);
        den = dot(r0, v, n);
        if (!isfinite(den) || fabs(den) < 1e-300) break;
        alpha = rho / den;
        for (i = 0; i < n; ++i) s[i] = r[i] - alpha * v[i];
        rn = norm2(s, n);
        if (rn / bn < tol) {
            for (i = 0; i < n; ++i) x[i] += alpha * ph[i];
            return it + 1;
        }
        pvk2d_ilu_solve(c, s, sh);
        matvec(c, sh, t);
        omega = dot(t, s, n) / fmax(dot(t, t, n), 1e-300);
        if (!isfinite(omega) || fabs(omega) < 1e-300) break;
        for (i = 0; i < n; ++i) x[i] += alpha * ph[i] + omega * sh[i];
        for (i = 0; i < n; ++i) r[i] = s[i] - omega * t[i];
        if (norm2(r, n) / bn < tol) return it + 1;
    }
    if (getenv("PVK2D_DEBUG"))
        fprintf(stderr, "[pvk2d] bicgstab exhausted: rel=%.2e\n", norm2(r, n) / bn);
    return -1;
}

/* 对角预条件：就地除以 ILU 的 U 对角 */
static void pvk2d_jacobi_solve(pvk2d_ctx* c, double* x) {
    int i, n = c->n3;
    int* diag = (int*)c->diagpos;
    for (i = 0; i < n; ++i) {
        double d = c->lu[diag[i]];
        if (fabs(d) > 1e-300) x[i] /= d;
    }
}

/* GMRES(m) + 右预条件：用于高偏压下病态 Jacobian 上 ILU0-BiCGSTAB 停滞
 * （实测 FG010 在 1.13 V 起 800 步只到 8.6e-4）。use_ilu=1 用 ILU(0)，
 * 否则用对角预条件；两者都失败则由上层退回预条件方向作近似步。 */
static int pvk2d_gmres(pvk2d_ctx* c, const double* b, double* x,
                       int restart, int maxit, double tol, int use_ilu) {
    int n = c->n3, i, j, k, it = 0, m, ret = -1;
    size_t vn = (size_t)(restart + 1) * (size_t)n;
    double bn = norm2(b, n);
    double *V = NULL, *Z = NULL, *H = NULL, *cs = NULL, *sn = NULL;
    double *g = NULL, *y = NULL, *w = NULL, *r = NULL, *z = NULL;
    if (bn == 0.0) {
        memset(x, 0, sizeof(double) * (size_t)n);
        return 0;
    }
    V = (double*)pvk2d_alloc(sizeof(double) * vn);
    Z = (double*)pvk2d_alloc(sizeof(double) * (size_t)restart * (size_t)n);
    H = (double*)pvk2d_alloc(sizeof(double) * (size_t)(restart + 1)
                             * (size_t)restart);
    cs = (double*)pvk2d_alloc(sizeof(double) * (size_t)restart);
    sn = (double*)pvk2d_alloc(sizeof(double) * (size_t)restart);
    g = (double*)pvk2d_alloc(sizeof(double) * (size_t)(restart + 1));
    y = (double*)pvk2d_alloc(sizeof(double) * (size_t)restart);
    w = (double*)pvk2d_alloc(sizeof(double) * (size_t)n);
    r = (double*)pvk2d_alloc(sizeof(double) * (size_t)n);
    z = (double*)pvk2d_alloc(sizeof(double) * (size_t)n);
    if (!V || !Z || !H || !cs || !sn || !g || !y || !w || !r || !z) goto done;
    memset(x, 0, sizeof(double) * (size_t)n);
    memcpy(r, b, sizeof(double) * (size_t)n);
    while (it < maxit) {
        double rn = norm2(r, n);
        if (!isfinite(rn)) break;
        if (rn / bn <= tol) { ret = 0; break; }
        for (i = 0; i < n; ++i) V[i] = r[i] / rn;
        memset(g, 0, sizeof(double) * (size_t)(restart + 1));
        memset(H, 0, sizeof(double) * (size_t)(restart + 1) * (size_t)restart);
        g[0] = rn;
        m = 0;
        for (j = 0; j < restart && it < maxit; ++j, ++it) {
            double hnext;
            if (use_ilu) {
                pvk2d_ilu_solve(c, V + (size_t)j * n, z);
            } else {
                memcpy(z, V + (size_t)j * n, sizeof(double) * (size_t)n);
                pvk2d_jacobi_solve(c, z);
            }
            matvec(c, z, w);
            memcpy(Z + (size_t)j * n, z, sizeof(double) * (size_t)n);
            for (i = 0; i <= j; ++i) {
                double h = dot(w, V + (size_t)i * n, n);
                H[(size_t)i * restart + j] = h;
                for (k = 0; k < n; ++k) w[k] -= h * V[(size_t)i * n + k];
            }
            hnext = norm2(w, n);
            H[(size_t)(j + 1) * restart + j] = hnext;
            /* 之前各步的 Givens 旋转消元 */
            for (i = 0; i < j; ++i) {
                double a = H[(size_t)i * restart + j];
                double bb = H[(size_t)(i + 1) * restart + j];
                H[(size_t)i * restart + j] = cs[i] * a + sn[i] * bb;
                H[(size_t)(i + 1) * restart + j] = -sn[i] * a + cs[i] * bb;
            }
            {
                double a = H[(size_t)j * restart + j];
                double rr = hypot(a, hnext);
                cs[j] = (rr > 1e-300) ? a / rr : 1.0;
                sn[j] = (rr > 1e-300) ? hnext / rr : 0.0;
                H[(size_t)j * restart + j] = rr;
                H[(size_t)(j + 1) * restart + j] = 0.0;
                g[j + 1] = -sn[j] * g[j];
                g[j] = cs[j] * g[j];
            }
            m = j + 1;
            if (fabs(g[j + 1]) / bn <= tol) break;
            if (!(hnext > 1e-300)) break;
            for (k = 0; k < n; ++k)
                V[(size_t)(j + 1) * n + k] = w[k] / hnext;
        }
        /* 上三角回代 H(0:m,0:m) y = g */
        for (i = m - 1; i >= 0; --i) {
            double s = g[i];
            for (k = i + 1; k < m; ++k)
                s -= H[(size_t)i * restart + k] * y[k];
            y[i] = (fabs(H[(size_t)i * restart + i]) > 1e-300)
                       ? s / H[(size_t)i * restart + i] : 0.0;
        }
        for (i = 0; i < m; ++i) {
            double yi = y[i];
            for (k = 0; k < n; ++k) x[k] += yi * Z[(size_t)i * n + k];
        }
        if (m == 0) break;
        /* 重启前重算真实残差 */
        matvec(c, x, r);
        for (i = 0; i < n; ++i) r[i] = b[i] - r[i];
    }
    if (ret < 0) {
        double rn = norm2(r, n);
        if (isfinite(rn) && rn / bn <= tol) ret = 0;
    }
done:
    free(V); free(Z); free(H); free(cs); free(sn);
    free(g); free(y); free(w); free(r); free(z);
    return ret;
}

/* 线性方程组 A x = rhs 的真实相对残差（行缩放后） */
static double pvk2d_lin_rel(pvk2d_ctx* c, const double* x) {
    int i;
    double num = 0.0, den = 0.0;
    matvec(c, x, c->rbuf);
    for (i = 0; i < c->n3; ++i) {
        double d = c->rhs[i] - c->rbuf[i];
        num += d * d;
        den += c->rhs[i] * c->rhs[i];
    }
    return sqrt(num / fmax(den, 1e-300));
}

/* 稠密 LU（部分主元）：小规模 2D 系统的最后兜底。
 * 近 Voc 时 Jacobian 极端病态，ILU 迭代法可能完全停滞后仍能靠直接解通过；
 * 仅在 n3 ≤ PVK2D_DENSE_MAX 时启用，避免大网格 O(n³) 爆炸。 */
#define PVK2D_DENSE_MAX 1800
static int pvk2d_dense_lu(pvk2d_ctx* c, const double* b, double* x) {
    int n = c->n3, i, j, k, piv;
    double* A = NULL;
    if (n > PVK2D_DENSE_MAX) return -1;
    A = (double*)pvk2d_alloc(sizeof(double) * (size_t)n * (size_t)n);
    if (!A) return -1;
    for (i = 0; i < n; ++i) {
        int p;
        for (p = c->rowptr[i]; p < c->rowptr[i + 1]; ++p)
            A[(size_t)i * n + c->col[p]] = c->val[p];
    }
    memcpy(x, b, sizeof(double) * (size_t)n);
    for (k = 0; k < n; ++k) {
        double mx = fabs(A[(size_t)k * n + k]);
        double d;
        piv = k;
        for (i = k + 1; i < n; ++i) {
            double v = fabs(A[(size_t)i * n + k]);
            if (v > mx) { mx = v; piv = i; }
        }
        if (!(mx > 1e-300)) { free(A); return -1; }
        if (piv != k) {
            for (j = 0; j < n; ++j) {
                double t = A[(size_t)k * n + j];
                A[(size_t)k * n + j] = A[(size_t)piv * n + j];
                A[(size_t)piv * n + j] = t;
            }
            { double t = x[k]; x[k] = x[piv]; x[piv] = t; }
        }
        d = A[(size_t)k * n + k];
        for (i = k + 1; i < n; ++i) {
            double m = A[(size_t)i * n + k] / d;
            if (m == 0.0) continue;
            A[(size_t)i * n + k] = 0.0;
            for (j = k + 1; j < n; ++j)
                A[(size_t)i * n + j] -= m * A[(size_t)k * n + j];
            x[i] -= m * x[k];
        }
    }
    for (i = n - 1; i >= 0; --i) {
        double s = x[i];
        for (j = i + 1; j < n; ++j) s -= A[(size_t)i * n + j] * x[j];
        x[i] = s / A[(size_t)i * n + i];
    }
    free(A);
    return 0;
}

/* ------------------------------------------------------------------ */
/* 钉扎/约束的目标值                                                   */
/* ------------------------------------------------------------------ */

static void pvk2d_apply_pins(pvk2d_ctx* c, double V, double* psi,
                             double* Yn, double* Zp) {
    int i, N = c->N;
    for (i = 0; i < N; ++i) {
        if (c->pin_psi && c->pin_psi[i])
            psi[i] = c->psi_neu[i] + ((c->pin_anod && c->pin_anod[i]) ? V : 0.0);
        if (c->pin_yn && c->pin_yn[i]) Yn[i] = c->yn_pin[i];
        if (c->pin_zp && c->pin_zp[i]) Zp[i] = c->zp_pin[i];
    }
    for (i = 0; i < N; ++i) {
        if (c->slave_p && c->slave_p[i]) {
            Yn[i] = fmax(Yn[i], 1e-300);
            Zp[i] = 1.0 / Yn[i];
        }
        if (c->slave_n && c->slave_n[i]) {
            Zp[i] = fmax(Zp[i], 1e-300);
            Yn[i] = 1.0 / Zp[i];
        }
    }
}

/* ------------------------------------------------------------------ */
/* 平衡初值（沿深度 1D Boltzmann-Poisson）                             */
/* ------------------------------------------------------------------ */

static int pvk2d_compute_equilibrium(pvk2d_ctx* c) {
    int ny = c->ny, nx = c->nx, n = c->N, i, j, it, lo;
    double Vt = c->Vt, q = PVK2D_Q;
    double *psi1, *am, *ap, *res, *du, *lo_a, *diag, *up_a;
    int iL = c->iL, iR = c->iR;

    if (!c->psi) return PVK_EAPI;
    psi1 = (double*)pvk2d_alloc(sizeof(double) * (size_t)ny);
    res = (double*)pvk2d_alloc(sizeof(double) * (size_t)ny);
    du = (double*)pvk2d_alloc(sizeof(double) * (size_t)ny);
    am = (double*)pvk2d_alloc(sizeof(double) * (size_t)ny);
    ap = (double*)pvk2d_alloc(sizeof(double) * (size_t)ny);
    lo_a = (double*)pvk2d_alloc(sizeof(double) * (size_t)ny);
    diag = (double*)pvk2d_alloc(sizeof(double) * (size_t)ny);
    up_a = (double*)pvk2d_alloc(sizeof(double) * (size_t)ny);
    if (!psi1 || !res || !du || !am || !ap || !lo_a || !diag || !up_a) {
        free(psi1); free(res); free(du); free(am); free(ap);
        free(lo_a); free(diag); free(up_a);
        return PVK_EMEM;
    }

    {
        int kR = (iR + 1 < ny ? iR + 1 : ny - 1) * nx;
        double pt = c->psi_neu[0], pb = c->psi_neu[kR];
        for (j = 0; j < ny; ++j)
            psi1[j] = pt + (pb - pt) * (ny > 1 ? (double)j / (ny - 1) : 0.0);
        for (j = 0; j <= iL; ++j) psi1[j] = c->psi_neu[j * nx];
        for (j = iR + 1; j < ny; ++j) psi1[j] = pb;
    }

    for (j = 0; j < ny - 1; ++j) {
        double h = c->y[j + 1] - c->y[j];
        if (h < 1e-30) h = 1e-30;
        ap[j] = 0.5 * (c->eps[(j + 1) * nx] + c->eps[j * nx]) / h;
        am[j] = ap[j];
    }
    for (j = 1; j < ny; ++j) am[j] = ap[j - 1];

    lo = (iL > 0) ? iL : 1;
    for (it = 0; it < 200; ++it) {
        int m = iR - lo + 1, k;
        double dmax = 0.0;
        for (j = lo; j <= iR; ++j) {
            double arg = (psi1[j] + c->u_off[j * nx]) / Vt;
            double nn, pp;
            arg = clampd(arg, -700.0, 700.0);
            nn = c->ni[j * nx] * exp(arg);
            pp = c->ni[j * nx] * exp(-arg);
            res[j] = (am[j] + ap[j]) * psi1[j] - am[j] * psi1[j - 1]
                     - ap[j] * psi1[j + 1]
                     - q * (pp - nn + (c->Nd[j * nx] - c->Na[j * nx])) * c->dyw[j];
            diag[j] = am[j] + ap[j] + q * c->dyw[j] * (nn + pp) / Vt;
            lo_a[j] = -am[j];
            up_a[j] = -ap[j];
        }
        /* Thomas */
        for (k = 0; k < m; ++k) {
            int jj = lo + k;
            double w = (k > 0) ? (lo_a[jj] / diag[jj - 1]) : 0.0;
            double dd = diag[jj] - ((k > 0) ? up_a[jj - 1] * w : 0.0);
            if (fabs(dd) < 1e-300) dd = 1e-300;
            diag[jj] = dd;
            res[jj] = -res[jj] - ((k > 0) ? res[jj - 1] * w : 0.0);
        }
        for (k = m - 1; k >= 0; --k) {
            int jj = lo + k;
            double s = res[jj];
            if (k < m - 1) s -= up_a[jj] * du[jj + 1];
            du[jj] = s / diag[jj];
        }
        for (j = lo; j <= iR; ++j) {
            double d = clampd(du[j], -0.3, 0.3);
            psi1[j] += d;
            if (fabs(d) > dmax) dmax = fabs(d);
        }
        if (dmax < 1e-11) break;
    }

    for (j = 0; j < ny; ++j) {
        double arg = clampd((psi1[j] + c->u_off[j * nx]) / Vt, -700.0, 700.0);
        for (i = 0; i < nx; ++i) {
            int idx = j * nx + i;
            c->psi[idx] = psi1[j];
            c->Yn[idx] = exp(arg);
            c->Zp[idx] = exp(-arg);
        }
    }
    pvk2d_apply_pins(c, 0.0, c->psi, c->Yn, c->Zp);
    c->have_state = 1;
    c->need_seed = 1;

    free(psi1); free(res); free(du); free(am); free(ap);
    free(lo_a); free(diag); free(up_a);
    (void)n;
    return PVK_OK;
}

/* ------------------------------------------------------------------ */
/* Newton                                                              */
/* ------------------------------------------------------------------ */

static double residual_max_scaled(pvk2d_ctx* c, const double* R) {
    int r;
    double m = 0.0;
    for (r = 0; r < c->n3; ++r) {
        double v = fabs(R[r] / c->srow[r]);
        if (v > m) m = v;
    }
    return m;
}

static int pvk2d_newton(pvk2d_ctx* c, double V, const double* G,
                        double* psi, double* Yn, double* Zp,
                        int maxit, double tol, double* res_out, int* it_out) {
    int N = c->N, it, i, r;
    int stagnant = 0;
    int lin_ok = -1, approx = 0;
    pvk2d_apply_pins(c, V, psi, Yn, Zp);
    for (it = 0; it < maxit; ++it) {
        double base, lam, upd, upd_full, rt = INFINITY;
        int got = 0, ls;
        pvk2d_assemble(c, V, G, psi, Yn, Zp, c->rhs, 1);
        pvk2d_scale_cols(c, Yn, Zp);
        if (getenv("PVK2D_DEBUG") && it == 0 && V == 0.0 && c->ny > 25) {
            int row = c->N + 25 * c->nx + 3, p;
            fprintf(stderr, "[pvk2d] rowdump Yn(25,3):");
            for (p = c->rowptr[row]; p < c->rowptr[row + 1]; ++p)
                fprintf(stderr, " (%d,%.3e)", c->col[p], c->val[p]);
            fprintf(stderr, "\n");
        }
        for (r = 0; r < c->n3; ++r) {
            double m = 0.0;
            int p;
            for (p = c->rowptr[r]; p < c->rowptr[r + 1]; ++p)
                if (fabs(c->val[p]) > m) m = fabs(c->val[p]);
            c->srow[r] = (m > 1e-300) ? m : 1e-300;
            c->rhs[r] = -c->rhs[r] / c->srow[r];      /* 解 J·du = −R */
            for (p = c->rowptr[r]; p < c->rowptr[r + 1]; ++p)
                c->val[p] /= c->srow[r];
        }
        base = 0.0;
        for (r = 0; r < c->n3; ++r) {
            double v = fabs(c->rhs[r]);
            if (v > base) base = v;
        }
        pvk2d_ilu0(c);
        lin_ok = pvk2d_bicgstab(c, c->rhs, c->du, 800, 1e-12, 0);
        approx = 0;
        if (lin_ok < 0) {
            /* 停滞时先用当前残差作新影子残差重启一次 */
            lin_ok = pvk2d_bicgstab(c, c->rhs, c->du, 800, 1e-12, 1);
            if (getenv("PVK2D_DEBUG"))
                fprintf(stderr, "[pvk2d] bicgstab restart -> %d (tr=%.2e)\n",
                        lin_ok, pvk2d_lin_rel(c, c->du));
        }
        if (lin_ok < 0) {
            /* 高偏压病态 Jacobian：ILU0-BiCGSTAB 会停滞（FG010 在 1.13 V
             * 起 800 步只到 8.6e-4）。先看它的部分解是否已够用（非精确
             * Newton，相对残差 ≤1e-2 就接受）；不够再退 GMRES(40)+ILU0、
             * GMRES(40)+对角预条件；都不行就用 ILU 预条件方向。 */
            double tr = pvk2d_lin_rel(c, c->du);
            if (getenv("PVK2D_DEBUG"))
                fprintf(stderr, "[pvk2d] bicgstab fail tr=%.2e, try GMRES\n", tr);
            if (tr > 1e-2) {
                lin_ok = pvk2d_gmres(c, c->rhs, c->du, 40, 600, 1e-10, 1);
                if (lin_ok < 0)
                    lin_ok = pvk2d_gmres(c, c->rhs, c->du, 40, 600, 1e-8, 0);
                tr = pvk2d_lin_rel(c, c->du);
                if (getenv("PVK2D_DEBUG"))
                    fprintf(stderr, "[pvk2d] GMRES rel=%.2e\n", tr);
            }
            if (lin_ok < 0 && tr > 1e-2) {
                /* 小系统最后退直接解（n3 ≤ 1800），近 Voc 病态时最稳 */
                lin_ok = pvk2d_dense_lu(c, c->rhs, c->du);
                tr = pvk2d_lin_rel(c, c->du);
                if (getenv("PVK2D_DEBUG"))
                    fprintf(stderr, "[pvk2d] dense LU -> %d (rel=%.2e)\n",
                            lin_ok, tr);
            }
            if (lin_ok < 0) {
                if (tr <= 1e-2) {
                    approx = 1;         /* 部分解作非精确 Newton 步 */
                } else {
                    pvk2d_ilu_solve(c, c->rhs, c->du);
                    approx = 1;
                }
            }
            if (getenv("PVK2D_DEBUG"))
                fprintf(stderr, "[pvk2d] linear fallback: %s (base=%.2e)\n",
                        approx ? "approx step" : "exact ok", base);
        }
        /* 迭代精化：病态系统下 ILU-BiCGSTAB 的残差不足以保证分量精度 */
        if (!approx) {
            int ref;
            for (ref = 0; ref < 4; ++ref) {
                double num = 0.0, den = 0.0;
                matvec(c, c->du, c->rbuf);
                for (i = 0; i < c->n3; ++i) {
                    c->rbuf[i] = c->rhs[i] - c->rbuf[i];
                    num += c->rbuf[i] * c->rbuf[i];
                    den += c->rhs[i] * c->rhs[i];
                }
                if (sqrt(num / fmax(den, 1e-300)) < 1e-13) break;
                if (pvk2d_bicgstab(c, c->rbuf, c->du2, 400, 1e-13, 0) < 0) break;
                for (i = 0; i < c->n3; ++i) c->du[i] += c->du2[i];
            }
            if (getenv("PVK2D_DEBUG")) {
                double num = 0.0, den = 0.0;
                matvec(c, c->du, c->rbuf);
                for (i = 0; i < c->n3; ++i) {
                    double d = c->rhs[i] - c->rbuf[i];
                    num += d * d; den += c->rhs[i] * c->rhs[i];
                }
                fprintf(stderr, "[pvk2d]   refine true_rel=%.2e\n",
                        sqrt(num / fmax(den, 1e-300)));
            }
        }
        if (getenv("PVK2D_DEBUG")) {
            double mp = 0.0, my = 0.0, mz = 0.0;
            for (i = 0; i < N; ++i) {
                if (fabs(c->du[i]) > mp) mp = fabs(c->du[i]);
                if (fabs(c->du[N + i]) > my) my = fabs(c->du[N + i]);
                if (fabs(c->du[2 * N + i]) > mz) mz = fabs(c->du[2 * N + i]);
            }
            fprintf(stderr, "[pvk2d] it=%d base=%.2e V=%.3f du: |dpsi|=%.2e "
                    "Vt|dlnY|=%.2e Vt|dlnZ|=%.2e\n", it, base, V,
                    mp, c->Vt * my, c->Vt * mz);
        }
        /* 保留未裁剪方向：逐分量裁剪失败时改沿该方向整体缩放 */
        memcpy(c->du2, c->du, sizeof(double) * (size_t)c->n3);
        for (i = 0; i < N; ++i) {
            c->du[i] = clampd(c->du[i], -0.3, 0.3);
            c->du[N + i] = clampd(c->du[N + i], -30.0, 30.0);
            c->du[2 * N + i] = clampd(c->du[2 * N + i], -30.0, 30.0);
        }
        upd_full = 0.0;
        for (i = 0; i < N; ++i) {
            double a = fabs(c->du[i]);
            double b = c->Vt * fabs(c->du[N + i]);
            double d = c->Vt * fabs(c->du[2 * N + i]);
            if (a > upd_full) upd_full = a;
            if (b > upd_full) upd_full = b;
            if (d > upd_full) upd_full = d;
        }
        lam = 1.0;
        {
            for (i = 0; i < N; ++i) {
                c->psi_bak[i] = psi[i];
                c->Yn_bak[i] = Yn[i];
                c->Zp_bak[i] = Zp[i];
            }
            for (ls = 0; ls < 12; ++ls) {
                for (i = 0; i < N; ++i) {
                    psi[i] = c->psi_bak[i] + lam * c->du[i];
                    Yn[i] = c->Yn_bak[i] * exp(lam * c->du[N + i]);
                    Zp[i] = c->Zp_bak[i] * exp(lam * c->du[2 * N + i]);
                }
                pvk2d_apply_pins(c, V, psi, Yn, Zp);
                pvk2d_assemble(c, V, G, psi, Yn, Zp, c->rtmp, 0);
                rt = residual_max_scaled(c, c->rtmp);
                if (getenv("PVK2D_DEBUG") && it == 0) {
                    int rw = 0, rr;
                    double mv = 0.0;
                    for (rr = 0; rr < c->n3; ++rr) {
                        double v = fabs(c->rtmp[rr] / c->srow[rr]);
                        if (v > mv) { mv = v; rw = rr; }
                    }
                    fprintf(stderr, "[pvk2d]   lam=%.3f rt=%.3e (base %.2e) "
                            "worst blk=%d node j=%d i=%d\n", lam, rt, base,
                            rw / c->N, (rw % c->N) / c->nx, (rw % c->N) % c->nx);
                    if (lam == 0.25 && c->ny > 25) {
                        int q1 = 25 * c->nx + 3, q2 = 25 * c->nx + 9;
                        fprintf(stderr, "[pvk2d]     Yn(25,3): R=%.4e srow=%.4e | "
                                "Yn(25,9): R=%.4e srow=%.4e | "
                                "Yn(25,0) val=%.3e Zp=%.3e\n",
                                c->rtmp[c->N + q1], c->srow[c->N + q1],
                                c->rtmp[c->N + q2], c->srow[c->N + q2],
                                Yn[q1], Zp[q1]);
                    }
                }
                if (isfinite(rt) && (rt <= base || rt < 1e-30)) { got = 1; break; }
                lam *= 0.5;
            }
            if (!got) {
                /* 逐分量裁剪会破坏 Newton 方向：极端贫化节点需要 Δln≈50，
                 * 而线性化在其它节点给出 1e14+ 的垃圾分量，裁剪后的方向
                 * 已不是下降方向。改沿未裁剪方向整体缩放（ψ≤0.5 V、
                 * ln≤3 每步，再线搜索）——小步长下必为下降方向。 */
                double ap = 0.0, ay = 0.0, az = 0.0, lam0, rawfull;
                int g2;
                for (i = 0; i < N; ++i) {
                    double a = fabs(c->du2[i]);
                    double b = fabs(c->du2[N + i]);
                    double d = fabs(c->du2[2 * N + i]);
                    if (a > ap) ap = a;
                    if (b > ay) ay = b;
                    if (d > az) az = d;
                }
                rawfull = ap;
                if (c->Vt * ay > rawfull) rawfull = c->Vt * ay;
                if (c->Vt * az > rawfull) rawfull = c->Vt * az;
                lam0 = 1.0;
                if (0.5 / (ap > 1e-30 ? ap : 1e-30) < lam0)
                    lam0 = 0.5 / (ap > 1e-30 ? ap : 1e-30);
                if (3.0 / (ay > 1e-30 ? ay : 1e-30) < lam0)
                    lam0 = 3.0 / (ay > 1e-30 ? ay : 1e-30);
                if (3.0 / (az > 1e-30 ? az : 1e-30) < lam0)
                    lam0 = 3.0 / (az > 1e-30 ? az : 1e-30);
                lam = lam0;
                for (g2 = 0; g2 < 24; ++g2) {
                    for (i = 0; i < N; ++i) {
                        psi[i] = c->psi_bak[i] + lam * c->du2[i];
                        Yn[i] = c->Yn_bak[i] * exp(lam * c->du2[N + i]);
                        Zp[i] = c->Zp_bak[i] * exp(lam * c->du2[2 * N + i]);
                    }
                    pvk2d_apply_pins(c, V, psi, Yn, Zp);
                    pvk2d_assemble(c, V, G, psi, Yn, Zp, c->rtmp, 0);
                    rt = residual_max_scaled(c, c->rtmp);
                    if (isfinite(rt) && (rt <= base || rt < 1e-30)) {
                        got = 1;
                        upd_full = rawfull;
                        break;
                    }
                    lam *= 0.5;
                }
                if (!got) {
                    /* 回滚本步 */
                    for (i = 0; i < N; ++i) {
                        psi[i] = c->psi_bak[i];
                        Yn[i] = c->Yn_bak[i];
                        Zp[i] = c->Zp_bak[i];
                    }
                    *res_out = base; *it_out = it;
                    return 0;
                }
            }
        }
        if (rt < 1e-30) rt = 1e-30;
        upd = lam * upd_full;
        *res_out = rt; *it_out = it + 1;
        if (upd < tol) return 1;
        if (rt < 1e-13) return 1;
        if (rt >= base * (1.0 - 1e-6)) ++stagnant; else stagnant = 0;
        if (stagnant >= 12) return 0;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* 电流                                                                */
/* ------------------------------------------------------------------ */

static double pvk2d_current(pvk2d_ctx* c) {
    int nx = c->nx, e, m = c->iR - c->iL;
    double sum = 0.0;
    if (m <= 0) return 0.0;
    pvk2d_compute_edges(c, c->psi);
    {
        /* 每个内部层 j∈[iL,iR) 的 y 边电流之和，再对层取平均 */
        double* per = (double*)pvk2d_alloc(sizeof(double) * (size_t)m);
        if (!per) return 0.0;
        for (e = 0; e < c->nye; ++e) {
            pvk2d_edge* ed = &c->ye[e];
            int j0 = ed->k0 / nx;
            if (j0 < c->iL || j0 >= c->iR) continue;
            {
                double Jn = ed->cn * c->Yn[ed->k1] - ed->an * c->Yn[ed->k0];
                double Jp = ed->bp * c->Zp[ed->k0] - ed->ap * c->Zp[ed->k1];
                per[j0 - c->iL] += Jn + Jp;
            }
        }
        for (e = 0; e < m; ++e) sum += per[e];
        sum /= (double)m;
        free(per);
    }
    return -sum / c->width;
}

/* ------------------------------------------------------------------ */
/* 公开 API                                                            */
/* ------------------------------------------------------------------ */

int pvk2d_abi_version(void) { return PVK2D_ABI_VERSION; }

pvk2d_ctx* pvk2d_create(void) {
    pvk2d_ctx* c = (pvk2d_ctx*)pvk2d_alloc(sizeof(pvk2d_ctx));
    if (c) c->err[0] = '\0';
    return c;
}

void pvk2d_destroy(pvk2d_ctx* c) {
    if (!c) return;
    pvk2d_free_arrays(c);
    free(c);
}

int pvk2d_last_error(pvk2d_ctx* c, char* buf, int buflen) {
    int n;
    if (!c || !buf || buflen <= 0) return 0;
    n = (int)strlen(c->err);
    if (n >= buflen) n = buflen - 1;
    memcpy(buf, c->err, (size_t)n);
    buf[n] = '\0';
    return n + 1;
}

int pvk2d_equilibrium(pvk2d_ctx* c) {
    if (!c || !c->psi) return PVK_EAPI;
    return pvk2d_compute_equilibrium(c);
}

int pvk2d_solve(pvk2d_ctx* c, double V, pvk2d_info* info) {
    int it = 0;
    double res = INFINITY;
    int ok;
    double *sp = NULL, *sy = NULL, *sz = NULL;
    if (!c || !c->psi) return PVK_EAPI;
    if (!c->have_state) {
        int rc = pvk2d_compute_equilibrium(c);
        if (rc != PVK_OK) return rc;
    }
    int from_eq = c->need_seed;      /* 本次是否从平衡态冷启动 */
    if (c->need_seed) {
        /* 光生载流子初值（只改初值）：n≈G·τ */
        c->need_seed = 0;
        if (c->G && c->gen) {
            int i;
            double gmax = 0.0;
            for (i = 0; i < c->N; ++i) if (c->G[i] > gmax) gmax = c->G[i];
            if (gmax > 0.0) {
                for (i = 0; i < c->N; ++i) {
                    if (!c->gen[i] || c->G[i] <= 0.0) continue;
                    {
                        double nq = c->G[i] * c->taun[i] / c->ni[i];
                        double pq = c->G[i] * c->taup[i] / c->ni[i];
                        if (c->Yn[i] < nq) c->Yn[i] = nq;
                        if (c->Zp[i] < pq) c->Zp[i] = pq;
                    }
                }
                pvk2d_apply_pins(c, V, c->psi, c->Yn, c->Zp);
            }
        }
    }
    /* 失败回滚：记录本次求解前的状态，所有尝试都失败时恢复，
     * 避免坏解污染后续偏压（与 Python fv2d.solve_iv 的暖启动回滚同口径）。 */
    sp = (double*)pvk2d_alloc(sizeof(double) * (size_t)c->N);
    sy = (double*)pvk2d_alloc(sizeof(double) * (size_t)c->N);
    sz = (double*)pvk2d_alloc(sizeof(double) * (size_t)c->N);
    if (!sp || !sy || !sz) {
        free(sp); free(sy); free(sz);
        return PVK_EMEM;
    }
    memcpy(sp, c->psi, sizeof(double) * (size_t)c->N);
    memcpy(sy, c->Yn, sizeof(double) * (size_t)c->N);
    memcpy(sz, c->Zp, sizeof(double) * (size_t)c->N);
    ok = pvk2d_newton(c, V, c->G, c->psi, c->Yn, c->Zp, 80, 1e-9, &res, &it);
    if (!ok && from_eq && c->G) {
        /* 光强 ramp 回退（与 Python fv2d.solve 同一序列）：从平衡态按
         * 1e-10…1 的几何序列逐步升光强，每步暖启动。细化网格下首步全光照
         * 的线性化跨 20+ 个量级，直接解会失败；逐步升光强每步跨 ~1 个
         * 量级，稳定收敛。 */
        double* gs = (double*)pvk2d_alloc(sizeof(double) * (size_t)c->N);
        int k, ramp_ok = 1;
        if (!gs) return PVK_EMEM;
        if (pvk2d_compute_equilibrium(c) != PVK_OK) {
            ramp_ok = 0;
        }
        for (k = 0; k < 11 && ramp_ok; ++k) {
            double alpha = pow(10.0, -10.0 + (double)k);
            int i;
            for (i = 0; i < c->N; ++i) gs[i] = c->G[i] * alpha;
            if (!pvk2d_newton(c, V, gs, c->psi, c->Yn, c->Zp,
                              80, 1e-9, &res, &it)) {
                ramp_ok = 0;
            }
        }
        free(gs);
        c->need_seed = 0;
        ok = ramp_ok;
    }
    if (!ok) {
        /* 回滚到本次求解前的状态（失败点不污染后续偏压） */
        memcpy(c->psi, sp, sizeof(double) * (size_t)c->N);
        memcpy(c->Yn, sy, sizeof(double) * (size_t)c->N);
        memcpy(c->Zp, sz, sizeof(double) * (size_t)c->N);
    }
    free(sp); free(sy); free(sz);
    if (info) {
        info->n_node = c->N;
        info->J_mA_cm2 = ok ? pvk2d_current(c) * 1e3 : 0.0;
        info->converged = ok;
        info->residual = res;
        info->iterations = it;
        info->v_applied = V;
    }
    if (!ok) {
        char buf[128];
        snprintf(buf, sizeof(buf), "Newton failed at V=%.4f (it=%d res=%.2e)", V, it, res);
        pvk2d_seterr(c, buf);
        return PVK_ECONV;
    }
    return PVK_OK;
}

int pvk2d_get_state(pvk2d_ctx* c, double* psi, double* Yn, double* Zp) {
    if (!c || !c->psi) return PVK_EAPI;
    if (psi) memcpy(psi, c->psi, sizeof(double) * (size_t)c->N);
    if (Yn) memcpy(Yn, c->Yn, sizeof(double) * (size_t)c->N);
    if (Zp) memcpy(Zp, c->Zp, sizeof(double) * (size_t)c->N);
    return PVK_OK;
}

int pvk2d_get_field(pvk2d_ctx* c, int field, double* out, int n) {
    int i, N;
    if (!c || !c->psi) return PVK_EAPI;
    N = c->N;
    if (!out || n < N) return PVK_EINVAL;
    for (i = 0; i < N; ++i) {
        switch (field) {
            case PVK2D_PSI:  out[i] = c->psi[i]; break;
            case PVK2D_N:    out[i] = c->ni[i] * c->Yn[i]; break;
            case PVK2D_P:    out[i] = c->ni[i] * c->Zp[i]; break;
            case PVK2D_PHIN:
                out[i] = c->psi[i] + c->u_off[i]
                         - c->Vt * log(fmax(c->Yn[i], 1e-300)); break;
            case PVK2D_PHIP:
                out[i] = c->psi[i] + c->u_off[i]
                         + c->Vt * log(fmax(c->Zp[i], 1e-300)); break;
            default: return PVK_EINVAL;
        }
    }
    return PVK_OK;
}
