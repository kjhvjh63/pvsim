/*
 * pvk2d.h — 2D 稳态漂移-扩散求解器（原型，独立 ABI，不改 1D 接口）。
 *
 * 与 Python 原型 perovskite/solvers/fv2d.py 的离散完全一致：
 *   · 张量积网格（x 横向、y 深度），有限体积 + Scharfetter–Gummel 边通量；
 *   · 未知量 (ψ, Yn, Zp)，n = ni·Yn、p = ni·Zp；
 *   · 全耦合阻尼 Newton + 稀疏 Jacobian（行均衡 + ILU(0)/BiCGSTAB）。
 *
 * 节点参数、掩码与初值由调用方（Python 侧网格/参数装配）按节点数组传入，
 * 这样 M4-4 只移植数值核心；网格生成、材料映射与光学仍在 Python，
 * 待数值对齐后再下沉（M4-5）。
 *
 * 节点编号：idx = j*nx + i（y 从上电极到下电极，x 从左到右），与 fv2d 相同。
 */
#ifndef PVK_PVK2D_H
#define PVK_PVK2D_H

#include "pvk/pvk.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PVK2D_ABI_VERSION 1

typedef struct pvk2d_ctx pvk2d_ctx;

/* 求解结果 */
typedef struct {
    int    n_node;
    double J_mA_cm2;      /* 阳极→阴极（底部→顶部）方向；光照短路为负 */
    int    converged;
    double residual;
    int    iterations;
    double v_applied;
} pvk2d_info;

/* 场编号（pvk2d_get_field） */
enum pvk2d_field {
    PVK2D_PSI = 0,
    PVK2D_N   = 1,
    PVK2D_P   = 2,
    PVK2D_PHIN = 3,
    PVK2D_PHIP = 4
};

/* 二维网格 + 逐节点参数（全部长度 nx*ny，除 x/y/dxw 外） */
typedef struct {
    int    nx, ny;
    double T_K;
    const double* x_cm;          /* [nx] */
    const double* y_cm;          /* [ny] */
    const double* dxw;           /* [nx] 横向控制体宽度（cm） */

    const double* eps;           /* F/cm */
    const double* chi;           /* eV */
    const double* Eg;            /* eV */
    const double* Nc;            /* cm^-3 */
    const double* Nv;
    const double* ni;
    const double* u_off;
    const double* mun;
    const double* mup;
    const double* taun;
    const double* taup;
    const double* brad;
    const double* Nd;
    const double* Na;
    const double* vol;           /* 控制体面积 cm^2 */

    /* 钉扎/约束掩码（0/1） */
    const unsigned char* pin_psi;
    const unsigned char* pin_yn;
    const unsigned char* pin_zp;
    const unsigned char* pin_anod;   /* pin_psi 的子集：ψ 目标 = psi_neu + V */
    const unsigned char* slave_n;    /* 电子行换成 Yn·Zp=1 约束 */
    const unsigned char* slave_p;
    const unsigned char* surf;       /* 顶面空穴 SRV 节点 */
    const unsigned char* surf_e;     /* 顶面电子 SRV 节点 */
    const unsigned char* gen;        /* 吸收层节点（光生载流子初值用），可 NULL */

    const double* psi_neu;       /* [N] 电中性电位 */
    const double* yn_pin;        /* [N] 钉扎 Yn 值 */
    const double* zp_pin;        /* [N] 钉扎 Zp 值 */
    const double* p0s;           /* [N] 空穴 SRV 平衡浓度，可 NULL */
    const double* n0s;           /* [N] 电子 SRV 平衡浓度，可 NULL */
    double top_srv;              /* cm/s */

    int    iL, iR;               /* 电流统计用的内部层索引区间 [iL, iR) */
    double width_cm;             /* 器件横向总宽度 */
} pvk2d_mesh;

PVK_API int pvk2d_abi_version(void);
PVK_API pvk2d_ctx* pvk2d_create(void);
PVK_API void pvk2d_destroy(pvk2d_ctx* ctx);
PVK_API int pvk2d_last_error(pvk2d_ctx* ctx, char* buf, int buflen);

/* 设置网格与逐节点参数（内部深拷贝）。 */
PVK_API int pvk2d_set_mesh(pvk2d_ctx* ctx, const pvk2d_mesh* mesh);

/* 产生率 G（cm^-3 s^-1，长度 nx*ny）。 */
PVK_API int pvk2d_set_generation(pvk2d_ctx* ctx, const double* G);

/* 平衡初值（暗态 V=0）。solve 首次调用会自动执行。 */
PVK_API int pvk2d_equilibrium(pvk2d_ctx* ctx);

/* 单点求解；warm start 由 ctx 内部状态自动保留。 */
PVK_API int pvk2d_solve(pvk2d_ctx* ctx, double V, pvk2d_info* info);

/* 读取状态/场（out 长度 >= nx*ny）。 */
PVK_API int pvk2d_get_state(pvk2d_ctx* ctx, double* psi, double* Yn, double* Zp);
PVK_API int pvk2d_get_field(pvk2d_ctx* ctx, int field, double* out, int n);

#ifdef __cplusplus
}
#endif

#endif /* PVK_PVK2D_H */
