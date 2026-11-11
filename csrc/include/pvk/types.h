/*
 * types.h — Plain Old Data (POD) structures that cross the C/Python
 * boundary.  Every field must be public, the struct must be
 * flat-allocatable (no function pointers, no variable arrays), and
 * the member order MUST match the Python structs.py ctypes layout.
 *
 * These structs are authored from schema.py's LayerSpec/DeviceSpec/
 * OpticsSpec/SweepSpec/SolverOpts — the mapping is 1:1.  If a field
 * is added to schema.py it MUST be added here; if removed, removed
 * here too.  No divergence is allowed.
 *
 * ABI stability rule: existing field order and types are frozen.
 * New fields go at the end of their struct.  ABI version is bumped
 * only when the Python side can no longer safely cast an older build.
 */
#ifndef PVK_TYPES_H
#define PVK_TYPES_H

#include "pvk.h"

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/* ------------------------------------------------------------ */
/* Field IDs — ordered to match VALID_FIELDS in schema.py.
 * Python and the kernel both index fields with this enum. */
enum pvk_field_id {
    PVK_FIELD_X       =  0,
    PVK_FIELD_PSI     =  1,
    PVK_FIELD_N       =  2,
    PVK_FIELD_P       =  3,
    PVK_FIELD_EFIELD  =  4,
    PVK_FIELD_JN      =  5,
    PVK_FIELD_JP      =  6,
    PVK_FIELD_JTOTAL  =  7,
    PVK_FIELD_R_NET   =  8,
    PVK_FIELD_G_OPT   =  9,
    PVK_FIELD_CB      = 10,
    PVK_FIELD_VB      = 11,
    PVK_FIELD_PHIN    = 12,
    PVK_FIELD_PHIP    = 13,
    PVK_FIELD_COUNT   = 14
};

/* ------------------------------------------------------------ */
/* pvk_layer — Python LayerSpec */
typedef struct {
    const char* name;              /* 友好标签，如 "MAPbI3" */
    const char* material;          /* .mat 库键名；NULL 则用显式参数 */
    double thick_nm;
    double eps_r;
    double Eg_eV;
    double chi_eV;
    double Nc_cm3;
    double Nv_cm3;
    double mun_cm2Vs;
    double mup_cm2Vs;
    double taun_s;
    double taup_s;
    double Nd_cm3;
    double Na_cm3;
    int    is_contact;             /* bool-ish, 0/1 */
    int    is_gen;                 /* bool-ish, 0/1 */
} pvk_layer;

/* ------------------------------------------------------------ */
/* pvk_device — Python DeviceSpec */
typedef struct {
    const pvk_layer* layers;
    int    n_layers;
    double T_K;
    int    nodes;
    double interface_refine;
} pvk_device;

/* ------------------------------------------------------------ */
/* pvk_optics_in — Python OpticsSpec */
typedef struct {
    const double* wl_nm;
    int    n_wl;
    const double* flux;            /* 长度 = n_wl，spectral photon flux */
    double intensity_mW_cm2;
    int    bounces;                /* 1 / 100 / 1000 */
    int    bandgap_mask;           /* 0/1 */
    const char* nk_dir;            /* NULL = 内核用默认 n,k */
} pvk_optics_in;

/* ------------------------------------------------------------ */
/* pvk_bias — Python SweepSpec 里的固定偏压边界条件 */
typedef struct {
    double anode_V;                /* 固定电极电位 V */
    double cathode_V;
    int    bc_type;                /* 0=OHMIC, 1=SCHOTTKY */
    double work_fn_anode_eV;       /* 仅 SCHOTTKY 生效 */
    double work_fn_cathode_eV;
} pvk_bias;

/* ------------------------------------------------------------ */
/* pvk_solver_opts — Python SolverOpts */
typedef struct {
    double res_tol;
    int    max_iter;
    int    method;                 /* 0=NEWTON, 1=GUMMEL, 2=NEWTON_WITH_WARMSTART */
    double damping;
    double v_init_step;
    double v_min_step;
    double v_max_step;
    int    clamp_residual;         /* 0/1 */
} pvk_solver_opts;

/* ------------------------------------------------------------ */
/* pvk_solve_info — 单次求解的元信息（内核只读，Python 写回 SimResult） */
typedef struct {
    int    n;                      /* 网格点数，由内核填入 */
    double J_total_mA_cm2;
    int    converged;              /* 0/1 */
    double residual;
    int    iterations;
    double v_applied;              /* 本次偏压（V） */
} pvk_solve_info;

#ifdef __cplusplus
}
#endif

#endif /* PVK_TYPES_H */
