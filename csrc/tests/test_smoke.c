/*
 * test_smoke.c — 最小冒烟测试。
 * 跑过 pvk_create/destroy + build_mesh + solve_dc + get_field，
 * 退出码非零 = 失败。
 */
#include "pvk/pvk.h"
#include "pvk/types.h"
#include "pvk/mesh.h"
#include "pvk/physics.h"
#include "pvk/solver.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg) do { if (!(cond)) { \
    fprintf(stderr, "FAIL: %s\n", msg); return 1; } } while(0)

static void log_cb(void* u, int lvl, const char* msg) {
    (void)lvl; (void)u;
    /* 首期忽略日志，printf 出来也行 */
    printf("[pvk log] %s\n", msg);
}

int main(void) {
    /* ABI 校验 */
    int abi = pvk_abi_version();
    CHECK(abi == PVK_ABI_VERSION, "ABI version mismatch");
    printf("ABI: %d (expected %d)\n", abi, PVK_ABI_VERSION);

    /* create */
    pvk_ctx* ctx = pvk_create(log_cb, NULL);
    CHECK(ctx != NULL, "pvk_create returned NULL");

    /* 一个 3 层最小 device：contact / bulk / contact
     * （电极层与吸收层的 chi 必须接近，否则亲和势差达到几个 eV，
     *   界面上载流子浓度差会超出 double 表示范围。） */
    pvk_layer layers[3] = {
        {"Cathode", NULL, 100.0, 9.0,  1.6, 4.0,  1e20, 1e20, 10.0, 10.0, 1e-9, 1e-9, 1e20, 0.0, 1, 0},
        {"Bulk",    NULL, 300.0, 10.0, 1.6, 3.93, 1e19, 1e19, 1.0,  1.0,  1e-9, 1e-9, 0.0,  0.0, 0, 1},
        {"Anode",   NULL, 100.0, 9.0,  1.6, 4.0,  1e20, 1e20, 10.0, 10.0, 1e-9, 1e-9, 0.0,  1e20, 1, 0},
    };

    pvk_device dev;
    dev.layers = layers; dev.n_layers = 3; dev.T_K = 300.0;
    dev.nodes = 31; dev.interface_refine = 0.0;

    /* mesh */
    int rc = pvk_build_mesh(ctx, &dev);
    char buf[256];
    if (rc != PVK_OK) {
        pvk_last_error(ctx, buf, sizeof(buf));
        fprintf(stderr, "build_mesh failed: %s\n", buf);
        return 1;
    }
    int n = pvk_node_count(ctx);
    CHECK(n >= 3 && n <= 100, "unexpected mesh node count");
    printf("mesh nodes: %d\n", n);

    /* optics：最小光谱 1 点 */
    double wl[1] = {550.0};
    double fl[1] = {1e15};
    pvk_optics_in opt = {wl, 1, fl, 100.0, 100, 1, NULL};

    /* bias */
    pvk_bias bias = {0.0, 0.0, 0, 4.7, 4.7};

    /* 单点 solve */
    pvk_solve_info info;
    rc = pvk_solve_dc(ctx, &dev, &opt, &bias, &info);
    if (rc != PVK_OK) {
        pvk_last_error(ctx, buf, sizeof(buf));
        fprintf(stderr, "solve_dc failed: %s\n", buf);
        return 1;
    }
    CHECK(info.converged == 1, "solve did not converge");
    printf("solve ok: n=%d, J_total=%g mA/cm2\n", info.n, info.J_total_mA_cm2);

    /* 读几个 field */
    double* xs = (double*)calloc(n, sizeof(double));
    CHECK(xs != NULL, "calloc");
    rc = pvk_get_x(ctx, xs, n);
    CHECK(rc == PVK_OK, "get_x");
    printf("x[0]=0, x[-1]=%g cm\n", xs[n-1]);
    free(xs);

    /* PSI */
    double* psi = (double*)calloc(n, sizeof(double));
    rc = pvk_get_field(ctx, PVK_FIELD_PSI, psi, n);
    CHECK(rc == PVK_OK, "get PSI");
    printf("PSI[0]=%g, PSI[mid]=%g\n", psi[0], psi[n/2]);
    free(psi);

    /* destroy */
    pvk_destroy(ctx);
    printf("ALL TESTS PASSED\n");
    return 0;
}
