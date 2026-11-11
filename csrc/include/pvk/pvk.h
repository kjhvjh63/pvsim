/*
 * pvk.h — lifecycle, ABI version, error codes
 *
 * This header is the "core" of the public C ABI.  It defines the opaque
 * context handle, all public function signatures, and every error code.
 * The opaque struct is intentionally incomplete here — its full layout
 * lives in pvk_internal.h inside csrc/src.
 *
 * The ABI version number is a single int returned by pvk_abi_version().
 * Python checks it before making any other call and aborts if there is a
 * mismatch, preventing silent crashes when the library gets swapped for
 * an older build.
 */
#ifndef PVK_PVK_H
#define PVK_PVK_H

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------ */
/* ABI contract.  Bump when any public struct or function sig
 * changes.  Internal-only (pvk_internal.h) changes DO NOT bump. */
#define PVK_ABI_VERSION 1

/* ------------------------------------------------------------ */
/* PVK_API — DLL export/import decorator (Windows) or no-op (POSIX).
 * 由 CMake 在 build 侧定义 PVK_BUILDING；consumer 侧不定义它。 */
#if defined(_WIN32)
#  if defined(PVK_BUILDING)
#    define PVK_API __declspec(dllexport)
#  else
#    define PVK_API __declspec(dllimport)
#  endif
#else
#  define PVK_API
#endif

/* ------------------------------------------------------------ */
/* Error codes.  Every exported function returns one of these.
 * Last error message is exposed via pvk_last_error() for richer info. */
enum pvk_err {
    PVK_OK     = 0,
    PVK_EINVAL = 1,   /* invalid argument */
    PVK_ECONV  = 2,   /* solver failed to converge */
    PVK_EMEM   = 3,   /* out of memory */
    PVK_EIO    = 4,   /* file not found, bad format */
    PVK_EMAT   = 5,   /* material lookup / .mat parse failed */
    PVK_EAPI   = 6,   /* API usage order violated */
    PVK_EMATH  = 7    /* singularity / non-finite intermediate */
};

/* ------------------------------------------------------------ */
/* Opaque handle — the only state the library owns.
 * No global variables; everything hangs off pvk_ctx*. */
typedef struct pvk_ctx pvk_ctx;

/* ------------------------------------------------------------ */
/* Log callback.  Python sets this on create() so the GUI log panel
 * receives solver timings/iterations in real time.  msg is a
 * nul-terminated string borrowed by the caller — do not free inside
 * the callback and do not cache across the callback return. */
typedef void (*pvk_log_fn)(void* user, int level, const char* msg);

enum pvk_log_level {
    PVK_LOG_DEBUG = 0,
    PVK_LOG_INFO  = 1,
    PVK_LOG_WARN  = 2,
    PVK_LOG_ERROR = 3
};

/* ------------------------------------------------------------ */
/* Lifecycle */

int pvk_abi_version(void);

/* Create: allocates + zeros a fresh ctx.  log_cb may be NULL for
 * silent operation.  On failure returns NULL — call pvk_last_error
 * for detail before destroying anything. */
pvk_ctx* pvk_create(pvk_log_fn log_cb, void* user);

/* Destroy: frees everything owned by ctx.  Safe to call with NULL. */
void pvk_destroy(pvk_ctx* ctx);

/* Human-readable last-error.  buf must be caller-allocated; returns
 * the number of chars written (including NUL).  Always returns 0 on
 * PVK_OK. */
int pvk_last_error(pvk_ctx* ctx, char* buf, int buflen);

#ifdef __cplusplus
}
#endif

#endif /* PVK_PVK_H */
