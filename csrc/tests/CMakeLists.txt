# ─────────────────────────────────────────────────────────────────────
# Tests
#
# v0.1 只有 1 个冒烟测试：加载 → ABI 校验 → create/destroy →
# pvk_build_mesh → 取 field → sweep_dc 单点。 后续把真实数值回归
# 加进来（同质结 dd_solver 交叉对照等）。
# ─────────────────────────────────────────────────────────────────────

add_executable(test_smoke test_smoke.c)
target_link_libraries(test_smoke PRIVATE pvk)
target_include_directories(test_smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)

add_test(NAME smoke COMMAND test_smoke)
