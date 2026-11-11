# ─────────────────────────────────────────────────────────────────────
# Top-level CMakeLists.txt for libpvk
#
# Single source of truth: one build tree, one target (pvk), no
# "import targets vs hardcoded paths" duplication.  Tests link against
# the same target that Python loads via ctypes — so anything that
# breaks ctypes loading breaks the test too.
# ─────────────────────────────────────────────────────────────────────
cmake_minimum_required(VERSION 3.16)
project(pvk C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Debug / Release defaults — Python 端无所谓，但 Windows 下 DLL 导出
# 的符号表在 Release 里更干净。
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 头文件目录（public ABI）
set(PVK_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ─────────────────────────────────────────────────────────────────────
# Library target:  libpvk.{dll,so,dylib}
#
# 所有源文件集中在 src/，不再拆分子目录——首期代码量小，拆分反
# 而增加维护成本。需要再拆时再拆（physics/mesh/solver/optics/
# material/core 六个模块已经在注释里分区）。
# ─────────────────────────────────────────────────────────────────────
file(GLOB PVK_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c")

add_library(pvk SHARED ${PVK_SOURCES})
target_include_directories(pvk PUBLIC ${PVK_INCLUDE_DIRS})
set_target_properties(pvk PROPERTIES
    OUTPUT_NAME  pvk
    VERSION      0.1.0
    SOVERSION    0
)

# Windows: 显式 DLL 导出所有符号（首期简单做法）。
# 成熟后用 pvk_export.h 白名单控制。
if(WIN32)
    target_compile_definitions(pvk PRIVATE PVK_BUILDING)
    target_compile_options(pvk PRIVATE -Wall -Wextra)
endif()

# ─────────────────────────────────────────────────────────────────────
# Tests (ctest)
#
# 先只做 1 个冒烟测试：加载 libpvk → 校验 ABI 版本 → create/destroy。
# 后续真实 solver 数值回归加进来。
# ─────────────────────────────────────────────────────────────────────
enable_testing()
add_subdirectory(tests)
