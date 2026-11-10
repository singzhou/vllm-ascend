# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

if (NOT "ascend910b" IN_LIST ASCEND_COMPUTE_UNIT)
    return()
endif()

set(PTO_ISA_INCLUDE_DIR "${OPS_TRANSFORMER_DIR}/third_party/pto-isa/include")
if (NOT EXISTS "${PTO_ISA_INCLUDE_DIR}/pto/pto-inst.hpp")
    message(FATAL_ERROR "MegaChunkGdn requires the pinned csrc/third_party/pto-isa submodule.")
endif()

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        mega_chunk_gdn_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME MegaChunkGdn
    OPTIONS --cce-auto-sync=on
            -Wno-deprecated-declarations
            -Wno-error
            -DGDN_KERNEL_NAME=mega_chunk_gdn
            -DGDN_D=128
            -DGDN_C=128
            -I${CMAKE_CURRENT_SOURCE_DIR}
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
            -I${PTO_ISA_INCLUDE_DIR}
            -I${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/include
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE mega_chunk_gdn ACLNNTYPE aclnn_exclude)
endif()
