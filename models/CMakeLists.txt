aux_source_directory(. DIR_LIB_SRCS)

add_library(icdl_models SHARED ${DIR_LIB_SRCS} ${OPERATOR_SRCS} ${PROTO_SRCS})
add_library(icdl_models_static STATIC ${DIR_LIB_SRCS} ${OPERATOR_SRCS} ${PROTO_SRCS})
set_target_properties(icdl_models_static PROPERTIES OUTPUT_NAME "icdl")
set_target_properties(icdl_models PROPERTIES CLEAN_DIRECT_OUTPUT 1)
set_target_properties(icdl_models_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)
target_link_libraries(icdl_models icdl)
target_link_libraries(icdl_models_static icdl)

