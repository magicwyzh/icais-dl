aux_source_directory(. DIR_LIB_SRCS)
#add_subdirectory(operators)
aux_source_directory(operators OPERATOR_SRCS)
aux_source_directory(protos PROTO_SRCS)

#include_directories("${PROJECT_SOURCE_DIR}/include")

add_library(icdl SHARED ${DIR_LIB_SRCS} ${OPERATOR_SRCS} ${PROTO_SRCS})
add_library(icdl_static STATIC ${DIR_LIB_SRCS} ${OPERATOR_SRCS} ${PROTO_SRCS})
set_target_properties(icdl_static PROPERTIES OUTPUT_NAME "icdl")
set_target_properties(icdl PROPERTIES CLEAN_DIRECT_OUTPUT 1)
set_target_properties(icdl_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)
target_link_libraries(icdl torch caffe2 c10 protobuf)
target_link_libraries(icdl_static torch caffe2 c10 protobuf)

