aux_source_directory(. DIR_SRCS)

add_executable(test_all ${DIR_SRCS})
#add_executable(test_all test_all.cxx Tensor_TEST.cxx)

target_include_directories(test_all 
PRIVATE "/usr/local/include"
PRIVATE "${PROJECT_SOURCE_DIR}/include"
PRIVATE "${PYTORCH_LIB_DIR}/include"
PRIVATE "${PYTORCH_LIB_DIR}/include/torch/csrc/api/include")
link_directories(
"/usr/local/lib"
"${PYTORCH_LIB_DIR}/lib"
"${PROJECT_BINARY_DIR}/lib"
)
#target_compile_options(test_all PRIVATE -Wall -fPIC -Werror)
#target_link_libraries(test_all icdl /usr/local/lib/libgtest.a pthread torch caffe2 c10)
target_link_libraries(test_all icdl gtest pthread)


