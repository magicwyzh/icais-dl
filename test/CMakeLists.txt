aux_source_directory(. DIR_SRCS)

add_executable(test_all ${DIR_SRCS})
#add_executable(test_all test_all.cxx Tensor_TEST.cxx)

target_include_directories(test_all 
PRIVATE "${PROJECT_SOURCE_DIR}/include"
PRIVATE "${PYTORCH_LIB_DIR}/include"
PRIVATE "${PYTORCH_LIB_DIR}/include/torch/csrc/api/include"
PRIVATE "/usr/local/include")
link_directories(
"${PYTORCH_LIB_DIR}/lib"
"${PROJECT_BINARY_DIR}/lib"
"/usr/local/lib"
)
target_compile_options(test_all PRIVATE -Wall -Werror -no-pie -fno-pie)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -no-pie")
#target_link_libraries(test_all icdl /usr/local/lib/libgtest.a pthread torch caffe2 c10)
target_link_libraries(test_all icdl gtest gtest_main pthread)


