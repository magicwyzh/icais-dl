aux_source_directory(. DIR_SRCS)

add_executable(test_all ${DIR_SRCS})
add_executable(try main_src/try.cxx)
target_include_directories(test_all 
PRIVATE "${PROJECT_SOURCE_DIR}/include/protos"
PRIVATE "${PROJECT_SOURCE_DIR}/include"
PRIVATE "${PROJECT_SOURCE_DIR}/deps/include"
PRIVATE "${PROJECT_SOURCE_DIR}/deps/include/torch/csrc/api/include"
PRIVATE "${PROJECT_SOURCE_DIR}/models"
)
link_directories(
"${PROJECT_SOURCE_DIR}/deps/lib"
"${PROJECT_BINARY_DIR}/lib"
)

target_compile_options(test_all PRIVATE -Wall -Werror -no-pie -fno-pie -g)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -no-pie")
#target_link_libraries(test_all icdl /usr/local/lib/libgtest.a pthread torch caffe2 c10)
target_link_libraries(test_all icdl gtest gtest_main pthread protobuf dl icdl_models)


