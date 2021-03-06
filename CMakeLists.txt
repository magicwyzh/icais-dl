cmake_minimum_required (VERSION 3.1)
project(ICAIS_DL)


SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

#set(PYTORCH_LIB_DIR "/home/wyzh/libtorch")

include_directories(
"${PROJECT_SOURCE_DIR}/deps/include"
"${PROJECT_SOURCE_DIR}/deps/include/torch/csrc/api/include"
"${PROJECT_SOURCE_DIR}/include"
"${PROJECT_SOURCE_DIR}/include/protos"
"${PROJECT_SOURCE_DIR}"
)
link_directories(
"${PROJECT_SOURCE_DIR}/deps/lib"
"${PROJECT_BINARY_DIR}/lib"
)
option(USE_PYTORCH "Use pytorch backend for testing." ON)
# We use the MACRO -D_GLIBCXX_USE_CXX11_ABI=0 is because: when we use the libtorch from binary package
# rather than build from source, the libtorch is built with GCC4.9 and the C++11 ABI is the old version.
# This makes us unable to use the new C++ABI to build ICDL in our GCC7.4. Meanwhile, the gtest used in this
# project is built by myself, and by default it use the new C++11 ABI. As a result, if we choose new ABI, then
# libtorch not work, if we use old ABI, the newly-built gtest not work. 
# To avoid rebuild pytorch, which is such a bummer, we choose to make ICDL compatible to PyTorch, and finally rebuild
# the gtest with Old ABI. 
# This comment here is to explain the reason for this macro, and if this project will be transplanted to other platform
# the ABI issue should be taken into consideration.
if(USE_PYTORCH)
    message("USE_PYTORCH: ON")
    add_definitions(-DPYTORCH_BACKEND_ENABLE)
else()
endif()
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0 -DLIBBOOST_ENABLE)

set(LIBRARY_OUTPUT_PATH "${PROJECT_BINARY_DIR}/lib")
set(EXECUTABLE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/bin")


add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(models)

include(CTest)
enable_testing()
add_test(NAME test_run COMMAND bin/test_all)

if(USE_PYTORCH)
    execute_process(
        COMMAND bash -c "cd ${PROJECT_SOURCE_DIR}/python && python test/serialize_resnet.py"
    )
endif()
