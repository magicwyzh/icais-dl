aux_source_directory(. DIR_SRCS)


add_executable(test_all ${DIR_SRCS})
target_include_directories(test_all PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(test_all icdl gtest pthread)
