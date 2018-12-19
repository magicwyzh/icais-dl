#include <gtest/gtest.h>
#include "TensorStorage.h"
#include <memory>
#include <random>
#include <iostream>
using namespace icdl;

TEST(StorageTest, InitTest){
    size_t num_data = 3*3*3*32; //e.g., weight tensor
    TensorDataLocation tensor_loc = CPU_MEMORY;
    /******* TEST 1 *************/
    // just test whether they are correctly init with those input args.
    StoragePtr float_cpu_storage(new Float32TensorStorage(num_data, tensor_loc));
    EXPECT_NE(nullptr, float_cpu_storage->data_ptr());
    EXPECT_EQ(tensor_loc, float_cpu_storage->get_data_location());
    EXPECT_EQ(num_data, float_cpu_storage->get_data_num());
    tensor_loc = ACCELERATOR_MEMORY;
    StoragePtr float_accelerator_storage(new Float32TensorStorage(num_data, tensor_loc));
    EXPECT_NE(nullptr, float_accelerator_storage->data_ptr());
    EXPECT_EQ(tensor_loc, float_accelerator_storage->get_data_location());
    EXPECT_EQ(num_data, float_accelerator_storage->get_data_num());

    FixpointRepresent fix_repre(8, true, 0);
    tensor_loc = CPU_MEMORY;
    StoragePtr fix8_cpu_storage(new FixpointTensorStorage(num_data, fix_repre, tensor_loc));
    EXPECT_NE(nullptr, fix8_cpu_storage->data_ptr());
    EXPECT_EQ(num_data, fix8_cpu_storage->get_data_num());
    EXPECT_EQ(fix_repre, fix8_cpu_storage->get_data_represent());

    /*********** TEST 2**************/
    //get data ptr, set to some value, do some copy, or clone, and compare.
    // directly manipulate the raw pointer, because in the Operator Implementation for FPGAs
    // one may directly use the raw pointer to do some memory copy from CPU Memory to FPGA Memory
    // random generator
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> normal_dist{0,2}; //mean=0, std_dev=2
    std::uniform_int_distribution<int8_t> uniform_int_dist{-128, 127};
    // float storage copy/clone Test
    float* fp_raw_ptr = static_cast<float*>(float_cpu_storage->data_ptr());
    for(size_t i = 0; i < float_cpu_storage->get_data_num(); i++){
        fp_raw_ptr[i] = normal_dist(gen);
    }
    StoragePtr cloned_float_storage = float_cpu_storage->clone();
    float* cloned_fp_raw_ptr = static_cast<float*>(cloned_float_storage->data_ptr());
    StoragePtr copied_float_storage = float_cpu_storage;
    StoragePtr copied_float_storage2(float_cpu_storage);
    // for copied tensor, should just be shallow copy
    EXPECT_EQ(copied_float_storage->data_ptr(), float_cpu_storage->data_ptr());
    EXPECT_EQ(copied_float_storage2->data_ptr(), float_cpu_storage->data_ptr());
    EXPECT_EQ(copied_float_storage->get_data_num(), float_cpu_storage->get_data_num());
    EXPECT_EQ(copied_float_storage->get_data_type(), float_cpu_storage->get_data_type());
    EXPECT_EQ(copied_float_storage->get_data_location(), float_cpu_storage->get_data_location());
    EXPECT_EQ(copied_float_storage2->get_data_num(), float_cpu_storage->get_data_num());
    EXPECT_EQ(copied_float_storage2->get_data_type(), float_cpu_storage->get_data_type());
    EXPECT_EQ(copied_float_storage2->get_data_location(), float_cpu_storage->get_data_location());
    EXPECT_EQ(copied_float_storage, float_cpu_storage);
    EXPECT_EQ(copied_float_storage2, float_cpu_storage);
    // for cloned tensor, the raw pointer should not be the same (deep copy)
    EXPECT_NE(cloned_float_storage->data_ptr(), float_cpu_storage->data_ptr());
    for(size_t i = 0; i < float_cpu_storage->get_data_num(); i++){
        EXPECT_EQ(cloned_fp_raw_ptr[i], fp_raw_ptr[i]);
    }
    // fix point storage copy/clone test
    int8_t* fix8_raw_ptr = static_cast<int8_t*>(fix8_cpu_storage->data_ptr());
    for(size_t i = 0; i < float_cpu_storage->get_data_num(); i++){
        fix8_raw_ptr[i] = uniform_int_dist(gen);
    }

    StoragePtr cloned_fix8_storage = fix8_cpu_storage->clone();
    int8_t* cloned_fix8_raw_ptr = static_cast<int8_t*>(cloned_fix8_storage->data_ptr());
    StoragePtr copied_fix8_storage = fix8_cpu_storage;
    StoragePtr copied_fix8_storage2(fix8_cpu_storage);
    EXPECT_EQ(copied_fix8_storage, fix8_cpu_storage);
    EXPECT_EQ(copied_fix8_storage2, fix8_cpu_storage);
    EXPECT_NE(cloned_fix8_raw_ptr, fix8_raw_ptr);
    EXPECT_EQ(copied_fix8_storage->get_data_num(), fix8_cpu_storage->get_data_num());
    EXPECT_EQ(copied_fix8_storage->get_data_type(), fix8_cpu_storage->get_data_type());
    EXPECT_EQ(copied_fix8_storage->get_data_location(), fix8_cpu_storage->get_data_location());
    EXPECT_EQ(copied_fix8_storage2->get_data_num(), fix8_cpu_storage->get_data_num());
    EXPECT_EQ(copied_fix8_storage2->get_data_type(), fix8_cpu_storage->get_data_type());
    EXPECT_EQ(copied_fix8_storage2->get_data_location(), fix8_cpu_storage->get_data_location());
    for(size_t i = 0; i < fix8_cpu_storage->get_data_num(); i++){
        EXPECT_EQ(cloned_fix8_raw_ptr[i], fix8_raw_ptr[i]);
    }

}
