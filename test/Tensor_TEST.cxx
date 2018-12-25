#include <gtest/gtest.h>
#include "Tensor.h"
#include <memory>
#include <random>
#include <iostream>
using namespace icdl;

// Only dense tensor in CPU Mem are tested.
TEST(TensorTest, InitTest){

    Tensor images_float({1,3,32,32}, TensorDataDescriptor().dtype(kFloat32));
    EXPECT_EQ(images_float.dtype(), kFloat32);
    EXPECT_EQ(images_float.size(), TensorSize({1,3,32,32}));
    EXPECT_EQ(images_float.get_data_location(), kCPUMem);
    EXPECT_EQ(images_float.get_mem_layout(), kDense);
    EXPECT_EQ(images_float.nelement(), 1*3*32*32);
    //should have data
    EXPECT_NE(images_float.data_ptr(), nullptr);
    // but dense tensor should not have aux_info
    EXPECT_EQ(images_float.aux_info_ptr(), nullptr);


    /*The following two initilizations should be the same except for the underlying storage*/
    Tensor images_fixpoint({8,3,32,32}, TensorDataDescriptor().dtype(kFixpoint).represent({8, true, 0}));
    Tensor images_fix_init_with_brace({8,3,32,32}, {8, true, 0});
    // Firstly test one of the tensor is initialized as expected
    TensorDataDescriptor descript(FixpointRepresent(8, true, 0));
    EXPECT_EQ(images_fixpoint.get_data_descript(), descript);
    EXPECT_EQ(images_fixpoint.dtype(), kFixpoint);
    EXPECT_EQ(images_fixpoint.size(), TensorSize({8,3,32,32}));
    EXPECT_EQ(images_fixpoint.get_data_location(), kCPUMem);
    EXPECT_EQ(images_fixpoint.get_mem_layout(), kDense);
    EXPECT_EQ(images_fixpoint.nelement(), 8*3*32*32);
    //should have data
    EXPECT_NE(images_fixpoint.data_ptr(), nullptr);
    // but dense tensor should not have aux_info
    EXPECT_EQ(images_fixpoint.aux_info_ptr(), nullptr);
    // Secondly test whether two tensor are the same except for underlying storage.
    EXPECT_NE(images_fixpoint, images_fix_init_with_brace);
    EXPECT_EQ(images_fixpoint.dtype(), images_fix_init_with_brace.dtype());
    EXPECT_EQ(images_fixpoint.size(), images_fix_init_with_brace.size());
    EXPECT_EQ(images_fixpoint.get_data_descript(), images_fix_init_with_brace.get_data_descript());
    EXPECT_EQ(images_fixpoint.get_data_location(), images_fix_init_with_brace.get_data_location());
    EXPECT_EQ(images_fixpoint.get_mem_layout(), images_fix_init_with_brace.get_mem_layout());
    EXPECT_EQ(images_fixpoint.nelement(), images_fix_init_with_brace.nelement());
    EXPECT_NE(images_fixpoint.data_ptr(), images_fix_init_with_brace.data_ptr());

    int8_t * raw_image_ptr = new int8_t[8*3*16*16];
    Tensor t_fr_blob(static_cast<void*>(raw_image_ptr), {8,3,16,16}, {8, true, 0});
    EXPECT_EQ(static_cast<int8_t*>(t_fr_blob.data_ptr()), raw_image_ptr);
    EXPECT_EQ(t_fr_blob.size(), TensorSize({8,3,16,16}));
    EXPECT_EQ(t_fr_blob.dtype(), kFixpoint);
    EXPECT_EQ(t_fr_blob.get_data_descript(), descript);
    EXPECT_EQ(t_fr_blob.nelement(), 8*3*16*16);
    EXPECT_EQ(t_fr_blob.get_data_location(), kCPUMem);
    delete [] raw_image_ptr;
}

TEST(StorageTest, InitTest){
    size_t num_data = 3*3*3*32; //e.g., weight tensor
    TensorDataLocation tensor_loc = CPU_MEMORY;
    /******* TEST 1 *************/
    // just test whether they are correctly init with those input args.
    StoragePtr float_cpu_storage(new icdl::Float32TensorStorage(num_data, tensor_loc));
    EXPECT_NE(nullptr, float_cpu_storage->data_ptr());
    EXPECT_EQ(tensor_loc, float_cpu_storage->get_data_location());
    EXPECT_EQ(num_data, float_cpu_storage->get_data_num());
    tensor_loc = ACCELERATOR_MEMORY;
    StoragePtr float_accelerator_storage(new icdl::Float32TensorStorage(num_data, tensor_loc));
    EXPECT_NE(nullptr, float_accelerator_storage->data_ptr());
    EXPECT_EQ(tensor_loc, float_accelerator_storage->get_data_location());
    EXPECT_EQ(num_data, float_accelerator_storage->get_data_num());

    FixpointRepresent fix_repre(8, true, 0);
    tensor_loc = CPU_MEMORY;
    StoragePtr fix8_cpu_storage(new icdl::FixpointTensorStorage(num_data, fix_repre, tensor_loc));
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
    auto cloned_float_storage = float_cpu_storage->clone();
    float* cloned_fp_raw_ptr = static_cast<float*>(cloned_float_storage->data_ptr());
    auto copied_float_storage = float_cpu_storage;
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

    auto cloned_fix8_storage = fix8_cpu_storage->clone();
    int8_t* cloned_fix8_raw_ptr = static_cast<int8_t*>(cloned_fix8_storage->data_ptr());
    auto copied_fix8_storage = fix8_cpu_storage;
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
