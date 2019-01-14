#include <gtest/gtest.h>
#include "Tensor.h"
#include <memory>
#include <random>
#include <iostream>
#include <fstream>
#include "protos/Tensor.pb.h"
#include "test_utils.h"
using namespace icdl;

TEST(TensorTest, Float2FixConvertTest){
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> normal_dist{0,2}; //mean=0, std_dev=2
    auto float_tensor = Tensor({2,3,32,32}, Float32Descriptor());
    auto data_ptr_float = static_cast<float*>(float_tensor.data_ptr());

    // random init
    for(size_t i = 0; i < float_tensor.nelement(); i++){
        data_ptr_float[i] = normal_dist(gen);
    }
    FixpointRepresent fix_repr_12b{12, true, 8};
    FixpointRepresent fix_repr_8b{8, true, 4};
    auto fix_descript = TensorDataDescriptor(fix_repr_12b);
    auto fix_tensor_12b = float_tensor.convert_to(fix_descript);
    auto data_ptr_int16 = static_cast<int16_t*>(fix_tensor_12b.data_ptr());
    EXPECT_EQ(fix_tensor_12b.nelement(), float_tensor.nelement());
    EXPECT_EQ(fix_tensor_12b.dtype(), kFixpoint);
    EXPECT_NE(fix_tensor_12b.data_ptr(), float_tensor.data_ptr());
    for(size_t i = 0; i < fix_tensor_12b.nelement();i++){
        EXPECT_EQ(
            DefaultStorageConverter::single_data_flo32_to_fixp(data_ptr_float[i], fix_repr_12b),
            data_ptr_int16[i]
        );
    }
    auto fix_tensor_8b = float_tensor.convert_to(TensorDataDescriptor(fix_repr_8b));
    auto data_ptr_int8 = static_cast<int8_t*>(fix_tensor_8b.data_ptr());
    for(size_t i = 0; i < fix_tensor_8b.nelement(); i++){
        EXPECT_EQ(
            static_cast<int8_t>(DefaultStorageConverter::single_data_flo32_to_fixp(data_ptr_float[i], fix_repr_8b)),
            data_ptr_int8[i]
        );
    }
    auto x = Tensor({2,3,32,32}, Float32Descriptor());
    auto old_data_ptr = x.data_ptr();
    x = x.convert_to(fix_descript);
    auto new_data_ptr = x.data_ptr();
    EXPECT_NE(old_data_ptr, new_data_ptr);
}


TEST(TensorTest, Fix2FloatConvertTest){
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::uniform_int_distribution<int8_t> uniform_int_dist{-128, 127};
    auto int8_tensor = Tensor({1,3,32,32}, FixpointDescriptor(8, true, 3));
    auto data_ptr_int8 = static_cast<int8_t*>(int8_tensor.data_ptr());
    EXPECT_EQ(int8_tensor.dtype(), kFixpoint);

    // random init
    for(size_t i = 0; i < int8_tensor.nelement(); i++){
        data_ptr_int8[i] = uniform_int_dist(gen);
    }    
    auto float_tensor = int8_tensor.convert_to(Float32Descriptor());
    auto data_ptr_float = static_cast<float*>(float_tensor.data_ptr());
    for(size_t i = 0; i < int8_tensor.nelement(); i++){
        auto temp = DefaultStorageConverter::single_data_fixp_to_flo32(data_ptr_int8[i], int8_tensor.get_data_descript().get_represent().fix_point);
        EXPECT_EQ(
            temp,
            data_ptr_float[i]
        );
    }    
}

TEST(TensorTest, Fix2FixConvertTest){
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::uniform_int_distribution<int16_t> uniform_int_dist{-1024, 1023};
    auto int12_tensor = Tensor({8,3,32,32}, FixpointDescriptor(12, true, 6));
    auto data_ptr_int12 = static_cast<int16_t*>(int12_tensor.data_ptr());

    // random init
    for(size_t i = 0; i < int12_tensor.nelement(); i++){
        data_ptr_int12[i] = uniform_int_dist(gen);
        EXPECT_EQ(data_ptr_int12[i] < 1024, true);
    }    

    auto int6_tensor = int12_tensor.convert_to(FixpointDescriptor(6, true, 3));

    auto data_ptr_int6 = static_cast<int8_t*>(int6_tensor.data_ptr());

    for(size_t i = 0; i < int12_tensor.nelement(); i++){
        auto origin_data_16b = DefaultStorageConverter::fixpoint_to_int16(&data_ptr_int12[i], int12_tensor.get_data_descript().get_represent().fix_point, 0);
        auto temp = DefaultStorageConverter::single_data_fixp_to_fixp(origin_data_16b, int12_tensor.get_data_descript().get_represent().fix_point, int6_tensor.get_data_descript().get_represent().fix_point);
        auto temp_8b = static_cast<int8_t>(temp);
        EXPECT_EQ(
            temp_8b,
            data_ptr_int6[i]
        );
    }    
}

TEST(DefaultStorageConverter, fix2int16Test){
    int32_t mem_for_test = 0xfffffe73;//fe73=1111_1110_0111_0011
    auto cvt = [](const void* data_ptr, const FixpointRepresent& fix_repr,  const size_t bit_offset = 0){
        return static_cast<uint16_t>(DefaultStorageConverter::fixpoint_to_int16(data_ptr, fix_repr, bit_offset));
    };
    //FixpointRepresent fix_repr{8, true,0};
    // get lower 8 bits directly
    EXPECT_EQ(cvt(&mem_for_test, {8, true, 0}), 0b01110011);
    EXPECT_EQ(cvt(&mem_for_test, {6, true, 0}), 0b1111111111110011);// the 6-th bit is 1 and it is signed
    EXPECT_EQ(cvt(&mem_for_test, {6, false, 0}), 0b00110011);
    EXPECT_EQ(cvt(&mem_for_test, {16, true, 0}), 0xfe73);
    EXPECT_EQ(cvt(&mem_for_test, {16, false, 0}), 0xfe73);
    EXPECT_EQ(cvt(&mem_for_test, {10, true, 0}), 0b1111111001110011);
    EXPECT_EQ(cvt(&mem_for_test, {10, false, 0}), 0b0000001001110011);
    // compact memory
    EXPECT_EQ(cvt(&mem_for_test, {8, true, 0},  2), 0b1111111110011100);
    EXPECT_EQ(cvt(&mem_for_test, {8, false,0},  2), 0b0000000010011100);
    EXPECT_EQ(cvt(&mem_for_test, {6, true, 0},  2), 0b0000000000011100);
    EXPECT_EQ(cvt(&mem_for_test, {6, true, 0},  1), 0b1111111111111001);
    EXPECT_EQ(cvt(&mem_for_test, {10, false, 0}, 1), 0b0000001100111001);
    EXPECT_EQ(cvt(&mem_for_test, {10, true, 0},  2), 0b1111111110011100);
}

TEST(DefaultStorageConverterTest, SingleDataConvertTest){
    //auto& converter = DefaultStorageConverter::get();
    FixpointRepresent fix_represent_frac_1(5, true, 1);
    FixpointRepresent fix_represent_frac_n1(5, true, -1);
    /*****Float to fix point test**********/
    // 5.8125 = (sign) + 101.1101
    EXPECT_EQ(
        DefaultStorageConverter::single_data_flo32_to_fixp(5.8125, fix_represent_frac_1),
        12
    );
    EXPECT_EQ(
        DefaultStorageConverter::single_data_flo32_to_fixp(-5.8125, fix_represent_frac_1),
        -12
    );
    EXPECT_EQ(
        DefaultStorageConverter::single_data_flo32_to_fixp(5.8125, fix_represent_frac_n1),
        3
    );
    EXPECT_EQ(
        DefaultStorageConverter::single_data_flo32_to_fixp(-5.8125, fix_represent_frac_n1),
        -3
    );
    // 0.59375 = 0.100110
    EXPECT_EQ(
        DefaultStorageConverter::single_data_flo32_to_fixp(0.59375, fix_represent_frac_1),
        1
    );
    EXPECT_EQ(
        DefaultStorageConverter::single_data_flo32_to_fixp(-0.59375, fix_represent_frac_1),
        -1
    );
    // round to zero
    EXPECT_EQ(
        DefaultStorageConverter::single_data_flo32_to_fixp(0.59375, fix_represent_frac_n1),
        0
    );
    EXPECT_EQ(
        DefaultStorageConverter::single_data_flo32_to_fixp(-0.59375, fix_represent_frac_n1),
        0
    );
    // 29.8125=sign+11101.1101, quantize to 5bits, should saturate.
    EXPECT_EQ(
        DefaultStorageConverter::single_data_flo32_to_fixp(29.8125, fix_represent_frac_1),
        15
    );
    EXPECT_EQ(
        DefaultStorageConverter::single_data_flo32_to_fixp(-29.8125, fix_represent_frac_1),
        -16
    );
    
    /*** Fixpoint to Float Test****/
    FixpointRepresent repr{8, true, 4};
    auto fix2float = [](const int16_t src_data, const FixpointRepresent& fix_represent){
        return DefaultStorageConverter::single_data_fixp_to_flo32(src_data, fix_represent);
    };

    EXPECT_EQ(
        fix2float(53, repr),//0011.0101=53
        3.3125
    );
    EXPECT_EQ(
        fix2float(-53, repr),//-0011.0101=-53
        -3.3125
    );
    FixpointRepresent repr2{8, true, -4};
    EXPECT_EQ(
        fix2float(53, repr2),//0011.0101
        848
    );
    EXPECT_EQ(
        fix2float(-53, repr2),//-0011.0101
        -848
    );

    /*** Fixpoint to Fixpoint Test***/
    // dont change total_bits. just shift and truncate(floor)
    auto fix2fix = [](const int16_t src_data, 
                    const FixpointRepresent& src_fix_represent, 
                    const FixpointRepresent& dst_fix_represent){
        return DefaultStorageConverter::single_data_fixp_to_fixp(src_data, src_fix_represent, dst_fix_represent);
    };
    // from 4 frac bits to 2 frac bits
    EXPECT_EQ(
        fix2fix(0b01100110, {8, true, 4}, {8, true, 2}),
        0b00011001
    );
    
}


// Only dense tensor in CPU Mem are tested.
TEST(TensorTest, InitTest){

    Tensor images_float({1,3,32,32}, Float32Descriptor());
    EXPECT_EQ(images_float.dtype(), kFloat32);
    EXPECT_EQ(images_float.size(), TensorSize({1,3,32,32}));
    EXPECT_EQ(images_float.get_data_location(), kCPUMem);
    EXPECT_EQ(images_float.get_mem_layout(), kDense);
    EXPECT_EQ(images_float.nelement(), static_cast<size_t>(1*3*32*32));
    //should have data
    EXPECT_NE(images_float.data_ptr(), nullptr);
    // but dense tensor should not have aux_info
    EXPECT_EQ(images_float.aux_info_ptr(), nullptr);


    /*The following two initilizations should be the same except for the underlying storage*/
    Tensor images_fixpoint({8,3,32,32}, FixpointDescriptor(8, true, 0));
    Tensor images_fix_init_with_brace({8,3,32,32}, {8, true, 0});
    // Firstly test one of the tensor is initialized as expected
    TensorDataDescriptor descript(FixpointRepresent(8, true, 0));
    EXPECT_EQ(images_fixpoint.get_data_descript(), descript);
    EXPECT_EQ(images_fixpoint.dtype(), kFixpoint);
    EXPECT_EQ(images_fixpoint.size(), TensorSize({8,3,32,32}));
    EXPECT_EQ(images_fixpoint.get_data_location(), kCPUMem);
    EXPECT_EQ(images_fixpoint.get_mem_layout(), kDense);
    EXPECT_EQ(images_fixpoint.nelement(), static_cast<size_t>(8*3*32*32));
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
    EXPECT_EQ(t_fr_blob.nelement(), static_cast<size_t>(8*3*16*16));
    EXPECT_EQ(t_fr_blob.get_data_location(), kCPUMem);
    delete [] raw_image_ptr;
}

TEST(StorageTest, InitTest){
    size_t num_data = 3*3*3*32; //e.g., weight tensor
    TensorDataLocation tensor_loc = kCPUMem;
    /******* TEST 1 *************/
    // just test whether they are correctly init with those input args.
    StoragePtr float_cpu_storage(new icdl::Float32TensorStorage(num_data, tensor_loc));
    EXPECT_NE(nullptr, float_cpu_storage->data_ptr());
    EXPECT_EQ(tensor_loc, float_cpu_storage->get_data_location());
    EXPECT_EQ(num_data, float_cpu_storage->get_data_num());
    tensor_loc = kAccMem;
    StoragePtr float_accelerator_storage(new icdl::Float32TensorStorage(num_data, tensor_loc));
    EXPECT_NE(nullptr, float_accelerator_storage->data_ptr());
    EXPECT_EQ(tensor_loc, float_accelerator_storage->get_data_location());
    EXPECT_EQ(num_data, float_accelerator_storage->get_data_num());

    FixpointRepresent fix_repre(8, true, 0);
    tensor_loc = kCPUMem;
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


TEST(TensorTest, SeDeserializeTest){
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    auto t1 = icdl::Tensor({1,3,25,25}, FixpointDescriptor(8, true, 3));
    auto p1 = static_cast<int8_t*>(t1.data_ptr());
    EXPECT_EQ(icdl::TensorDataType::FIXPOINT, t1.dtype());
    for(size_t i = 0; i < t1.nelement();i++){
        p1[i] = i;
    }
    auto t1_pb = t1.serialize();

    std::fstream output("t1.icdl_tensor", std::ios::out| std::ios::trunc| std::ios::binary);
    t1_pb.SerializeToOstream(&output);

    auto t2 = icdl::Tensor();
    std::fstream input("t1.icdl_tensor", std::ios::in | std::ios::binary);
    icdl_proto::Tensor t2_pb;
    t2_pb.ParseFromIstream(&input);
    t2.deserialize(t2_pb);
    EXPECT_EQ(t1.get_data_descript().get_represent().fix_point, t2.get_data_descript().get_represent().fix_point);
    EXPECT_EQ(t1.nelement(), t2.nelement());
    auto p2 = static_cast<int8_t*>(t2.data_ptr());
    for(size_t i = 0; i < t1.nelement();i++){
        EXPECT_EQ(p1[i], p2[i]);
    }
    google::protobuf::ShutdownProtobufLibrary();
}
