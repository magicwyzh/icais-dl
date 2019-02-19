#include <iostream>
#include <cstddef>
#include <vector>
#include <bitset>
struct NormalStorage{
    int8_t value;
    int16_t channel_idx;
    int8_t kernel_idx;
};

struct __attribute__ ((packed)) CompactStorage{
    int16_t channel_idx: 12;
    int8_t kernel_idx: 4;
    int8_t value: 8;
};

int main(){
    NormalStorage ns;
    CompactStorage cs;
    std::cout<< "NS=" << sizeof(ns) << ", CS = " << sizeof(cs) << std::endl;
    std::vector<CompactStorage> cs_vec;
    cs_vec.emplace_back(CompactStorage{1, 2, 3});
    cs_vec.emplace_back(CompactStorage{4, 5, 6});
    cs_vec.emplace_back(CompactStorage{7, 8, 9});
    auto data_ptr = (int32_t*)(cs_vec.data());
    std::bitset<32> x(*data_ptr);
    std::cout << data_ptr <<":" << x << std::endl;
    data_ptr++;
    std::bitset<32> x2(*data_ptr);
    std::cout << data_ptr <<":" << x2 << std::endl;
    data_ptr++;
    std::bitset<32> x3(*data_ptr);
    std::cout << data_ptr <<":" << x3 << std::endl;
    return 0;
}