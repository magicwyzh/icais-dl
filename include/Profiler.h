#pragma once
#include <chrono>
namespace icdl{
class Operator;

struct ProfileResults{
public:
    std::chrono::nanoseconds time_duration{0};
    int64_t num_cycle{0};
    std::chrono::microseconds get_time_duration_us() const;
    std::chrono::milliseconds get_time_duration_ms() const;
};

class Profiler{
private:
    Operator* _op{nullptr};
    std::chrono::time_point<std::chrono::system_clock> _start_time;
public: 
    Profiler(Operator* op_ptr = nullptr);
    virtual ~Profiler();
};
}//namespace icdl