#include "Profiler.h"
#include "Operator.h"
namespace icdl{

    std::chrono::microseconds ProfileResults::get_time_duration_us() const{
        return std::chrono::duration_cast<std::chrono::microseconds>(time_duration);
    }
    std::chrono::milliseconds ProfileResults::get_time_duration_ms() const{
        return std::chrono::duration_cast<std::chrono::milliseconds>(time_duration);
    }

    Profiler::Profiler(Operator* op_ptr){
        if(op_ptr && op_ptr->_profile){
            _op = op_ptr;
            _start_time = std::chrono::system_clock::now();
        }
    }

    Profiler::~Profiler(){
        if(_op && _op->_profile){
            auto end_time = std::chrono::system_clock::now();
            _op->_prof_results.time_duration = end_time - _start_time;
        }
    }
}//namespace icdl