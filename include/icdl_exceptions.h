#pragma once
#include <boost/assert.hpp>
#include <sstream>

#define ICDL_ASSERT(cond, msg) do{ \
    if(!(cond)) \
    { \
        std::stringstream str; \
        str << msg; \
        BOOST_ASSERT_MSG(cond, str.str().c_str()); \
    } \
}while(0)

