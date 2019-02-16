#pragma once
#ifdef LIBBOOST_ENABLE
#include <boost/assert.hpp>
#include <boost/stacktrace.hpp>
#endif
#include <sstream>
#include <iostream>
#include <assert.h>
#ifdef LIBBOOST_ENABLE
    #define ICDL_ASSERT(cond, msg) do{ \
        if(!(cond)) \
        { \
            std::stringstream str; \
            str << "-------------Assertion Messages:------------"<<std::endl;\
            str << msg << std::endl; \
            str << "Backtrace:" << std::endl;\
            str << boost::stacktrace::stacktrace() << std::endl;\
            str << "----------End of Assertion Messages---------"<<std::endl;\
            std::cerr << str.str();\
            BOOST_ASSERT_MSG(cond, str.str().c_str()); \
        } \
    }while(0)
#else
    #define ICDL_ASSERT(cond, msg) do{\
        if(!(cond))\
        {\
            std::stringstream str; \
            str << "-------------Assertion Messages:------------"<<std::endl;\
            str << msg << std::endl; \
            str << "----------End of Assertion Messages---------"<<std::endl;\
            std::cerr << str;\
            assert(cond);\
        }\
    }while(0)
#endif
