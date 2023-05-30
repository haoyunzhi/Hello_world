cmake_minimum_required(VERSION 3.25)
project(NEWNEWNEWNEWNEW_SM4_AVX2)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(NEWNEWNEWNEWNEW_SM4_AVX2
        call.cpp
        ctr+.cpp
        ctr.cpp
        function.h
        gcm+.cpp
        gcm.cpp
        sm4+.cpp
        sm4.cpp
        test.cpp)
