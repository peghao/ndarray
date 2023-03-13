cmake_minimum_required(VERSION 3.16.3) #test in 3.16.3

project(ndarray)

find_package(GTest REQUIRED)
find_package(Threads REQUIRED)

add_executable(test_main test/test_main.cpp)
target_include_directories(test_main PUBLIC include)
target_link_libraries(test_main PUBLIC
    GTest::gtest
    Threads::Threads
        fmt
)