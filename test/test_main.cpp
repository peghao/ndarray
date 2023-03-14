//
// Created by penghao on 23-3-10.
//

#include <gtest/gtest.h>

#include "NdArray.hpp"

TEST(NDARRAY, NdArray){
    std::shared_ptr<nd::NdArray<int>> X(new nd::NdArray<int>{1,2,3});
    EXPECT_EQ(X->get_dim(), 1);
    EXPECT_EQ(nd::get_size(X),3);

    std::cout << nd::to_string(X) << std::endl;
    std::cout << nd::to_string(X->get_shape()) << std::endl;
}

TEST(CREATION, array){
    auto X = nd::array<int>({1,2,3,4}, {2,2});
    EXPECT_EQ(X->get_dim(),2);
    EXPECT_EQ(nd::get_size(X), 4);
    std::cout << nd::to_string(X) << std::endl;
}

TEST(CREATION, empty){
    auto X = nd::empty<int>({2,3});
    EXPECT_EQ(X->get_dim(), 2);
    EXPECT_EQ(nd::get_size(X), 6);
    std::cout << nd::to_string(X) << std::endl;
}

TEST(CREATION, constant){
    auto X = nd::constant<int>({2,3}, 1);
    EXPECT_EQ(X->get_dim(), 2);
    EXPECT_EQ(nd::get_size(X), 6);
    std::cout << nd::to_string(X) << std::endl;
}

TEST(SLICE, slice_item){
    auto X1 = nd::array<int>({1,2,3,4,5,6,7,8}, {4,2});
    auto X2 = nd::array<int>({1,2,3,4,5,6,7,8,9}, {3,1,3});
    EXPECT_EQ(nd::slice_item(X1,{0,0}), 1);
    EXPECT_EQ(nd::slice_item(X1,{3,1}), 8);
    EXPECT_EQ(nd::slice_item(X2, {0,0,0}), 1);
    EXPECT_EQ(nd::slice_item(X2, {2,0,2}), 9);
}

TEST(SLICE, slice_2d){
    auto X = nd::array<int>({1,2,3,4,5,6,7,8}, {4,2});
    auto Y1 = nd::slice(X, {{0,2}, {0,SLICE_END}});
    auto Y2 = nd::slice(X, {{2,SLICE_END}, {1,SLICE_END}});
    std::cout << nd::to_string(X) << std::endl;
    std::cout << nd::to_string(Y1) << std::endl;
    std::cout << nd::to_string(Y2->get_shape()) << std::endl;
}

TEST(BROADCAST, times_broadcast){
    auto X1 = nd::array<int>({1,2,3,4}, {2,1,2});
    auto X2 = nd::array<int>({-3,-2,-1,0,1,2,3,4}, {4,2});
    auto Y = nd::times_broadcast(X1,X2);
    EXPECT_EQ(Y->dim, 3);
    EXPECT_EQ(Y->shape[0], 2);EXPECT_EQ(Y->shape[1], 4);EXPECT_EQ(Y->shape[2], 2);

    std::cout << nd::to_string(Y) << std::endl;
}

TEST(TRANSFORM, matrix_transpose){
    auto X = nd::array<float>({1,0,0,1},{2,1,2});
    auto Y = nd::transpose(X);
    std::cout << nd::to_string(Y) << std::endl;

    X = nd::array<float>({1,2,3,4},{1,2,2});
    Y = nd::transpose(X);
    std::cout << nd::to_string(Y) << std::endl;
}

TEST(TRANSFORM, matmul){
    auto X1 = nd::array({1,2,3,4,5,6,7,8},{2,2,2});
    auto X2 = nd::array({9,10,11,12}, {2,2,1});
    auto Y = nd::matmul(X1,X2);
    std::cout << nd::to_string(Y) << std::endl;
}

TEST(TRANSFORM, get_rotation_matrix2d){
    auto X = nd::array<float>({1,0,0,1},{2,1,2});
    auto R = nd::getRotationMatrix2d <float>(nd::array<float>({M_PI/2, -M_PI/2}, {2,1}));
    auto Y = nd::matmul(R, nd::transpose(X));

    std::cout << nd::to_string(Y) << std::endl;
}

TEST(CREATION, fromfile){
    std::string file_path{"../test_data/a.bin"};
    auto X = nd::fromfile<float>(file_path);
    X = nd::reshape(X, {2,3});
    std::cout << nd::to_string(X) << std::endl;
}

int main(){
    ::testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}