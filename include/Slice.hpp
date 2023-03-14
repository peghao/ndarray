#pragma once

#include "NdArray.hpp"

#include <algorithm>

namespace nd{
//    template <typename T>
//    std::shared_ptr<nd::NdArray<T>> slice(std::shared_ptr<nd::NdArray<T>> X, std::initializer_list<nd::shape_t> slice_list){
//        if(slice_list.size() > X.dim){
//            throw std::runtime_error(fmt::format("The slice(which is length {}) is out of input array dims(which is {})!", slice_list.size(), X.dim));
//        } else if(slice_list.size() == X->dim){
//            throw std::runtime_error(fmt::format("The slice length is eq to input array dim, please call \"nd::slice_item()\" instead!"));
//        }
//
//        std::vector<nd::shape_t> size_list(1,1);
//        for(nd::shape_t i=0; i<X->dim; ++i){
//            size_list.push_back(size_list.back() * X->shape[i]);
//        }
//
//        std::reverse(size_list.begin(), size_list.end());
//        size_list.pop_back(); // remove "1"
//        size_list.erase(size_list.begin());
//
//        nd::shape_t offset = 0, length=0;
//        nd::shape_t i=0;
//        for(auto &x: slice_list){
//            if(x < 0){
//                throw std::runtime_error(fmt::format("slice index must be a non-negative value, but got {} at index {}", x, i));
//            }
//            offset += x * size_list[i++];
//        }
//
//        length = size_list[i-1];
//
//    }
    std::vector<shape_t> slice2vector(std::initializer_list<nd::shape_t> slice_list){
        std::vector<shape_t> slice_vec;
        for(auto &x: slice_list){
            slice_vec.push_back(x);
        }
        return slice_vec;
    }

    std::vector<std::vector<shape_t>> slice2vector2d(std::initializer_list<std::initializer_list<nd::shape_t>> slice_list){
        std::vector<std::vector<shape_t>> slice_vec;
        for(auto &row: slice_list){
            std::vector<shape_t> slice_row;
            for(auto &col: row){
                slice_row.push_back(col);
            }
            slice_vec.push_back(slice_row);
        }
        return slice_vec;
    }

    template <typename T>
    T& slice_item(std::shared_ptr<nd::NdArray<T>> X, std::initializer_list<nd::shape_t> slice_list){
        if(slice_list.size() != X->dim){
            throw std::runtime_error(fmt::format("Error in {}: index(which is dim {}) is not match input dim(which is {})", __func__, slice_list.size(), X->dim));
        }
        auto X_size_list = get_size_list(std::vector<shape_t>(X->shape, X->shape+X->dim));
        auto X_index = initial_list2vector(slice_list);

        shape_t X_offset = 0;

        for(size_t i=0; i<X->dim; ++i){
            X_offset += X_index[i]*X_size_list[i];
        }
        return X->data[X_offset];
    }

    template <typename T>
    std::shared_ptr<nd::NdArray<T>> slice(std::shared_ptr<nd::NdArray<T>> X, std::initializer_list<std::initializer_list<nd::shape_t>> slice_list){
        if(slice_list.size() != X->dim){
            throw std::runtime_error(fmt::format("Slice length(which is length {}) not eq to input array dims(which is {}) is not support!", slice_list.size(), X->dim));
        }
        if(X->dim > 2){
            throw std::runtime_error(fmt::format("Only 2d array is support now, but got input array dim {}", X->dim));
        }

        shape_t row_len=0, col_len = 0;
        auto slice_vec = slice2vector2d(slice_list);
        shape_t row_start = slice_vec[0][0];
        shape_t row_end = slice_vec[0][1] == SLICE_END ? X->shape[0] : slice_vec[0][1];
        shape_t col_start = slice_vec[1][0];
        shape_t col_end = slice_vec[1][1] == SLICE_END ? X->shape[1] : slice_vec[1][1];

        row_len = row_end - row_start;
        col_len = col_end - col_start;

        auto Y = empty<T>({row_len, col_len});
        for(shape_t i=0; i<row_len; ++i){
            shape_t X_width=X->shape[1],  X_offset=(i+row_start)*X_width + col_start, copy_len=col_len;
            shape_t Y_offset = i*col_len;
            memcpy(Y->data+Y_offset, X->data+X_offset, copy_len*sizeof(T));
        }

        return Y;
    }
}