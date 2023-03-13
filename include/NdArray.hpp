#pragma once

#include <iostream>
#include <initializer_list>
#include <vector>

#include "fmt/format.h"

namespace nd
{

typedef uint32_t dim_t;
typedef uint64_t shape_t;
#define SLICE_END UINT64_MAX

template <typename T>
class NdArray{

public:
    NdArray() = default;

    dim_t dim{0};
    shape_t *shape{nullptr};
    T *data;

public:

    NdArray(std::initializer_list<T> data){
        dim = 1;
        shape = new shape_t(data.size());
        this->data = new T[data.size()];

        shape_t i=0;
        for(auto &x: data){
            this->data[i++] = x;
        }
    }

    ~NdArray(){
        delete[] shape;
        delete[] data;
    }

    dim_t get_dim(){return this->dim;}

    std::shared_ptr<NdArray<shape_t>> get_shape(){
        auto X = std::make_shared<NdArray<shape_t>>();
        X->dim = 1;
        X->shape = new shape_t(dim);
        X->data = new shape_t[dim];
        for(dim_t i=0; i<X->dim; ++i)
            X->data[i] = shape[i];
        return X;
    };
};

    shape_t get_size(std::initializer_list<nd::shape_t> shape){
        shape_t size=1;
        for(auto &x: shape) size *= x;
        return size;
    }

    template <typename T>
    shape_t get_size(std::shared_ptr<NdArray<T>> X){
        shape_t size = 1;
        for(dim_t i=0; i<X->dim; ++i) size *= X->shape[i];
        return size;
    }

    template <typename T>
    std::string to_string(std::shared_ptr<NdArray<T>> X){
        std::string str;
        shape_t array_size = get_size(X);
        for(uint64_t i=0; i<array_size; ++i)
            str += (std::to_string(X->data[i]) + ",");
        return str;
    }

    template <typename T>
    std::shared_ptr<NdArray<T>> reshape(std::shared_ptr<NdArray<T>> X, std::initializer_list<shape_t> shape){
        if(get_size(X) != get_size(shape)){
            throw std::runtime_error("input shape does not match the input data!");
        }
        X->dim = shape.size();
        delete[] X->shape;

        X->shape = new shape_t[X->dim];
        shape_t i=0;
        for(auto &x: shape){
            X->shape[i++] = x;
        }
        return X;
    }

}


#include "Creation.hpp"
#include "Slice.hpp"
#include "Broadcast.hpp"