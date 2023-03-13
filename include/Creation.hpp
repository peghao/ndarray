#pragma once

#include "NdArray.hpp"

namespace nd {
    template<typename T>
    std::shared_ptr<NdArray<T>> empty(std::initializer_list<shape_t> shape) {
        auto X = std::make_shared<NdArray<T>>();
        X->dim = shape.size();
        X->shape = new shape_t[X->dim];
        dim_t i=0;
        for(auto &x: shape){
            X->shape[i++] = x;
        }

        X->data = new T[get_size(shape)];
        return X;
    }

    template<typename T>
    std::shared_ptr<NdArray<T>> array(std::initializer_list<T> data, std::initializer_list<nd::shape_t> shape = {}) {
        std::shared_ptr<NdArray<T>> X(new NdArray<T>(data));
        if(shape.size() == 0){
            return  X;
        }
        return reshape(X, shape);
    }

}