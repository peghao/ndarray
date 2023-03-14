#pragma once

#include "NdArray.hpp"

namespace nd{
    template <typename T>
    std::shared_ptr<NdArray<T>> getRotationMatrix2d(double theta){
        T s = (T)std::sin(theta), c = (T)std::cos(theta);
        return array({c,s,-s,c},{2,2});
    }

    template <typename T>
    std::shared_ptr<NdArray<T>> getRotationMatrix2d(std::shared_ptr<NdArray<T>> theta){
//        if(theta->dim != 1 && theta->shape[1] != 1){
//            throw std::runtime_error("the input  shape must be (N,) or (N,1)!");
//        }
        auto R = empty<T>({theta->shape[0], 2,2});
        for(shape_t i=0; i<R->shape[0]; ++i){
            shape_t offset = i*4;
            T c=std::cos(theta->data[i]), s=std::sin(theta->data[i]);
            (R->data + offset)[0] = c;
            (R->data + offset)[1] = s;
            (R->data + offset)[2] = -s;
            (R->data + offset)[3] = c;
        }
        return R;
    }

}