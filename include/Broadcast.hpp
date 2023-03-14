#pragma once

#include "NdArray.hpp"

namespace nd{

    std::vector<shape_t> get_index(std::vector<shape_t> &size_list, shape_t i){
        std::vector<shape_t> index;
        shape_t remainder = i;
        for(auto &size: size_list){
            index.push_back(remainder/size);
            remainder = remainder%size;
        }
        return index;
    }

    template <typename T>
    void show(std::vector<T> X){
        for(auto &x: X){
            std::cout << x << ",";
        }
        std::cout << std::endl;
    }

    template <typename T>
    void pad_vector(std::vector<T> *X, size_t pad_size, T pad_value){
        if(pad_size < X->size()){
            throw std::runtime_error(fmt::format("Can not pad X(which is length {}) to {}", X->size(), pad_size));
        }
        pad_size = pad_size - X->size();
        for(size_t i=0; i<pad_size; ++i){
            X->insert(X->begin(), pad_value);
        }
    }

    template <typename T>
    std::shared_ptr<NdArray<T>> times_broadcast(std::shared_ptr<NdArray<T>> X1, std::shared_ptr<NdArray<T>>X2){
        dim_t min_dim = std::min(X1->dim, X2->dim);
        dim_t max_dim = std::max(X1->dim, X2->dim);

        // check if shape is match
        for(dim_t i=0; i<min_dim; ++i){
            shape_t shape_i_X1 = X1->shape[X1->dim-i-1], shape_i_X2 = X2->shape[X2->dim-i-1];
            if(shape_i_X1 != shape_i_X2 && ( shape_i_X1 != 1 && shape_i_X2 != 1)){
                throw std::runtime_error("shape not match!");
            }
        }

        std::vector<shape_t> X1_shape(X1->shape, X1->shape+X1->dim);
        std::vector<shape_t> X2_shape(X2->shape, X2->shape+X2->dim);
        if(X1_shape.size() > X2_shape.size()){
            // pad X2
            pad_vector<shape_t>(&X2_shape, max_dim, 1);
        }else{
            pad_vector<shape_t>(&X1_shape, max_dim, 1);
        }

        std::vector<shape_t> out_shape(max_dim);
        for(size_t i=0; i<max_dim; ++i){
            out_shape[i] = std::max(X1_shape[i], X2_shape[i]);
        }

        std::vector<bool> X1_repeat_flag(max_dim);
        std::vector<bool> X2_repeat_flag(max_dim);
        for(size_t i=0; i<max_dim; ++i){
            X1_repeat_flag[i] = X1_shape[i]==1 ? true : false;
            X2_repeat_flag[i] = X2_shape[i]==1 ? true : false;
        }

        auto Y = std::make_shared<NdArray<T>>();
        Y->dim = max_dim; // set dim
        Y->shape = new shape_t[max_dim];
        memcpy(Y->shape, out_shape.data(), max_dim*sizeof(shape_t)); //set shape
        Y->data = new T[get_size(Y)];

        shape_t Y_size = get_size(Y);
        auto Y_size_list = get_size_list(out_shape);
        auto X1_size_list = get_size_list(X1_shape);
        auto X2_size_list = get_size_list(X2_shape);
        for(shape_t global_index=0; global_index<Y_size; ++global_index){
            auto Y_index = get_index(Y_size_list, global_index);
//            show(index);
            std::vector<shape_t> X1_index(max_dim), X2_index(max_dim);
            for(size_t i=0; i<max_dim; ++i){
                X1_index[i] = X1_repeat_flag[i] == true ? 0 : Y_index[i];
                X2_index[i] = X2_repeat_flag[i] == true ? 0 : Y_index[i];
            }
            shape_t Y_offset = 0, X1_offset = 0, X2_offset = 0;

            for(size_t i=0; i<max_dim; ++i){
                Y_offset += Y_index[i]*Y_size_list[i];
                X1_offset += X1_index[i]*X1_size_list[i];
                X2_offset += X2_index[i]*X2_size_list[i];
            }
            *(Y->data + Y_offset) = (X1->data + X1_offset)[0] * (X2->data + X2_offset)[0];
        }
        return Y;
    }

    template <typename T>
    std::shared_ptr<NdArray<T>> add_broadcast(std::shared_ptr<NdArray<T>> X1, std::shared_ptr<NdArray<T>>X2){
        dim_t min_dim = std::min(X1->dim, X2->dim);
        dim_t max_dim = std::max(X1->dim, X2->dim);

        // check if shape is match
        for(dim_t i=0; i<min_dim; ++i){
            shape_t shape_i_X1 = X1->shape[X1->dim-i-1], shape_i_X2 = X2->shape[X2->dim-i-1];
            if(shape_i_X1 != shape_i_X2 && ( shape_i_X1 != 1 && shape_i_X2 != 1)){
                throw std::runtime_error("shape not match!");
            }
        }

        std::vector<shape_t> X1_shape(X1->shape, X1->shape+X1->dim);
        std::vector<shape_t> X2_shape(X2->shape, X2->shape+X2->dim);
        if(X1_shape.size() > X2_shape.size()){
            // pad X2
            pad_vector<shape_t>(&X2_shape, max_dim, 1);
        }else{
            pad_vector<shape_t>(&X1_shape, max_dim, 1);
        }

        std::vector<shape_t> out_shape(max_dim);
        for(size_t i=0; i<max_dim; ++i){
            out_shape[i] = std::max(X1_shape[i], X2_shape[i]);
        }

        std::vector<bool> X1_repeat_flag(max_dim);
        std::vector<bool> X2_repeat_flag(max_dim);
        for(size_t i=0; i<max_dim; ++i){
            X1_repeat_flag[i] = X1_shape[i]==1 ? true : false;
            X2_repeat_flag[i] = X2_shape[i]==1 ? true : false;
        }

        auto Y = std::make_shared<NdArray<T>>();
        Y->dim = max_dim; // set dim
        Y->shape = new shape_t[max_dim];
        memcpy(Y->shape, out_shape.data(), max_dim*sizeof(shape_t)); //set shape
        Y->data = new T[get_size(Y)];

        shape_t Y_size = get_size(Y);
        auto Y_size_list = get_size_list(out_shape);
        auto X1_size_list = get_size_list(X1_shape);
        auto X2_size_list = get_size_list(X2_shape);
        for(shape_t global_index=0; global_index<Y_size; ++global_index){
            auto Y_index = get_index(Y_size_list, global_index);
//            show(index);
            std::vector<shape_t> X1_index(max_dim), X2_index(max_dim);
            for(size_t i=0; i<max_dim; ++i){
                X1_index[i] = X1_repeat_flag[i] == true ? 0 : Y_index[i];
                X2_index[i] = X2_repeat_flag[i] == true ? 0 : Y_index[i];
            }
            shape_t Y_offset = 0, X1_offset = 0, X2_offset = 0;

            for(size_t i=0; i<max_dim; ++i){
                Y_offset += Y_index[i]*Y_size_list[i];
                X1_offset += X1_index[i]*X1_size_list[i];
                X2_offset += X2_index[i]*X2_size_list[i];
            }
            *(Y->data + Y_offset) = (X1->data + X1_offset)[0] + (X2->data + X2_offset)[0];
        }
        return Y;
    }

    template <typename T>
    void raw_matrix_mul(T *M1, T *M2, T *M, shape_t m, shape_t s, shape_t n){
        /*算法1：按照公式实现的，计算复杂度为O(m*s*n)*/
        for(size_t i=0; i<m*n; ++i) M[i] = 0;
        for(size_t i=0; i<m; i++)
            for(size_t j=0; j<n; j++)
                for(size_t k=0; k<s; k++)
                    M[i*n + j] += M1[i*s + k] * M2[k*n + j];
    }

    template <typename T>
    void matrix_transpose(T *M, T *M_T, shape_t m, shape_t n)
    {
        for(size_t i=0; i<n; i++)
            for(size_t j=0; j<m; j++)
                M_T[i*m+j] = M[j*n+i];
    }

    template <typename T>
    std::shared_ptr<NdArray<T>> matmul(std::shared_ptr<NdArray<T>> X1, std::shared_ptr<NdArray<T>> X2){
        // TODO check the input shape to make sure legal
        shape_t N=X1->shape[0], m=X1->shape[1], s=X1->shape[2], n=X2->shape[2];
        auto Y = std::make_shared<NdArray<T>>();
        Y->dim = 3;
        Y->shape = new shape_t[3]{N,m,n};
        Y->data = new T[get_size(Y)];
        shape_t Y_matrix_size = m*n, X1_matrix_size = m*s, X2_matrix_size = s*n;
        for(shape_t i=0; i<N; ++i){
            raw_matrix_mul(X1->data+i*X1_matrix_size, X2->data+i*X2_matrix_size, Y->data+i*Y_matrix_size, m,s,n);
        }

        return Y;
    }

    template <typename T>
    std::shared_ptr<NdArray<T>> transpose(std::shared_ptr<NdArray<T>> X){
        // TODO now only support 3d array
        shape_t N=X->shape[0], m=X->shape[1], n=X->shape[2];
        auto Y = std::make_shared<NdArray<T>>();
        Y->dim = 3;
        Y->shape = new shape_t[3]{N,n,m};
        Y->data = new T[get_size(Y)];
        for(shape_t i=0; i<N; ++i){
            matrix_transpose(X->data+i*m*n, Y->data+i*n*m, m,n);
        }
        return Y;
    }
}