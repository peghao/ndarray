#pragma once

#include "NdArray.hpp"

#include <fcntl.h>
#include <cassert>
#include <unistd.h>
#include <sys/stat.h>

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
    std::shared_ptr<NdArray<T>> constant(std::initializer_list<shape_t> shape, T C) {
        auto X = empty<T>(shape);
        shape_t X_size = get_size(X);
        for(shape_t i=0; i<X_size; ++i){
            X->data[i] = C;
        }
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

    off_t get_file_size(const char *path)
    {
        assert(path != NULL);
        struct stat stat_buff = {};

        if(stat(path, &stat_buff) != 0){
            printf("%s\n", strerror(errno));
            exit(EXIT_FAILURE);
        }
        return stat_buff.st_size;
    }

    bool read_all(const char *file_path, uint8_t *buff, off_t read_size)
    {
        int fd = open(file_path, O_RDONLY);
        if(fd == -1){
            printf("%s: File open failed!\n", __func__);
            return false;
        }
        if(read(fd, buff, read_size) != read_size){
            printf("%s: File read failed!\n", __func__);
            return false;
        }
        close(fd);
        return true;
    }

    template<typename T>
    std::shared_ptr<NdArray<T>> fromfile(std::string &file_path, std::initializer_list<shape_t> shape={}){
        off_t file_length = get_file_size(file_path.c_str()); //in bytes
//        if(shape.size() == 0){
//            throw std::runtime_error("shape error!");
//        }
        auto Y = std::make_shared<NdArray<T>>();
        Y->dim = 1;
        Y->shape = new shape_t(file_length/sizeof(T));
        Y->data = new T[get_size(Y)];
        read_all(file_path.c_str(), (uint8_t *)(Y->data), file_length);
        return Y;

    }

}