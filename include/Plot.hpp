#pragma once

#include "NdArray.hpp"

extern "C"{
    #include <png.h>
}

namespace plt{

    int image_save(const char *outfile, uint8_t *data, uint32_t width, uint32_t height, int color_type=PNG_COLOR_TYPE_RGB){
        FILE *fp;
        png_structp png_ptr;
        png_infop info_ptr;
        uint32_t row_size;
        int bit_depth = 1;

        if(color_type == PNG_COLOR_TYPE_GRAY){
            bit_depth = 1;
            row_size = width;
        } else if(color_type == PNG_COLOR_TYPE_RGB) {
            bit_depth = 8;
            row_size = width*3;
        } else if(color_type == PNG_COLOR_TYPE_RGBA) {
            bit_depth = 8;
            row_size = width*4;
        } else {
            printf("not support color\n");
            return -1;
        }

        fp = fopen(outfile, "wb");
        if(fp == NULL) {
            printf("fail to create %s\n", outfile);
            return -1;
        }

        png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
        if(png_ptr == NULL) {
            printf("Failed to png_create_write_struct\n");
            fclose(fp);
            return -1;
        }

        info_ptr = png_create_info_struct(png_ptr);
        if(info_ptr == NULL) {
            printf("Failed to png_create_info_struct\n");
            fclose(fp);
            return -1;
        }
        if(setjmp(png_jmpbuf(png_ptr)))
        {
            printf("call png_jmpbuf fail!\n");
            fclose(fp);
            return -1;
        }

        png_init_io(png_ptr, fp);
        png_set_IHDR(png_ptr, info_ptr,
                     width, height,
                     bit_depth,//bit_depth
                     color_type,//color_type
                     PNG_INTERLACE_NONE,//interlace or not
                     PNG_COMPRESSION_TYPE_DEFAULT,//compression
                     PNG_FILTER_TYPE_DEFAULT);//filter
        png_set_packing(png_ptr);
        png_write_info(png_ptr, info_ptr);

        /* write data */
        for(uint32_t i= 0; i < height; ++i)
            png_write_row(png_ptr, data+i*row_size);

        png_write_end(png_ptr, info_ptr);
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);

        return 0;
    }

    class Canvas{

    private:
        struct Color{
            uint8_t r,g,b;
        };
        std::vector<std::pair<std::shared_ptr<nd::NdArray<float>>, Color>> points;
        struct Line{
            float start_x;
            float start_y;
            float end_x;
            float end_y;
            Color color;
        };
        std::vector<Line> lines;
        uint16_t ppm_x=10, ppm_y=10;
        int xmin=0, xmax=0, ymin=0, ymax=0;

    public:
        // ppm: pixel per meter
        Canvas(uint16_t ppm_x, uint16_t ppm_y, int xmin, int xmax, int ymin, int ymax): ppm_x(ppm_x), ppm_y(ppm_y), xmin(xmin), xmax(xmax),ymin(ymin),ymax(ymax){

        }

        void scatter(std::shared_ptr<nd::NdArray<float>> X, Color color){
            //TODO check shape
            points.emplace_back(X,color);
        }

        void line(float start_x,float start_y, float end_x, float end_y, Color color){
            lines.push_back({start_x,start_y,end_x,end_y,color});
        }

        void save(std::string save_path){
            uint32_t canvas_height = (ymax-ymin) * ppm_y, canvas_width = (xmax-xmin) * ppm_x;
            uint32_t x_offset = canvas_width/2, y_offset = canvas_height/2;

            auto X = nd::constant<uint8_t>({canvas_height, canvas_width, 3}, 255);

            // scatter all points
            for(auto &P_colored: points){
                auto P = P_colored.first;
                auto color = P_colored.second;
                nd::shape_t N = P->shape[0]; //num points in P
                for(nd::shape_t i=0; i<N; ++i){
                    auto x = nd::slice_item(P, {i,0});
                    auto y = nd::slice_item(P, {i,1});
                    if(x<xmin || x>xmax || y<ymin || y>ymax) continue;

                    x = x*ppm_x + x_offset;
                    y = -y*ppm_y + y_offset;

                    nd::slice_item(X, {(uint32_t)y,(uint32_t)x,0}) = color.r;
                    nd::slice_item(X, {(uint32_t)y,(uint32_t)x,1}) = color.g;
                    nd::slice_item(X, {(uint32_t)y,(uint32_t)x,2}) = color.b;
                }
            }

            // draw all lines use DDA algorithm
            for(auto &L: lines) {
//                if(L.start_x<xmin || L.start_x>xmax || L.start_y<ymin)
                auto x0 = L.start_x*ppm_x + x_offset, x1 = L.end_x*ppm_x+x_offset, y0 = -L.start_y*ppm_y + y_offset, y1 = -L.end_y*ppm_y+y_offset;
                int dx = x1 - x0, dy = y1 - y0, steps;
                float xIncrement, yIncrement, x = x0, y = y0;

                if (std::abs(dx) > std::abs(dy))	//判断增长方向
                    steps = std::abs(dx);		//以X为单位间隔取样计算
                else
                    steps = std::abs(dy);		//以Y为单位间隔取样计算

                xIncrement = (float)(dx) / (float)(steps);	//计算每个度量间隔的X方向增长量
                yIncrement = (float)(dy) / (float)(steps);	//计算每个度量间隔的Y方向增长量

                while(steps--)
                {
                    uint32_t round_x = std::round(x), round_y = std::round(y);
                    if(round_x < 0 || round_x>canvas_width || round_y < 0 || round_y > canvas_height) continue;
                    nd::slice_item(X, {round_y,round_x,0}) = L.color.r;
                    nd::slice_item(X, {round_y,round_x,1}) = L.color.g;
                    nd::slice_item(X, {round_y,round_x,2}) = L.color.b;
                    x += xIncrement;
                    y += yIncrement;
                }
            }

            image_save(save_path.c_str(), X->data, canvas_height, canvas_width);
        }

    };
}