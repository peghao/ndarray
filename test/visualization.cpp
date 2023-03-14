#include <NdArray.hpp>
//#include <Image.hpp>
#include <Plot.hpp>
#include <fmt/format.h>

std::shared_ptr<nd::NdArray<float>> bbox2corner2d(std::shared_ptr<nd::NdArray<float>> dets){
    //dets will be shape of (N,7) in order x,y,z,w,h,l,heading
    nd::shape_t N = dets->shape[0];
    auto corner2d = nd::array<float>({-0.5,-0.5, 0.5,-0.5, 0.5,0.5, -0.5,0.5}, {4,2});
    auto wh = nd::reshape(nd::slice(dets, {{0,SLICE_END}, {3,5}}), {N,1,2}); //(N,1,2)
    auto xy = nd::reshape(nd::slice(dets, {{0,SLICE_END}, {0,2}}), {N,1,2}); //(N,1,2)
    auto heading = nd::slice(dets, {{0,SLICE_END},{6,7}}); //(N,1)
    corner2d = nd::times_broadcast(wh, corner2d); //(N,4,2)
    auto R = nd::getRotationMatrix2d(heading);
    corner2d = nd::matmul(R, nd::transpose(corner2d)); //(N,2,2)*(N,2,4) -> (N,2,4)
    corner2d = nd::transpose(corner2d); //(N,4,2)
    return nd::add_broadcast(corner2d, xy);
}

auto my_data_loader(int frame_id){
    std::string pc_path = fmt::format("/home/penghao/obj-tracking/SimpleTrack/kitti/testing/0000/{:06d}.bin", frame_id);
    std::string dets_path = fmt::format("/home/penghao/obj-tracking/PointPillars/output/bin/{:06d}.bin", frame_id);
    auto pc = nd::fromfile<float>(pc_path);
    auto dets = nd::fromfile<float>(dets_path);
    pc = nd::reshape(pc, {pc->shape[0]/4, 4});
    dets = nd::reshape(dets, {dets->shape[0]/8, 8});
    return std::make_pair(pc, bbox2corner2d(dets));
}

void visualize(int frame_id, std::shared_ptr<nd::NdArray<float>> pc, std::shared_ptr<nd::NdArray<float>> corner2d){
    auto canvas = plt::Canvas(30, 30, -10,50,-30,30);

    canvas.scatter(pc, {80,80, 80});

    nd::shape_t N = corner2d->shape[0];
    for(nd::shape_t i=0; i<N; ++i){
        canvas.line(nd::slice_item(corner2d,{i,0,0}), nd::slice_item(corner2d,{i,0,1}), nd::slice_item(corner2d,{i,1,0}), nd::slice_item(corner2d,{i,1,1}), {255,0,0});
        canvas.line(nd::slice_item(corner2d,{i,1,0}), nd::slice_item(corner2d,{i,1,1}), nd::slice_item(corner2d,{i,2,0}), nd::slice_item(corner2d,{i,2,1}), {255,0,0});
        canvas.line(nd::slice_item(corner2d,{i,2,0}), nd::slice_item(corner2d,{i,2,1}), nd::slice_item(corner2d,{i,3,0}), nd::slice_item(corner2d,{i,3,1}), {255,0,0});
        canvas.line(nd::slice_item(corner2d,{i,3,0}), nd::slice_item(corner2d,{i,3,1}), nd::slice_item(corner2d,{i,0,0}), nd::slice_item(corner2d,{i,0,1}), {255,0,0});
    }

    canvas.save(fmt::format("{}/{:06d}.png", "./", frame_id));
}

int main(){
    for(size_t i=0; i<100; ++i){
        auto data = my_data_loader(i);
//        std::cout << nd::to_string(data.first->get_shape()) << std::endl;
//        std::cout << nd::to_string(data.second->get_shape()) << std::endl;
//        std::cout << nd::to_string(data.second) << std::endl;

        visualize(i, data.first, data.second);
        std::cout << i << std::endl;
    }

}