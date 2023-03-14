pointcloud.bin:
    一个街道场景的点云文件，用于测试scatter函数

detections.bin
    pointcloud.bin的目标识别结果，包含11个物体的边界框，用于测试drawline函数

simple_array.bin
    一个数组的二进制文件，用于测试nd::fromfile函数。该数组用如下python代码生成：
    import numpy as np
    np.array([1,2,3,4,5,6], dtype=np.float32).tofile("simple_array.bin")
