# deepstream-python yolov5
This is a simple app build on the top of deepstream-test1 using custom tensorrt yolov5.

[中文](https://blog.csdn.net/weixin_42202176/article/details/124604138?spm=1001.2014.3001.5502)
## Requirements
+ Deepstream 6.0
+ GStreamer 1.14.5
+ Cuda 11.4+
+ NVIDIA driver 470.63.01+
+ TensorRT 8+

Follow [deepstream](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html#dgpu-setup-for-ubuntu) official doc to install dependencies.

Deepstream docker is more recommended.
## Pretrained
Please refer to this [repo](https://github.com/wang-xinyu/tensorrtx) for pretrained models and serialized TensorRT engine.

## Installation
```
git clone https://github.com/zhouyuchong/yolov5-deepstream-python
cd yolov4-deepstream-python
make nvdsinfer_custom_impl_Yolo 
```
check all paths in `deepstream_yolov5_config.txt` and `main.py`. make sure they are correct.

## Usage
```
python3 main.py {VideoPath}
```
this app only supports **h264** format file.