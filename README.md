# Real-Time Ship Detector
A Real-time ship detector using [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) and [OpenCV](http://opencv.org/).


## Requirements
- Ubuntu 16.04
- Python 3.5
- [TensorFlow-GPU 1.9](https://www.tensorflow.org/install)
- [CUDA 9.0](https://developer.nvidia.com/cuda-downloads)
- [cuDNN 7.1.4](https://developer.nvidia.com/cudnn)
- OpenCV 3.4


## Getting Started
Creating virtualenv
```bash
$ cd RealTimeShipDetector
$ virtualenv env --python=python3.5
$ source env/bin/activate
```

Install dependencies
```bash
$ pip install -r requirements.txt
```

Run
```bash
$ python detector.py \
    --video=file_path \
    --model=model_name
```