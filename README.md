# Deep Learning Performance Characterization for Quantization Frameworks

## Abstract:
Deep learning finds applications in various fields like computer vision, natural language processing, robotics, and recommender systems. While large neural networks deliver high accuracy, they pose challenges in terms of training time, latency, energy consumption, and memory usage. To address these issues, optimization techniques and frameworks have been developed. This study evaluates the performance of quantization frameworks using metrics like training time, memory usage during training, and latency and throughput during inference on GPUs. In this paper, We have directed our attention to classification models to determine the prime factors in model architectures that affect quantization performance. The findings will help the developers and researchers create efficient deep learning models for GPUs, considering various factors like model type, dataset, image size, and batch size in both training and inference stages.

## Results:
We have found that model architecture and corresponding parameters are the major factors that affect quantization performance. VGG16 has a large number of parameters, while MobileNet_v1 and ResNet-50 reduce their parameters and operations using depthwise separable convolutional layers and 1×1 filters, resulting in lesser speedup and improvement factors after quantization as compared to VGG16.

![cifar_results](https://github.com/alishafique3/Deep_Learning_Performance_Characterization_for_Quantization_Frameworks/assets/17300597/574df405-efa9-473f-aaac-014805b590b7)

## Usage:
The code is built using NVIDIA container image of TensorFlow, release 22.03, which is available on [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow).\
The code is built using following libraries:

- Python 3.8
- NVIDIA cuDNN 8.3.3.40
- TensorFlow 2.8.0
- NVIDIA TensorRT 8.2.3
- TensorFlow-TensorRT (TF-TRT)
  
For Docker 19.03 or later, a typical command to launch the container is:
```
docker run --gpus all -it --rm nvcr.io/nvidia/tensorflow:xx.xx-tfx-py3
```
For Docker 19.02 or earlier, a typical command to launch the container is:
```
nvidia-docker run -it --rm nvcr.io/nvidia/tensorflow:xx.xx-tfx-py3
```
Where:
- xx.xx is the container version that is 22.03
- tfx is the version of TensorFlow that is tf2.

## Citation:
If this study is useful or relevant to your research, please kindly recognize our contributions by citing our paper
```
@article{shafique2023deep,
  title={Deep Learning Performance Characterization on GPUs for Various Quantization Frameworks},
  author={Shafique, Muhammad Ali and Munir, Arslan and Kong, Joonho},
  journal={AI},
  volume={4},
  number={4},
  pages={926--948},
  year={2023},
  publisher={MDPI}
}
```
