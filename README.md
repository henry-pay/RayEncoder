<div align="center">
    <h2>Conceptualizing Multi-scale Wavelet Attention and Ray-based Encoding for Human-Object Interaction Detection (IJCNN 2025)</h2>
</div>

<p align="center">
    <a href="https://github.com/henry-pay/RayEncoder/blob/main/LICENSE" alt="license">
        <img src="https://img.shields.io/badge/license-Apache--2.0-%23B7A800" />
    </a>
</p>

We conceptualize a wavelet attention-like backbone together with ray-based encoding technique for Human-Object Interaction Detection. The proposed mechanism delivers a competitive result with better efficiency.

This repository contains the PyTorch implementation.

## Image Classification

### 1. Installation

In this section, we provide instructions for ImageNet classification experiments.

#### 1.1 Dependency Setup

Create a new conda environment
```
conda create -y -n ray-encoder python=3.12
conda activate ray-encoder
```

Install [Pytorch](https://pytorch.org/)>=2.4.0, [torchvision](https://pytorch.org/vision/stable/index.html)>= 0.19.0 following official instructions. For example:
```
conda install -y pytorch==2.4.0 torchvision==0.19.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

Clone this repo and install required packages.
```
git clone https://github.com/henry-pay/SpaRTAN.git
conda install -y timm 
conda install -y hydra-core 
conda install -y cupy pkg-config libjpeg-turbo opencv numba
pip install ffcv
```

#### 1.2 Dataset Preparation

Download the [ImageNet-1k](http://image-net.org/) classification dataset and structure the data as follows. You can extract ImageNet with this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).
```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

Place the imagenet dataset under data directory within the repository.
```
│RayEncoder/
├──data/
│  ├── imagenet/
│  │   ├──train/
│  │   ├──val/
├──src/
```

### 2. Training

We provide ImageNet-1k training commands here.

Taking Wavelet+3 Rays as an example, you can use the following command to run the experiment on a single machine (4 GPUs)
```
OMP_NUM_THREADS=8 torchrun --nproc-per-node=4 image-classification/main.py
```

- Batch size scaling. The effective batch size is equal to ``--nproc-per-node`` * ``batch_size`` (which is specified in the [dataset config](image-classification/config/dataset/imagenet.yaml)). In the provided config file, the effective batch size is ``4*512=2048``. Running on machine, we can reduce ``batch_size`` and set ``use_amp`` flag in the [config](image-classification/config/config.yaml) to avoid OOM issues while keeping the total batch size unchanged.
- OMP_NUM_THREADS is the easiest switch that can be used to accelerate computations. It determines number of threads used for OpenMP computations. Details can be found in [documentation](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html).

To train other model variants, parameters within the [config](image-classification/config) need to be changed.

## Human-Object Interaction Detection

The experiment is carried out using [FGAHOI](https://github.com/xiaomabufei/FGAHOI). Please refer to the corresponding repository for installation and dataset preparation instructions. The training experiments can be run based on the given instructions in FGAHOI by replacing the models directory with [given models](hoi-detection/models)

## License

This project is licensed under the [Apache 2.0 License](LICENSE)

## Citation

If you find this repository helpful, please consider citing:
```

```

<p align="right">(<a href="#top">back to top</a>)</p>