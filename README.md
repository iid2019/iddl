# IDDL: An Integrated Distributed Deep Learning framework

Deep learning

## BIBD

Balanced Incomplete Block Design (BIBD) is the core technique in this project.

[bibd_visualization.ipynb](https://nbviewer.jupyter.org/github/DerekDick/iid2019-final-project/blob/master/bibd_visualization.ipynb)

## MLP experiments

Three models:
- MLP
- MLP with BIBD
- Random sparse MLP

You can check the live Jupyter Notebook here: [mlp_bibd_experiments.ipynb](https://nbviewer.jupyter.org/github/DerekDick/iid2019-final-project/blob/master/mlp/mlp_bibd_experiments.ipynb)

## Setup development environment

Use `conda` to manage Python environments.

There are two ways to setup the development environment: from environment file or installing dependencies manually.

### From environment file (recommended)

You can just install all the dependencies by one command from the environment file:

For cuda with version `9.x`:
```shell
$ conda create --name iddl_env --file iddl_env_cuda9_env.txt
```

For cuda with version `10.x`:
```shell
$ conda create --name iddl_env --file iddl_env_cuda10_env.txt
```

### Install dependencies manually

First create a new environemnt:

```shell
$ conda create --name iddl_env
$ conda activate iddl_env
```

#### Install PyTorch and `cudatoolkit`

You should check the version of cuda installed on your system.

For cuda with version `9.x`:

```shell
$ conda install pytorch=1.3.1 torchvision cudatoolkit=9.2 -c pytorch
```

For cuda with version `10.x`:

```shell
$ conda install pytorch=1.3.1 torchvision cudatoolkit=10.1 -c pytorch
```

#### Install `matplotlib`

```shell
$ conda install matplotlib
```

#### Install `jupyterlab`

```shell
$ conda install -c conda-forge jupyterlab
```

#### Install `thop`

```shell
$ pip install thop
```

#### Install `networkx`

```shell
$ pip install networkx
```

### Start `jupyterlab`

```shell
$ jupyter-lab --ip 0.0.0.0 --port 35681 --config config.py
```

## Experiments data record

| model | dataset | epoch | training time | accuracy | Params | FLOPs |
|---|---|---|---|---|---|---|
| mlp | mnist | 10 | 1min 55s | 96.4% | ? | 0.08M |
| mlp_bibd | mnist | 10 | 1min 49s | 96.0% | ? | 0.08M |
| mlp | cifar10 | 100 | 23min 25s | 50.7% | ? | 0.31M |
| mlp_bibd | cifar10 | 100 | 20min 36s | 48.9% | ? | 0.30M |
| resnet18 | cifar10 | 30 | (total) 35m22s | 81.440% | ? | 557.89M |
| resnet18_bibd | cifar10 | 30 | (total) 36m13s | 77.010% | ? |

Hardware info:
- GPU name: GeForce GTX 970
- GPU memory: 4GB
- GPU count: 1
- Cuda version: 10.2
- Driver version: 440.33.01
