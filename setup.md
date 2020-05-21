# Setup development environment

Use `conda` to manage Python environments.

There are two ways to setup the development environment: installing dependencies manually or from environment file.

## 1. Install dependencies manually

First create a new environemnt:

```shell
$ conda create --name iddl_env python=3.7
$ conda activate iddl_env
```

### Install PyTorch and `cudatoolkit`

You should check the version of cuda installed on your system.

For cuda with version `9.x`:

```shell
$ conda install pytorch=1.5 torchvision cudatoolkit=9.x -c pytorch
```

For cuda with version `10.x`:

```shell
$ conda install pytorch=1.5 torchvision cudatoolkit=10.x -c pytorch
```

### Install `matplotlib`

```shell
$ conda install matplotlib
```

### Install `jupyterlab`

```shell
$ conda install -c conda-forge jupyterlab
```

### Install `networkx`

```shell
$ conda install networkx
```

### Install `art`

```shell
$ conda install -c sepandhaghighi art=4.6
```

### Install `thop`

> Attention: The `thop` package does not provide a conda distribution, thus this package must be installed vis pip. Pip dependencies are not listed in the conda environment file, which means that you need to install the `thop` package through pip even if you create the conda environment from the environment file.

```shell
$ pip install thop
```

### Install `halo`

```shell
$ pip install halo==0.0.29
```

## 2. From environment file

You can just install all the dependencies by one command from the environment file:

For cuda with version `9.x`:
```shell
$ conda create --name iddl_env --file iddl_env_cuda9_env.txt
```

For cuda with version `10.x`:
```shell
$ conda create --name iddl_env --file iddl_env_cuda10_env.txt
```

## 3. Start `jupyterlab`

```shell
$ jupyter-lab --ip 0.0.0.0 --port 35681 --config config.py
```
