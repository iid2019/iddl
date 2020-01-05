# Final project of the team iid2019 for CS280

Deep learning

## Setup

Use `conda` to manage Python environments.

### Create a new environment

```shell
$ conda create --name iid2019_project_env
$ conda activate iid2019_project_env
```

### Install PyTorch and cudatoolkit

You should check the version of cuda installed on your system.

For cuda with version `9.x`:

```shell
$ conda install pytorch=1.3.1 torchvision cudatoolkit=9.2 -c pytorch
```

For cuda with version `10.x`:

```shell
$ conda install pytorch=1.3.1 torchvision cudatoolkit=10.1 -c pytorch
```

### Install matplotlib

```shell
$ conda install matplotlib
```

### Install jupyter-lab

```shell
$ conda install -c conda-forge jupyterlab
```

### Install thop

```shell
$ pip install thop
```

### Start jupyter-lab

```shell
$ jupyter-lab --ip 0.0.0.0 --port 35681 --config config.py
```

## Experiments data

| model | dataset | epoch | training time |
|---|---|---|---|
| mlp | mnist | 10 | 1min 46s |
| mlp_bibd | mnist | 10 | 1min 44s |
| mlp | cifar10 | 100 | 23min 25s |
| mlp_bibd | cifar10 | 100 | 20min 36s |
