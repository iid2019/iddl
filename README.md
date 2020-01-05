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
