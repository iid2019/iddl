# Models on the CIFAR10 dataset

[reference](https://github.com/kuangliu/pytorch-cifar)

All the `.py` with the key words `manual` is for the model with three exits.

In the `main.py, main_manual.py`, all the records will be saved in the name of `train/test_acc/loss.csv`. One can change that name and the folder by modifying the last few lines of them.

The `reults` folder includes the records of the experiments we ran.

FYI. EE = Early Exit, Ens = Ensemble, GC = Group Convolution.

## SparseResNetV

The `SparseResNetV` class provides a convenient and consistent method of building sparse ResNet models with vertical partioning.
