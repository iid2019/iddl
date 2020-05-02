# Models on the CIFAR10 dataset

[reference](https://github.com/kuangliu/pytorch-cifar)

All the `.py` with the key words `manual` is for the model with three exits.

In the `main.py, main_manual.py`, all the records will be saved in the name of `train/test_acc/loss.csv`. One can change that name and the folder by modifying the last few lines of them.

The `reults` folder includes the records of the experiments we ran.

FYI. EE = Early Exit, Ens = Ensemble, GC = Group Convolution.

## R-ResNet-18 on CIFAR-10

**Run the expreiment**

```shell
$ conda activate iddl_env # Make sure that you are using the correct conda environment
$ ./run_rresnet_cifar10.sh
```

**Output**
- Log file: `./log/rresnet_cifar10_${BEGIN}.log`
- Figures: `./fig_loss_rresnet_cifar10.eps` and `./fig_acc_rresnet_cifar10.eps`
