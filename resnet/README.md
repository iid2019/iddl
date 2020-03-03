# Models on the CIFAR10 dataset

[reference](https://github.com/kuangliu/pytorch-cifar)

All the `.py` with the key words `manual` is for the model with three exits.

In the `main, main1, main2, main_manual and main_manual_2.py`, all the records will be saved in the name of `train_XX.csv` and `test_XX.csv`. One can change that name by modifying the last few lines of them.

The `reults` folder includes the records of the experiments we ran. Loss and Accuracy are arranged in rows and epoches are arranged in columns. For the model with three exits, the rows are arranged by Loss of exit0, Accuracy of exit0, Loss of exit1, Accuracy of exit1, Loss of exit2, Accuracy of exit2.

FYI. EE = Early Exit, Ens = Ensemble, GC = Group Convolution.