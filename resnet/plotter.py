import matplotlib.pyplot as plt
import numpy as np
import warnings
import math

class Plotter(object):
    """Class to plot loss and accuracy charts (for training and validation data)."""
    def __init__(self,
                 title=None,
                 save_to_filepath=None,
                 show_loss_plot=True,
                 show_acc_plot=True,
                 show_plot_window=True,
                 x_label="Epoch"):
        """Constructs the plotter.

        Args:
            title: An optional title which will be shown at the top of the
                plot. E.g. the name of the experiment or some info about it.
                If set to None, no title will be shown. (Default is None.)
            save_to_filepath: The path to a file in which the plot will be saved,
                e.g. "/tmp/last_plot.png". If set to None, the chart will not be
                saved to a file. (Default is None.)
            show_averages: Whether to plot moving averages in the charts for
                each line (so for loss train, loss val, ...). This value may
                only be True or False. To change the interval (default is 20
                epochs), change the instance variable "averages_period" to the new
                integer value. (Default is True.)
            show_loss_plot: Whether to show the chart for the loss values. If
                set to False, only the accuracy chart will be shown. (Default
                is True.)
            show_acc_plot: Whether to show the chart for the accuracy value. If
                set to False, only the loss chart will be shown. (Default is True.)
            show_plot_window: Whether to show the plot in a window (True)
                or hide it (False). Hiding it makes only sense if you
                set save_to_filepath. (Default is True.)
            x_label: Label on the x-axes of the charts. Reasonable choices
                would be: "Epoch", "Batch" or "Example". (Default is "Epoch".)
        """
        assert show_loss_plot or show_acc_plot
        assert save_to_filepath is not None or show_plot_window

        self.title = title
        self.title_fontsize = 17
        self.show_loss_plot = show_loss_plot
        self.show_acc_plot = show_acc_plot
        self.show_plot_window = show_plot_window
        self.save_to_filepath = save_to_filepath
        self.x_label = x_label

        # whether to show grids in both charts
        self.grid = True

        # the colors of the lines, tuple(x / 255 for x in colors)
        # 0 = main exit, 1 = 1st exit, 2 = 2nd exit
        # sma = simple moving average
        self.colors = {
            "train_0": (31, 119, 180),
            "test_0": (174, 199, 232),
            "train_1": (44, 160, 44),
            "test_1": (152, 223, 138),
            "train_2": (214, 39, 40),
            "test_2": (255, 152, 150),
        }
        
        # these values will be set in _initialize_plot() upon the first call
        # of redraw()
        # fig: the figure of the whole plot
        # ax_loss: loss chart (left)
        # ax_acc: accuracy chart (right)
        self.fig = None
        self.ax_loss = None
        self.ax_acc = None 

    def plot_values(self, loss_train=None, loss_test=None, acc_train=None,
                   acc_test=None):
        """
        Args:
            loss_train: 
            loss_val: 
            acc_train:
            acc_val: 
                            number of exits x epoches
        """
        index = np.arange(loss_train.shape[1])
        num = loss_train.shape[0] # how many exits the result has
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(24, 8))
        self.fig = fig
        
        for ax, label in zip([ax1, ax2], ["Loss", "Accuracy"]):
            if ax:
                ax.clear()
                ax.set_title(label)
                ax.set_ylabel(label)
                ax.set_xlabel(self.x_label)
                ax.grid(self.grid)
        
        # plot the loss
        for i in range(num):
            color = self.colors["train_"+str(i)]
            ax1.plot(index, loss_train[i], c = tuple(x / 255 for x in color), label = "training loss of exit "+str(i))
        for i in range(num):
            color = self.colors["test_"+str(i)]
            ax1.plot(index, loss_test[i], c = tuple(x / 255 for x in color), label = "  testing loss of exit "+str(i))
        
        # plot the acc
        for i in range(num):
            color = self.colors["train_"+str(i)]
            ax2.plot(index, acc_train[i], c = tuple(x / 255 for x in color), label = "training accuracy of exit "+str(i))
        for i in range(num):
            color = self.colors["test_"+str(i)]
            ax2.plot(index, acc_test[i], c = tuple(x / 255 for x in color), label = "testing accuracy of exit "+str(i))
        
        ax1.legend(loc="upper center",
                       bbox_to_anchor=(0.5, -0.08),
                       ncol=2)
        
        ax2.legend(loc="upper center",
                       bbox_to_anchor=(0.5, -0.08),
                       ncol=2)
        
        fig.suptitle(self.title)
        plt.draw()

        # save the redrawn plot to a file upon every redraw.
        if self.save_to_filepath is not None:
            self.save_plot(self.save_to_filepath)
        
            
    def save_plot(self, filepath):
        """Saves the current plot to a file.

        Args:
            filepath: The path to the file, e.g. "/tmp/last_plot.png".
        """
        self.fig.savefig(filepath, bbox_inches="tight")

        
        
            
    