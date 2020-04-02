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
        self.show_loss_plot = show_loss_plot
        self.show_acc_plot = show_acc_plot
        self.show_plot_window = show_plot_window
        self.save_to_filepath = save_to_filepath
        self.x_label = x_label

        # whether to show grids in both charts
        self.grid = True

        # the colors of the lines, tuple(x / 255 for x in colors)
        self.colors = np.array([[0.11764705882352941, 0.5647058823529412, 1.0],[1,0,0],[0.19607843137254902, 0.803921568627451, 0.19607843137254902], [0.5411764705882353, 0.16862745098039217, 0.8862745098039215]])
        # these values will be set in _initialize_plot() upon the first call
        # of redraw()
        # fig: the figure of the whole plot
        # ax_loss: loss chart (left)
        # ax_acc: accuracy chart (right)
        self.fig = None
        self.ax_loss = None
        self.ax_acc = None 

    def plot_values(self, labels=None, loss_train=None, loss_test=None, acc_train=None,
                   acc_test=None):
        """
        Args:
            The () gives the size of each parameter.
            labels: (number of models/exits) model name or exits index; will be shown as legend;
            loss_train: (number of epoches x number of models/exits)
            loss_val: (number of epoches x number of models/exits)
            acc_train: (number of epoches x number of models/exits)
            acc_val: (number of epoches x number of models/exits)
                            
        """
        loss_train = loss_train.T
        loss_test = loss_test.T
        acc_train = acc_train.T
        acc_test = acc_test.T
        epoch = np.arange(loss_train.shape[1])
        num = loss_train.shape[0] # number of exits/models
        if num > 4: # the default color list is not enough
            for i in range(num-3):
                self.colors = np.append(self.colors, np.random.uniform(0,1,3).reshape(1,3), 0) # randomly generate the new color
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
            ax1.plot(epoch, loss_train[i], c = self.colors[i], label =labels[i] + " training loss")
        for i in range(num):
            ax1.plot(epoch, loss_test[i], c = self.colors[i], label = labels[i] + " testing loss", alpha = 0.75)
        
        # plot the acc
        for i in range(num):
            ax2.plot(epoch, acc_train[i], c = self.colors[i], label = labels[i] + " training accuracy")
        for i in range(num):
            ax2.plot(epoch, acc_test[i], c = self.colors[i], label = labels[i] + " testing accuracy", alpha = 0.75)
        
        ax1.legend(loc="upper center",
                       bbox_to_anchor=(0.5, -0.08),
                       ncol=2, fontsize = 15)
        
        ax2.legend(loc="upper center",
                       bbox_to_anchor=(0.5, -0.08),
                       ncol=2, fontsize = 15)
        
        fig.suptitle(self.title, fontsize = 23)
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

        
        
            
    