import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class CooccurrenceVisualizer:
    @staticmethod
    def plot_cooc(cooc, val_index, feat_name, log=True, figsize=(20, 15),
                  titlesize=20, ticksize=10, tickweight=400,
                  savepath=None, show=True):
        val_index_sorted = sorted(val_index.items(), key=lambda x: x[1])
        val_index_sorted = [val for val, index in val_index_sorted]
        matrix = np.log1p(cooc.toarray()) if log else cooc.toarray()

        plt.close()
        plt.figure(figsize=figsize)
        sns.heatmap(matrix,
                    cmap="YlGnBu",
                    xticklabels=val_index_sorted,
                    yticklabels=val_index_sorted)
        plt.xticks(fontsize=ticksize, fontweight=tickweight)
        plt.yticks(fontsize=ticksize, fontweight=tickweight)
        plt.title("log(Cooccurence) of feature '{}'".format(feat_name), fontsize=titlesize)

        if savepath is not None:
            plt.savefig(savepath)
        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def plot_cooc_downsample(cooc, val_index, feat_name, window, log=True, figsize=(20, 15),
                             titlesize=20, ticksize=10, tickweight=400,
                             savepath=None, show=True):
        val_index_sorted = sorted(val_index.items(), key=lambda x: x[1])
        val_index_sorted = [val for val, index in val_index_sorted]

        val_index_sample = val_index_sorted[::window]
        cooc_sample = cooc[::window, ::window]
        matrix = np.log1p(cooc_sample.toarray()) if log else cooc.toarray()

        plt.close()
        plt.figure(figsize=figsize)
        sns.heatmap(np.log1p(matrix),
                    cmap="YlGnBu",
                    xticklabels=val_index_sample,
                    yticklabels=val_index_sample)
        plt.xticks(fontsize=ticksize, fontweight=tickweight)
        plt.yticks(fontsize=ticksize, fontweight=tickweight)
        plt.title("log(Cooccurence) of feature '{}' (downsampled [windowSize={}])".format(feat_name, window),
                  fontsize=titlesize)

        if savepath is not None:
            plt.savefig(savepath)
        if show:
            plt.show()
        else:
            plt.close()
