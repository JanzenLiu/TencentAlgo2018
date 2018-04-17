import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap  # for type annotation

current_dir = os.path.dirname(os.path.abspath(__file__))
figure_folder_path = os.path.join(current_dir, '..', 'figures')

def get_cmap(n, name='hsv') -> Colormap:
    """Construct a `Colormap` instance that maps integer indices to distinct colors

    Adapted from https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib

    Parameters
    ----------
    n: int
        The number of entries desired to put in the `Colormap` instance

    name: string
        The standard mpl colormap name to construct the `Colormap` instance with

    Returns
    -------
    cm: Colormap
        Colormap instance that maps each index in 0, 1, ..., n-1 to a distinct
        RGB color.

    Examples
    --------
    >>> cm = get_cmap(10)
    >>> cm(5)  # a tuple representing an RGBA color
    (0.0, 0.718752625002625, 1.0, 1.0)
    """
    cm = plt.cm.get_cmap(name, n)
    return cm


class SingleFeatureVisualizer:
    """Helper to plot single feature's distribution.

    Parameters
    ----------
    dfs: dict
        Dictionary containing DataFrames as values and their names (which is
        given by you, it can be whatever thing as long as it's a string) as
        keys.

    Examples
    --------
    >>> df_train = pd.read_csv('train.csv')
    >>> df_test = pd.read_csv('test.csv')
    >>> sfv = SingleFeatureVisualizer({"train": df_train, "test":df_test})
    """
    def __init__(self, dfs):
        self.dfs = dfs

    @staticmethod
    def plot_numerical_feat(x, name, ax=None, bins=None, savepath=None, **kw):
        """Plot histogram for a single array representing a numerical feature

        Parameters
        ----------
        x: array-like
            Values of the feature. Free or not free from NaN are both acceptable.

        name: string
            Feature name to appear on the plot title.

        ax: plt.Axes
            `Axes` instance to plot the histogram on. If None, the histogram will
            be plotted on plt directly. Please set this with a valid `Axes` if you
            expect to plot the histogram in a subplot as a part of a larger
            figure.

        bins: int | array-like
            If int, it specifies the number of bins to divide the numerical values
            into. If array-like, it represents the bin edges to use to divide the
            numerical values.

        savepath: string
            A string containing a path to an output filename under the folder /code/figures.

        Examples
        --------
        >>> df = pd.read_csv('train.csv')
        >>> col = 'SalePrice'  # column to plot, which is supposed to be numerical
        >>> SingleFeatureVisualizer.plot_numerical_feat(df[col], col, savepath='test.png')
        """
        x_clean = x[~np.isnan(x)]  # filter out NaN values, which are supposed to be considered as well. To fix it
        nunique = len(np.unique(x_clean))
        bins = bins if bins is not None else 100 if nunique > 30 else 10 if nunique < 10 else 'auto'
        title = "{} Distribution".format(name)
        if ax is not None:
            ax.hist(x_clean, bins=bins, **kw)  # if ax is given, plot in it as a subplot
            ax.set_title(title)
        else:
            plt.hist(x_clean, bins=bins, **kw)  # if ax is not given, plot in plt globally
            plt.title(title)
            if savepath is not None:
                plt.savefig(os.path.join(figure_folder_path, savepath))

    @staticmethod
    def plot_numerical_feats(df, num_cols=None, ncols=3, height_per_plot=6, savepath=None):
        """Plot histograms for specified or simply all numerical features in a DataFrame

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame whose numerical features you are going to plot.

        num_cols: array-like
            Names of the numerical features to plot. If None, all features stored as
            numerical primitive (i.e. float and int) in the memory will be plotted.

        ncols: int
            Number of subplots in a row, default 3.

        height_per_plot: int
            Height of a subplot in the figure, default 6.

        savepath: string
            A string containing a path to an output filename under the folder /code/figures.

        Examples
        --------
        >>> df = pd.read_csv("train.csv")
        >>> SingleFeatureVisualizer.plot_numerical_feats(df, savepath='test.png')
        """
        # df_name = get_var_name(df)  # problem of get_var_name is not fixed so far, so just comment it
        # print(df_name)
        # print("-" * len(df_name))
        if num_cols is None:
            num_cols = df.dtypes[df.dtypes != object].index.tolist()
        num_col_counts = len(num_cols)
        nrows = int(np.ceil(num_col_counts / 3))
        fig, axs = plt.subplots(nrows, ncols, figsize=(20, nrows*height_per_plot))
        for i, feat in enumerate(num_cols):
            ix = i // ncols
            iy = i % ncols
            ax = axs[ix][iy] if nrows > 1 else axs[iy]  # axs is an 1-D array if nrows is 1, so the access is different
            SingleFeatureVisualizer.plot_numerical_feat(df[feat], feat, ax)
        for i in range(num_col_counts, nrows*ncols):
            fig.delaxes(axs.flatten()[i])  # delete unused subplots
        if savepath is not None:
            plt.savefig(os.path.join(figure_folder_path, savepath))

    @staticmethod
    def plot_numerical_feats_double(dfs, num_cols=None, ncols=3, height_per_plot=6):
        """Plot histograms for specified or simply all numerical features in two given DataFrames

        Parameters
        ----------
        dfs: list
            List containing two tuples, each has a DataFrame as its second entry, and the name of
            the DataFrame as its first entry. See the example below.

        num_cols: array-like
            Names of the numerical features to plot. If None, all features stored as
            numerical primitive (i.e. float and int) in the memory will be plotted.

        ncols: int
            Number of subplots in a row, default 3.

        height_per_plot: int
            Height of a subplot in the figure, default 6.

        Examples
        --------
        >>> df_train = pd.read_csv('train.csv')
        >>> df_test = pd.read_csv('test.csv')
        >>> dfs = [('train', df_train), ('test', df_test)]  # two elements indicating two DataFrames and their names
        >>> SingleFeatureVisualizer.plot_numerical_feats_double(dfs)
        """
        df1_name, df1 = dfs[0]
        df2_name, df2 = dfs[1]
        df_names = "{} and {}".format(df1_name, df2_name)
        print(df_names)
        print("-" * len(df_names))
        if num_cols is None:
            num_cols1 = df1.dtypes[df1.dtypes != object].index.tolist()
            num_cols2 = df2.dtypes[df2.dtypes != object].index.tolist()
            num_cols = num_cols1 and num_cols2
        num_col_counts = len(num_cols)
        nrows = int(np.ceil(num_col_counts / 3))
        fig, axs = plt.subplots(nrows, ncols, figsize=(20, nrows*height_per_plot))
        for i, feat in enumerate(num_cols):
            ix = i // ncols
            iy = i % ncols
            ax = axs[ix][iy] if nrows > 1 else axs[iy]
            if feat not in df2.columns:
                SingleFeatureVisualizer.plot_numerical_feat(df1[feat], feat, ax, alpha=.5, density=True)
            elif feat not in df1.columns:
                SingleFeatureVisualizer.plot_numerical_feat(df2[feat], feat, ax, alpha=.5, density=True)
            else:  # condition that feature appears in both DataFrame
                vals = np.concatenate([df1[feat].values, df2[feat].values])
                nunique = len(np.unique(vals))
                nbins = 100 if nunique > 30 else 10 if nunique < 10 else min(nunique, 20)
                _, bins = pd.cut(vals, bins=nbins, retbins=True)  # get bin edges for the histogram
                SingleFeatureVisualizer.plot_numerical_feat(df1[feat], feat, ax, bins, alpha=.5, label=df1_name, density=True)
                SingleFeatureVisualizer.plot_numerical_feat(df2[feat], feat, ax, bins, alpha=.5, label=df2_name, density=True)
                ax.legend([df1_name, df2_name])
            ax.legend([df1_name, df2_name])
        for i in range(num_col_counts, nrows*ncols):
            fig.delaxes(axs.flatten()[i])
        plt.show()
        print()

    @staticmethod
    def plot_categorical_feat(x, name, ax=None, keys=None, density=True, plot_nan=True, shift=None, **kw):
        """Plot barplot for a single array representing a categorical feature

        Parameters
        ----------
        x: array-like
            Values of the feature. Free or not free from NaN are both acceptable.

        name: string
            Feature name to appear on the plot title.

        ax: plt.Axes
            `Axes` instance to plot the histogram on. If None, the histogram will
            be plotted on plt directly. Please set this with a valid `Axes` if you
            expect to plot the histogram in a subplot as a part of a larger
            figure.

        keys: array-like
            Categorical values to be considered in the barplot. If None, all values
            will be considered.

        density: boolean
            If True, the xticks will be set to the density of each categorical value,
            instead of the absolute counts, default True.

        plot_nan: boolean
            If True, NaN will be also considered as a categorical values and will get
            plotted in the barplot. default True.

        shift: float
            Horizontal shift of bars to their default positions. This should be set within
            the range [0, 1).

        Examples
        --------
        >>> df = pd.read_csv('train.csv')
        >>> col = 'MSSubClass'  # column to plot, which is supposed to be categorical
        >>> SingleFeatureVisualizer.plot_categorical_feat(df[col], col)
        """
        nan_idx = pd.isnull(pd.Series(x))
        x_clean = x[~nan_idx]  # filter out NaN values
        if keys is None:
            vals = x_clean.value_counts()  # category name-count map
            keys = vals.index  # category names
            counts = vals.values  # category counts
        else:
            vals = x_clean.value_counts()  # category name-count map
            counts = np.array([vals.get(k, 0) for k in keys])  # category counts
        if plot_nan:  # add counts for Nan values
            keys = ['[NaN]'] + list(keys)
            counts = np.array([np.sum(nan_idx)] + list(counts))
        keys = [i + shift for i in range(len(keys))] if shift is not None else keys  # x coordinates if shift is needed
        counts = counts / len(x) if density else counts  # density instead of absolute count if needed
        title = "{} Distribution".format(name)
        if ax is not None:
            ax.bar(keys, counts, **kw)  # if ax is given, plot in it as a subplot
            ax.set_title(title)
        else:
            plt.bar(keys, counts, **kw)  # if ax is not given, plot in plt globally
            plt.title(title)
            plt.show()

    @staticmethod
    def plot_categorical_feats(df, cat_cols=None, ncols=3, height_per_plot=6):
        """Plot barplots for specified or simply all categorical features in a DataFrame

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame whose numerical features you are going to plot.

        cat_cols: array-like
            Names of the categorical features to plot. If None, all features stored as
            object type in the memory will be plotted.

        ncols: int
            Number of subplots in a row, default 3.

        height_per_plot: int
            Height of a subplot in the figure, default 6.

        Examples
        --------
        >>> df = pd.read_csv("train.csv")
        >>> SingleFeatureVisualizer.plot_categorical_feats(df)
        """
        # df_name = get_var_name(df)  # problem of get_var_name is not fixed so far, so just comment it
        # print(df_name)
        # print("-" * len(df_name))
        if cat_cols is None:
            cat_cols = df.dtypes[df.dtypes == object].index.tolist()
        num_col_counts = len(cat_cols)
        nrows = int(np.ceil(num_col_counts / 3))
        fig, axs = plt.subplots(nrows, ncols, figsize=(20, nrows * height_per_plot))
        for i, feat in enumerate(cat_cols):
            ix = i // ncols
            iy = i % ncols
            SingleFeatureVisualizer.plot_categorical_feat(df[feat], feat, axs[ix][iy])
        for i in range(num_col_counts, nrows * ncols):
            fig.delaxes(axs.flatten()[i])
        plt.show()
        print()

    @staticmethod
    def plot_categorical_feats_double(dfs, cat_cols=None, ncols=3, height_per_plot=6):
        """Plot barplots for specified or simply all categorical features in two given DataFrames

        Parameters
        ----------
        dfs: list
            List containing two tuples, each has a DataFrame as its second entry, and the name of
            the DataFrame as its first entry. See the example below.

        cat_cols: array-like
            Names of the categorical features to plot. If None, all features stored as
            object type in the memory will be plotted.

        ncols: int
            Number of subplots in a row, default 3.

        height_per_plot: int
            Height of a subplot in the figure, default 6.

        Examples
        --------
        >>> df_train = pd.read_csv('train.csv')
        >>> df_test = pd.read_csv('test.csv')
        >>> dfs = [('train', df_train), ('test', df_test)]  # two elements indicating two DataFrames and their names
        >>> SingleFeatureVisualizer.plot_categorical_feats_double(dfs)
        """
        df1_name, df1 = dfs[0]
        df2_name, df2 = dfs[1]
        df_names = "{} and {}".format(df1_name, df2_name)
        print(df_names)
        print("-" * len(df_names))
        if cat_cols is None:
            cat_cols1 = df1.dtypes[df1.dtypes == object].index.tolist()
            cat_cols2 = df2.dtypes[df2.dtypes == object].index.tolist()
            cat_cols = cat_cols1 and cat_cols2
        num_col_counts = len(cat_cols)
        nrows = int(np.ceil(num_col_counts / 3))
        fig, axs = plt.subplots(nrows, ncols, figsize=(20, nrows * height_per_plot))
        for i, feat in enumerate(cat_cols):
            ix = i // ncols
            iy = i % ncols
            ax = axs[ix][iy] if nrows > 1 else axs[iy]
            if feat not in df2.columns:
                SingleFeatureVisualizer.plot_categorical_feat(df1[feat], feat, ax, alpha=.5, density=True)
            elif feat not in df1.columns:
                SingleFeatureVisualizer.plot_categorical_feat(df2[feat], feat, ax, alpha=.5, density=True)
            else:  # feature appears in both DataFrame
                vals = np.concatenate([df1[feat].values, df2[feat].values])
                keys = pd.Series(vals).value_counts().index.tolist()
                nkeys = len(keys)
                cmap = get_cmap(max(nkeys + 1, 8))
                colors = [cmap(n) for n in range(nkeys + 1)]
                SingleFeatureVisualizer.plot_categorical_feat(df1[feat], feat, ax, keys,
                                                              color=colors, edgecolor='black', width=.4,
                                                              label=df1_name)
                SingleFeatureVisualizer.plot_categorical_feat(df2[feat], feat, ax, keys,
                                                              color=colors, edgecolor='black', shift=.4, width=.4,
                                                              label=df2_name)
                ax.legend([df1_name, df2_name])
            ax.legend([df1_name, df2_name])
        for i in range(num_col_counts, nrows * ncols):
            fig.delaxes(axs.flatten()[i])
        plt.show()
        print()

    def num_hist(self, df_name, col_name, ax=None, bins=None, **kw):
        """Plot histogram for a single array representing a numerical feature

        Parameters
        ----------
        df_name: string
            Name of the DataFrame whose feature you are going to plot. This is
            supposed to be one of the keys of `self.dfs`

        col_name: string
            Name of the feature to plot. This is supposed to be one of the column
            names of the DataFrame (`self.dfs[df_name]`)

        ax: plt.Axes
            `Axes` instance to plot the histogram on. If None, the histogram will
            be plotted on plt directly. Please set this with a valid `Axes` if you
            expect to plot the histogram in a subplot as a part of a larger
            figure.

        bins: int | array-like
            If int, it specifies the number of bins to divide the numerical values
            into. If array-like, it represents the bin edges to use to divide the
            numerical values.

        Examples
        --------
        >>> df = pd.read_csv('train.csv')
        >>> col = 'SalePrice'  # column to plot, which is supposed to be numerical
        >>> sfv = SingleFeatureVisualizer({"train": df})
        >>> sfv.num_hist("train", col)
        """
        df_name = list(self.dfs.keys())[0] if df_name is None else df_name
        df = self.dfs[df_name]
        x = df[col_name]
        SingleFeatureVisualizer.plot_numerical_feat(x, col_name, ax, bins, **kw)

    def num_hists(self, df_name, num_cols=None, ncols=3, height_per_plot=6):
        """Plot histograms for specified or simply all numerical features in a DataFrame

        Parameters
        ----------
        df_name: string
            Name of the DataFrame whose feature you are going to plot. This is
            supposed to be one of the keys of `self.dfs`

        num_cols: array-like
            Names of the numerical features to plot. If None, all features stored as
            numerical primitive (i.e. float and int) in the memory will be plotted.

        ncols: int
            Number of subplots in a row, default 3.

        height_per_plot: int
            Height of a subplot in the figure, default 6.

        Examples
        --------
        >>> df = pd.read_csv('train.csv')
        >>> sfv = SingleFeatureVisualizer({"train": df})
        >>> sfv.num_hists("train")
        """
        df = self.dfs[df_name]
        SingleFeatureVisualizer.plot_numerical_feats(df, num_cols, ncols, height_per_plot)

    def num_hists_double(self, df_names, num_cols=None, ncols=3, height_per_plot=6):
        """Plot histograms for specified or simply all numerical features in two given DataFrames

        Parameters
        ----------
        df_names: list
            List containing names of the DataFrame whose features you are going to plot.
            They are supposed to be keys of `self.dfs`

        num_cols: array-like
            Names of the numerical features to plot. If None, all features stored as
            numerical primitive (i.e. float and int) in the memory will be plotted.

        ncols: int
            Number of subplots in a row, default 3.

        height_per_plot: int
            Height of a subplot in the figure, default 6.

        Examples
        --------
        >>> df_train = pd.read_csv('train.csv')
        >>> df_test = pd.read_csv('test.csv')
        >>> sfv = SingleFeatureVisualizer({"train": df_train, "test":df_test})
        >>> sfv.num_hists_double(["train", "test"])
        """
        dfs = [(df_name, self.dfs[df_name]) for df_name in df_names]
        SingleFeatureVisualizer.plot_numerical_feats_double(dfs, num_cols, ncols, height_per_plot)

    def cat_bar(self, df_name, col_name, ax=None, keys=None, density=True, plot_nan=True, shift=None, **kw):
        """Plot histogram for a single array representing a numerical feature

        Parameters
        ----------
        df_name: string
            Name of the DataFrame whose feature you are going to plot. This is
            supposed to be one of the keys of `self.dfs`

        col_name: string
            Name of the feature to plot. This is supposed to be one of the column
            names of the DataFrame (`self.dfs[df_name]`)

        ax: plt.Axes
            `Axes` instance to plot the histogram on. If None, the histogram will
            be plotted on plt directly. Please set this with a valid `Axes` if you
            expect to plot the histogram in a subplot as a part of a larger
            figure.

        keys: array-like
            Categorical values to be considered in the barplot. If None, all values
            will be considered.

        density: boolean
            If True, the xticks will be set to the density of each categorical value,
            instead of the absolute counts, default True.

        plot_nan: boolean
            If True, NaN will be also considered as a categorical values and will get
            plotted in the barplot. default True.

        shift: float
            Horizontal shift of bars to their default positions. This should be set within
            the range [0, 1).

        Examples
        --------
        >>> df = pd.read_csv('train.csv')
        >>> col = 'SalePrice'  # column to plot, which is supposed to be numerical
        >>> sfv = SingleFeatureVisualizer({"train": df})
        >>> sfv.cat_bar("train", col)
        """
        df_name = list(self.dfs.keys())[0] if df_name is None else df_name
        df = self.dfs[df_name]
        x = df[col_name]
        SingleFeatureVisualizer.plot_categorical_feat(x, col_name, ax, keys, density, plot_nan, shift, **kw)

    def cat_bars(self, df_name, cat_cols=None, ncols=3, height_per_plot=6):
        """Plot histograms for specified or simply all numerical features in a DataFrame

        Parameters
        ----------
        df_name: string
            Name of the DataFrame whose feature you are going to plot. This is
            supposed to be one of the keys of `self.dfs`

        cat_cols: array-like
            Names of the numerical features to plot. If None, all features stored as
            object type in the memory will be plotted.

        ncols: int
            Number of subplots in a row, default 3.

        height_per_plot: int
            Height of a subplot in the figure, default 6.

        Examples
        --------
        >>> df = pd.read_csv('train.csv')
        >>> sfv = SingleFeatureVisualizer({"train": df})
        >>> sfv.cat_bars("train")
        """
        df = self.dfs[df_name]
        SingleFeatureVisualizer.plot_categorical_feats(df, cat_cols, ncols, height_per_plot)

    def cat_bars_double(self, df_names, cat_cols=None, ncols=3, height_per_plot=6):
        """Plot histograms for specified or simply all numerical features in two given DataFrames

        Parameters
        ----------
        df_names: list
            List containing names of the DataFrame whose features you are going to plot.
            They are supposed to be keys of `self.dfs`

        cat_cols: array-like
            Names of the numerical features to plot. If None, all features stored as
            object type in the memory will be plotted.

        ncols: int
            Number of subplots in a row, default 3.

        height_per_plot: int
            Height of a subplot in the figure, default 6.

        Examples
        --------
        >>> df_train = pd.read_csv('train.csv')
        >>> df_test = pd.read_csv('test.csv')
        >>> sfv = SingleFeatureVisualizer({"train": df_train, "test":df_test})
        >>> sfv.cat_bars_double(["train", "test"])
        """
        dfs = [(df_name, self.dfs[df_name]) for df_name in df_names]
        SingleFeatureVisualizer.plot_categorical_feats_double(dfs, cat_cols, ncols, height_per_plot)
