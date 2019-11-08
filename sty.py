import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from matplotlib import cm
import matplotlib.ticker
from matplotlib.ticker import FormatStrFormatter
import warnings
from IPython import get_ipython
import scipy.stats as ss

import mystyle.ana as ana


def reset_plots():
    """
    Makes axes large, and enables LaTeX for matplotlib plots
    """
    plt.close('all')
    fontsize = 20
    legsize = 15
    plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    plt.rc('text', usetex=True)
    font = {'size': fontsize}
    plt.rc('font', **font)
    rc = {'axes.labelsize': fontsize,
          'font.size': fontsize,
          'axes.titlesize': fontsize,
          'xtick.labelsize': fontsize,
          'ytick.labelsize': fontsize,
          'legend.fontsize': legsize}
    mpl.rcParams.update(**rc)
    mpl.rc('lines', markersize=10)
    plt.rcParams.update({'axes.labelsize': fontsize})
    mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}',
                                           r'\usepackage{amsfonts}']

def plot(nrows=1, ncols=1, figsize=5):
    """
    Generate a matplotlib plot and axis handle

    Parameters
    -----------------
    nrows : An int, number of rows for subplotting
    ncols : An int, number of columns for subplotting
    figsize : Numeric or array (xfigsize, yfigsize). The size of each axis.
    """
    if isinstance(figsize,(list, tuple)):
        xfigsize, yfigsize = figsize
    if isinstance(figsize,(int,float)):
        xfigsize = yfigsize = figsize
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*xfigsize, nrows*yfigsize))
    if nrows*ncols > 1:
        axs = axs.ravel()
    return fig, axs


def remove_tex_axis(ax, xtick_fmt='%d', ytick_fmt='%d', axis_remove='both'):
    """
    Makes axes normal font in matplotlib.

    Parameters
    ---------------
    xtick_fmt : A string, defining the format of the x-axis
    ytick_fmt : A string, defining the format of the y-axis
    axis_remove : A string, which axis to remove. ['x', 'y', 'both']
    """
    if axis_remove not in ['x','y','both']:
        raise Exception('axis_remove value not allowed.')
    fmt = matplotlib.ticker.StrMethodFormatter("{x}")

    if axis_remove == 'both':
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)
        ax.xaxis.set_major_formatter(FormatStrFormatter(xtick_fmt))
        ax.yaxis.set_major_formatter(FormatStrFormatter(ytick_fmt))
    elif axis_remove == 'x':
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_major_formatter(FormatStrFormatter(xtick_fmt))
    else:
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(FormatStrFormatter(ytick_fmt))


def simpleaxis(ax):
    """
    Remove top and right spines from a plot

    Parameters
    ---------------
    ax : A matplotlib axis
    """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


def ignore_warnings_all():
    """
    Switch off all user warnings
    """
    warnings.simplefilter("ignore")


def update_functions_on_fly():
    """
    Let python functions be updated whilst inside an iPython/Jupyter session
    """
    ipython = get_ipython()
    ipython.magic("reload_ext autoreload")
    ipython.magic("autoreload 2")


def legend_outside(ax, pointers=None, labels=None, size=15, frameon=True):
    """
    Put legend outside the plot area

    Parameters
    ---------------
    ax : A matplotlib axis
    """

    if pointers is None and labels is None:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                    prop={'size': size}, frameon=frameon)
    else:
        assert len(pointers) == len(labels)
        ax.legend(pointers, labels, loc='center left', bbox_to_anchor=(1, 0.5),
                    prop={'size': size}, frameon=frameon)


####################################################################
# Useful plots
####################################################################


def make_heatmap(matrix, xlabel, ylabel, zlabel, xticklabels, yticklabels, plotextensions=('svg', 'png'),
                 colorbar_heatmap=cm.jet, figname=None, vmin=None, vmax=None, out_dir=os.getcwd()):
    """
    Make a heatmap of a matrix where rows are plotted on the vertical axis, and columns plotted along the horizontal

    Parameters
    ---------------
    matrix : A numpy matrix, intensities for heatmap. Expect a square matrix of dimension (n_points x n_points)
    zlabel : A string, colorbar label of heatmap
    figname : A string, prefix to figure name. If none, do not write to file
    xlabel : A string, x-label of heatmap
    ylabel : A string, y-label of heatmap
    xticklabels : A list of strings, x tick labels
    yticklabels : A list of strings, y tick labels
    vmin : A float, minimum value for colorbar
    vmax : A float, maximum value for colorbar
    plotextensions :  A list of strings, default extensions for plotting
    colorbar_heatmap : The colormap for the heatmap
    out_dir : A string, the output directory for the plot
    """
    plt.close('all')
    nan_col = 0.4
    nan_rgb = nan_col * np.ones(3)
    colorbar_heatmap.set_bad(nan_rgb)

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    im = ax.imshow(np.flipud(matrix), cmap=colorbar_heatmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=zlabel)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(xticklabels)))
    ax.set_yticks(np.arange(len(yticklabels)))
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels[::-1])

    if figname is not None:
        for p in plotextensions:
            plt.savefig(out_dir + '/' + figname + '.' + p, bbox_inches='tight')


def get_non_null_and_jitter(data, name, dx, offset):
    """
    Strip out null values from a dataframe and return jittered x with y-values

    Parameters
    ---------------
    data : A pandas dataframe, contains numeric values in the column `name`
    name : A string, the name of the column to be plotted
    dx : A float, width of the jitter
    offset : A float, offset of the jitter

    Returns
    ----------
    datax : A pandas series, containing non-null values of the column to plot
    x : A numpy array, jittered x-values
    """
    datax = data[name]
    datax = datax[~datax.isnull()]
    x = np.ones(len(datax)) + offset + np.random.uniform(-dx, dx, size=len(datax))
    return datax, x


def make_jitter_plots(data, names, ylabel, dx=0.1, offset=0.0, ytick_fmt=None,
                      xlabels=None, ax_handle=None, alpha=1, color=None,
                      marker=None, markersize=12, return_plot_pointer=False):
    """
    Make a jitter plot of columns from a pandas dataframe

    Parameters
    ---------------
    data : A pandas dataframe, contains numeric values in the columns `names`
    names: An array of strings, the name of the columns to be plotted
    dx : A float, width of the jitter
    offset : A float, offset of the jitter
    ylabel : A string, the y-label for the plot
    ax_handle : A matplotlib axis handle. When defined, the function will add a jitter plot to an ax object
    xlabels : A list of strings, the names along the x-axis
    alpha : A float, transparency on data points
    color : A string or a list of strings, the color of the points
    marker : A string or a list of strings, the marker of the points
    ytick_fmt : A string, the format of the y-ticks
    markersize : An int, the marker size

    Returns
    --------
    fig : A matplotlib figure handle (only if ax_handle is None)
    ax : A matplotlib axis handle (only if ax_handle is None)
    p[0] : A matplotlib.lines.Line2D object corresponding to the marker used in
            the jitter plot (only if return_plot_pointer is True)
    """

    if isinstance(marker, (list, tuple)):
        assert len(marker) == len(names)
    if isinstance(color, (list, tuple)):
        assert len(color) == len(names)

    yx_tuples = []
    for name in names:
        yx_tuples.append(get_non_null_and_jitter(data, name, dx, offset))

    if ax_handle is None:
        fig, ax = plt.subplots(1, 1)
    else:
        ax = ax_handle

    for i in range(len(names)):
        if isinstance(marker, (list, tuple)):
            marker_i = marker[i]
        elif isinstance(marker, str):
            marker_i = marker
        else:
            marker_i = '.'
        if isinstance(color, (list, tuple)):
            color_i = color[i]
        elif isinstance(color, str):
            color_i = color
        else:
            color_i = 'k'

        yi = yx_tuples[i][0]
        xi = yx_tuples[i][1]
        p=ax.plot(i + xi, yi, marker_i, color=color_i, alpha=alpha,
                markersize=markersize, markeredgecolor='k')

    if ytick_fmt is not None:
        remove_tex_axis(ax, ytick_fmt=ytick_fmt)

    ax.set_xticks(1 + np.arange(len(names)))
    if xlabels is None:
        names = [s.replace('_', '-') for s in names]  # underscores make LaTeX unhappy
        ax.set_xticklabels(names)
    else:
        ax.set_xticklabels(xlabels)
    for t in ax.get_xticklabels():
        t.set_rotation(90)
    ax.set_ylabel(ylabel)

    if ax_handle is None:
        if return_plot_pointer:
            return fig, ax, p[0]
        else:
            return fig, ax

    if return_plot_pointer:
        return p[0]


def plot_w_m_lfc(w, m, q_low=2.5, q_high=97.5, ax_handle=None, B=100):
    """
    Plot data in the (w,m) plane and fit a linear feedback control

    Parameters
    --------------
    w : A numpy array, data for wild-type copy number
    m : A numpy array, data for mutant copy number
    q_low : A float, lower quantile on the steady state line fit
    q_high : A float, upper quantile on the steady state line fit
    ax_handle : A matplotlib axis handle, for adding onto an existing plot
    B : Number of bootstrap iterations

    Returns
    -------------
    fig : A matplotlib figure handle (if ax_handle is None)
    ax : A matplotlib axis handle (if ax_handle is None)
    summary_stats : A list containing the variables [deltas, kappas, delta_ml, kappa_ml], see mystyle.ana.bootstrap_lfc

    """
    slopes, intercepts, slope_ml, intercept_ml = ana.bootstrap_2d_pca(w, m, B)
    deltas = -1.0 / slopes
    kappas = intercepts * deltas

    delta_ml = -1.0 / slope_ml
    kappa_ml = intercept_ml * delta_ml

    summary_stats = [deltas, kappas, delta_ml, kappa_ml]

    # Quantiles
    w_sp = np.linspace(min(w) * 0.9, max(w) * 1.1, num=200)

    m_fits = np.zeros([B, len(w_sp)])
    for i in range(B):
        m_fits[i, :] = -w_sp / deltas[i] + kappas[i] / deltas[i]

    ql = np.percentile(m_fits, q_low, axis=0)
    qh = np.percentile(m_fits, q_high, axis=0)

    if ax_handle is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    else:
        ax = ax_handle

    ax.plot(w_sp, -w_sp / delta_ml + kappa_ml / delta_ml, '-r', label='PCA (ML)')
    ax.fill_between(w_sp, ql, qh, color='red', alpha=0.5, label='95\% Boot. C.I.')

    ax.legend()
    ax.set_xlabel('Wild-type copy number, $w$')
    ax.set_ylabel('Mutant copy number, $m$')

    if ax_handle is None:
        return fig, ax, summary_stats
    else:
        return summary_stats

def plot_2d_pca_bootstrap(x, y, q_low=2.5, q_high=97.5, ax_handle=None, B=1000):
    """
    Plot x-y and bootstrap PCA

    Parameters
    --------------
    x : A numpy array, x-values
    y : A numpy array, x-values
    q_low : A float, lower quantile on the steady state line fit
    q_high : A float, upper quantile on the steady state line fit
    ax_handle : A matplotlib axis handle, for adding onto an existing plot
    B : Number of bootstrap iterations

    Returns
    -------------
    fig : A matplotlib figure handle (if ax_handle is None)
    ax : A matplotlib axis handle (if ax_handle is None)
    summary_stats : A list containing the variables [deltas, kappas, delta_ml, kappa_ml], see mystyle.ana.bootstrap_lfc

    Note
    -------------
    Should standardize (x,y) first
    """
    slopes, intercepts, slope_ml, intercept_ml = ana.bootstrap_2d_pca(x, y, B)
    summary_stats = [slopes, intercepts, slope_ml, intercept_ml]
    # Quantiles
    x_sp = np.linspace(min(x) * 0.9, max(x) * 1.1, num=200)

    y_fits = np.zeros([B, len(x_sp)])

    for i in range(B):
        y_fits[i, :] = slopes[i]*x_sp  + intercepts[i]

    ql = np.percentile(y_fits, q_low, axis=0)
    qh = np.percentile(y_fits, q_high, axis=0)

    if ax_handle is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    else:
        ax = ax_handle

    ax.plot(x_sp, x_sp * slope_ml + intercept_ml, '-r', label='PCA (ML)')
    ax.fill_between(x_sp, ql, qh, color='red', alpha=0.5, label='95\% Boot. C.I.')

    ax.legend()

    if ax_handle is None:
        return fig, ax, summary_stats, x_sp
    else:
        return summary_stats

def plot_bootstrapped_lr(x, y, q_low=2.5, q_high=97.5, ax_handle=None, B=1000,
                        legend=True, alpha=0.5):
    """
    Plot bootstrapped linear regression trend line

    Parameters
    --------------
    x : A numpy array, x-values
    y : A numpy array, x-values
    q_low : A float, lower quantile on the steady state line fit
    q_high : A float, upper quantile on the steady state line fit
    ax_handle : A matplotlib axis handle, for adding onto an existing plot
    B : Number of bootstrap iterations
    legend : A bool, if True add to legend.

    Returns
    -------------
    fig : A matplotlib figure handle (if ax_handle is None)
    ax : A matplotlib axis handle (if ax_handle is None)
    summary_stats : A list containing the variables [slope_ml, intercept_ml, pval, r_sq], see mystyle.ana.bootstrap_lfc

    Note
    -------------
    Should remove NaN values first
    """
    x_sp, y_ql, y_qh = ana.bootstrap_lr(x, y,
        x_sp=None, q_low=q_low, q_high=q_high, B=B)
    lr_ml = ss.linregress(x, y)
    slope_ml = lr_ml.slope
    intercept_ml = lr_ml.intercept
    pval = lr_ml.pvalue
    r_sq = lr_ml.rvalue**2

    summary_stats = {'slope_ml':slope_ml, 'intercept_ml':intercept_ml,
        'pval':pval, 'r_sq':r_sq}


    if ax_handle is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    else:
        ax = ax_handle

    if legend:
        ax.plot(x_sp, x_sp * slope_ml + intercept_ml, '-r', label='LR')
        ax.fill_between(x_sp, y_ql, y_qh, color='red', alpha=alpha, label='95\% Boot. C.I.')
    else:
        ax.plot(x_sp, x_sp * slope_ml + intercept_ml, '-r')
        ax.fill_between(x_sp, y_ql, y_qh, color='red', alpha=alpha)

    if ax_handle is None:
        return fig, ax, summary_stats, x_sp
    else:
        return summary_stats


def plot_decision_boundary_2d(X, clf, dx=0.1, dy=0.1, alpha=0.4,
                              ax_handle=None, colors=None):
    """
    Plot the decision boundary of a 2D classifier as a contour plot

    Parameters
    -------------
    X : A 2D numpy array, the design matrix
    clf : A classifier, which has the method clf.predict(X)

    dx : A float, the mesh distance in x
    dy : A float, the mesh distance in y
    alpha : A float, transparency of the contour
    ax_handle : A matplotlib axis handle, for adding onto an existing plot
    colors : A list of strings, the colors of each contour

    Returns
    -------------
    fig : A matplotlib figure handle (if ax_handle is None)
    ax : A matplotlib axis handle (if ax_handle is None)
    """

    x_min, x_max = X[:, 0].min() * 0.9, X[:, 0].max() * 1.1
    y_min, y_max = X[:, 1].min() * 0.9, X[:, 1].max() * 1.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    if ax_handle is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    else:
        ax = ax_handle

    if colors is not None:
        cmap = mpl.colors.ListedColormap(colors)
        ax.contourf(xx, yy, Z, alpha=alpha, cmap=cmap)
    else:
        ax.contourf(xx, yy, Z, alpha=alpha)

    if ax_handle is None:
        return fig, ax

def plot_bar_whiskers_jitter_significance(data, comparison_columns,
                                          significant_comparison_columns,
                                          heights, ylabel,
                                          xlabels=None,
                                          ax_handle=None,
                                          median_notch=False,
                                          boxplot_color='black',
                                          boxplot_linewidth=2,
                                          markersize=12,
                                          xtick_rotation=90,
                                          marker=None,
                                          color=None,
                                          alpha=0.2,
                                          whis = [2.5, 97.5]):
    """
    Make a jittered boxplot significance test

    Parameters
    -------------------
    d : A pandas dataframe, where each column corresponds to data to be plotted with jitter + boxplot
    heights : A list, heights of the significance annotations, for each comparison
    comparison_columns : A list of lists, where each element corresponds to a pair of columns to compare
    significant_comparison_columns : A list of lists, where each element corresponds to a pair of significant column comparisons
    heights : A list of floats, the height of each comparison annotation
    xlabels : A list of strings, the x-labels
    ax_handle : A matplotlib axis handle, for adding onto an existing plot
    median_notch : A bool, to plot the lower and upper quartiles of the median
    boxplot_color : A string, the boxplot color
    boxplot_linewidth : A float, the boxplot linewidth
    markersize: An int, the marker size
    marker : A string or a list of strings, the marker of the points
    color : A string or a list of strings, the color of the points
    alpha : A float, transparency
    whis : A list of floats, the quantiles for whiskers

    Returns
    -------------
    fig : A matplotlib figure handle (if ax_handle is None)
    ax : A matplotlib axis handle (if ax_handle is None)

    """

    if ax_handle is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    else:
        ax = ax_handle

    make_jitter_plots(data, names=data.columns, ylabel=ylabel, ax_handle=ax,
                        alpha=alpha, markersize=markersize, xlabels=xlabels,
                        marker=marker, color=color)

    bp = data.boxplot(ax=ax,notch=median_notch, grid=False, whis = whis,
                        showfliers=False, return_type='dict')
    for _, line_list in bp.items():
        for line in line_list:
            line.set_color(boxplot_color)
            line.set_linewidth(boxplot_linewidth)

    previous_ymaxes = []

    for i, comparison in enumerate(comparison_columns):
        comp1, comp2 = comparison
        x1, x2 = np.nonzero(data.columns==comp1)[0][0]+1, np.nonzero(data.columns==comp2)[0][0]+1

        y_max = data.loc[:,[comp1,comp2]].max().values.max()
        previous_ymaxes.append(y_max)
        y, h, col = max(previous_ymaxes) + heights[i], 2, 'k'
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
        if comparison in significant_comparison_columns:
            ax.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col, fontsize=20)
        else:
            ax.text((x1+x2)*.5, y+h, "ns", ha='center', va='bottom', color=col, fontsize=20)
    if xlabels is not None:
        ax.set_xticklabels(xlabels, rotation=xtick_rotation)
    if ax_handle is None:
        return fig, ax

def plot_kde_2d(x, y, xlabel, ylabel, limits=None, ax_handle=None, cmap='Reds'):
    """
    Plot a 2D Gaussian KDE

    Parameters
    ---------------
    x : An array of floats
    y : An array of floats
    xlabel : A float
    ylabel : A float
    limits : A list of floats containing xmin, xmax, ymin, ymax for plot
    ax_handle : A matplotlib axis handle, for adding onto an existing plot
    cmap : A string, a matplotlib color map

    Returns
    -------------
    fig : A matplotlib figure handle (if ax_handle is None)
    ax : A matplotlib axis handle (if ax_handle is None)
    """

    if len(x)>20000:
        print("Warning: len(x)>20000, may be slow.".format(x))
    if ax_handle is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    else:
        ax = ax_handle
    if limits is None:
        xmin, xmax, ymin, ymax = min(x), max(x), min(y), max(y)
    else:
        xmin, xmax, ymin, ymax = limits

    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = ss.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)


    cfset = ax.contourf(xx, yy, f, cmap=cmap)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    cfset.cmap.set_under('w')
    cfset.set_clim(cfset.levels[1])

    if ax_handle is None:
        return fig, ax


########################################
# Useful misc functions
########################################

def to_latex(x, dp=1, double_backslash=True):
    """
    Convert a decimal into LaTeX scientific notation

    Parameters
    ---------------
    x : A float, the number to convert to LaTeX notation, e.g. 0.42
    dp : An int, the number of decimal places for the
    double_backslash : A bool, whether to use a double-backslash for LaTeX commands

    Returns
    -----------
    A string where x is cast in LaTeX as scientific notation, e.g. "4.2 \times 10^{-1}"

    """
    fmt = "%.{}e".format(dp)
    s = fmt % x
    arr = s.split('e')
    m = arr[0]
    n = str(int(arr[1]))
    if double_backslash:
        return str(m) + '\\times 10^{' + n + '}'
    else:
        return str(m) + '\times 10^{' + n + '}'
