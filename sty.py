import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from pdb import set_trace
from matplotlib import cm
import matplotlib.ticker
from matplotlib.ticker import FormatStrFormatter
import warnings
from IPython import get_ipython

import mystyle.ana as ana

def reset_plots():
	"""
	Makes axes large, and enables LaTeX for matplotlib plots
	"""
	plt.close('all')
	fontsize = 20
	legsize = 15
	plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
	plt.rc('text', usetex=True)
	font = {'size' : fontsize}
	plt.rc('font', **font)
	rc={'axes.labelsize': fontsize,
	'font.size': fontsize,
	'axes.titlesize': fontsize,
	'xtick.labelsize':fontsize,
	'ytick.labelsize':fontsize,
	'legend.fontsize': legsize}
	mpl.rcParams.update(**rc)
	mpl.rc('lines', markersize=10)
	plt.rcParams.update({'axes.labelsize': fontsize})
	mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

def remove_tex_axis(ax, xtick_fmt = '%d', ytick_fmt = '%d'):
	"""
	Makes axes normal font in matplotlib.

	Parameters
	---------------
	xtick_fmt : A string, defining the format of the x-axis
	ytick_fmt : A string, defining the format of the y-axis
	"""
	fmt = matplotlib.ticker.StrMethodFormatter("{x}")
	ax.xaxis.set_major_formatter(fmt)
	ax.yaxis.set_major_formatter(fmt)
	ax.xaxis.set_major_formatter(FormatStrFormatter(xtick_fmt))
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

def legend_outside(ax):
	"""
	Put legend outside the plot area

	Parameters
	---------------
	ax : A matplotlib axis
	"""
	ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

####################################################################
# Useful plots
####################################################################


def make_heatmap(matrix, xlabel, ylabel, zlabel, xticklabels, yticklabels,
		plotextensions = ['svg','png'],
		colorbar_heatmap=cm.jet,
		figname = None,
		vmin = None, vmax = None,out_dir=os.getcwd()):
	"""
	Make a heatmap of a matrix where rows are plotted on the vertical axis, and columns plotted along the horizontal

	Parameters
	---------------
	matrix : A numpy matrix, intensities for heatmap. Expect a square matrix of dimension (n_points x n_points)
	zlabel : A string, colorbar label of heatmap
	figname : A string, prefix to figure name. If none, do not write to file
	xlabel : A string, x-label of heatmap
	ylabel : A string, y-label of heatmap
	vmin : A float, minimum value for colorbar
	vmax : A float, maximum value for colorbar
	"""
	plt.close('all')
	nan_col = 0.4
	nan_rgb = nan_col*np.ones(3)
	colorbar_heatmap.set_bad(nan_rgb)

	fig, ax = plt.subplots(1,1, figsize = (9,9))
	im = ax.imshow(np.flipud(matrix), cmap = colorbar_heatmap, vmin=vmin, vmax=vmax)
	plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=zlabel)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.set_xticks(np.arange(len(xticklabels)))
	ax.set_yticks(np.arange(len(yticklabels)))
	ax.set_xticklabels(xticklabels)
	ax.set_yticklabels(yticklabels[::-1])

	if figname is not None:
		for p in plotextensions:
			plt.savefig(out_dir+'/'+figname+'.'+p, bbox_inches='tight')

def get_non_null_and_jitter(data, name, dx):
	"""
	Strip out null values from a dataframe and return jittered x with y-values

	Parameters
	---------------
	data : A pandas dataframe, contains numeric values in the column `name`
	name : A string, the name of the column to be plotted
	dx : A float, width of the jitter

	Returns
	----------
	datax : A pandas series, containing non-null values of the column to plot
	x : A numpy array, jittered x-values
	"""
	datax = data[name]
	datax = datax[~datax.isnull()]
	x = np.ones(len(datax))+np.random.uniform(-dx,dx,size=len(datax))
	return datax, x

def make_jitter_plots(data, names, ylabel, dx=0.1, ytick_fmt='%.2f', xlabels = None, ax_handle = None, alpha = 1):
	"""
	Make a jitter plot of columns from a pandas dataframe

	Parameters
	---------------
	data : A pandas dataframe, contains numeric values in the columns `names`
	names: An array of strings, the name of the columns to be plotted
	dx : A float, width of the jitter
	ylabel : A string, the y-label for the plot
	ax_handle : A matplotlib axis handle. When defined, the function will add a jitter plot to an ax object
	xlabels : A list of strings, the names along the x-axis
	alpha : A float, transparency on data points

	Returns
	--------
	fig : A matplotlib figure handle
	ax : A matplotlib axis handle
	"""
	yx_tuples = []
	for name in names:
		yx_tuples.append(get_non_null_and_jitter(data, name, dx))

	if ax_handle is None:
		fig, ax = plt.subplots(1,1)
	else:
		ax = ax_handle

	for i in range(len(names)):
		yi = yx_tuples[i][0]
		xi = yx_tuples[i][1]
		ax.plot(i+xi,yi,'.k',alpha=alpha)

	remove_tex_axis(ax,ytick_fmt=ytick_fmt)
	ax.set_xticks(1+np.arange(len(names)))
	if xlabels is None:
		names = [s.replace('_','-') for s in names] # underscores make LaTeX unhappy
		ax.set_xticklabels(names)
	else:
		ax.set_xticklabels(xlabels)
	for t in ax.get_xticklabels():
		t.set_rotation(90)
	ax.set_ylabel(ylabel)
	if ax_handle is None:
		return fig, ax

def plot_w_m_lfc(w, m, q_low = 2.5, q_high = 97.5, ax_handle=None, B=100):
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
	deltas, kappas, delta_ml, kappa_ml = ana.bootstrap_lfc(w,m,B)
	summary_stats = [deltas, kappas, delta_ml, kappa_ml]

	# Quantiles
	w_sp = np.linspace(min(w)*0.9,max(w)*1.1,num=200)

	m_fits = np.zeros([B,len(w_sp)])
	for i in range(B):
		m_fits[i,:] = -w_sp/deltas[i] + kappas[i]/deltas[i]

	ql = np.percentile(m_fits, q_low, axis = 0)
	qh = np.percentile(m_fits, q_high, axis = 0)

	if ax_handle is None:
		fig, ax = plt.subplots(1,1,figsize=(5,5))
	else:
		ax = ax_handle

	ax.plot(w_sp, -w_sp/delta_ml + kappa_ml/delta_ml, '-r', label='PCA (ML)')
	ax.fill_between(w_sp, ql, qh, color = 'red', alpha = 0.5, label = '95\% Boot. C.I.')

	ax.legend()
	ax.set_xlabel('Wild-type copy number, $w$')
	ax.set_ylabel('Mutant copy number, $m$')

	if ax_handle is None:
		return fig, ax, summary_stats
	else:
		return summary_stats

def plot_decision_boundary_2D(X,clf,dx=0.1,dy=0.1,alpha=0.4,
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

	x_min, x_max = X[:, 0].min()*0.9, X[:, 0].max()*1.1
	y_min, y_max = X[:, 1].min()*0.9, X[:, 1].max()*1.1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
	                     np.arange(y_min, y_max, 0.1))
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)

	if ax_handle is None:
		fig, ax = plt.subplots(1,1,figsize=(5,5))
	else:
		ax = ax_handle

	if colors is not None:
		cmap = mpl.colors.ListedColormap(colors)
		ax.contourf(xx, yy, Z, alpha=alpha, cmap=cmap)
	else:
		ax.contourf(xx, yy, Z, alpha=alpha)

	if ax_handle is None:
		return fig, ax



########################################
# Useful misc functions
########################################

def to_latex(x, dp=1, double_backslash=True):
	'''
	Convert a decimal into LaTeX scientific notation

	Parameters
	---------------
	x : A float, the number to convert to LaTeX notation, e.g. 0.42
	dp : An int, the number of decimal places for the
	double_backslash : A bool, whether to use a double-backslash for LaTeX commands

	Returns
	-----------
	A string where x is cast in LaTeX as scientific notation, e.g. "4.2 \times 10^{-1}"

	'''
	fmt = "%.{}e".format(dp)
	s = fmt%x
	arr = s.split('e')
	m = arr[0]
	n = str(int(arr[1]))
	if double_backslash:
		return str(m)+'\\times 10^{'+n+'}'
	else:
		return str(m)+'\times 10^{'+n+'}'
