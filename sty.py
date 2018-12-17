import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from pdb import set_trace
from matplotlib import cm
import matplotlib.ticker
from matplotlib.ticker import FormatStrFormatter

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
	Params:
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

	Params
	--------
	ax: A matplotlib axis
	"""
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()

####################################################################
# Useful plots
####################################################################


def make_heatmap(matrix,
		xlabel,
		ylabel,
		zlabel,
		xticklabels,
		yticklabels,
		plotextensions = ['svg','png'],
		colorbar_heatmap=cm.jet,
		figname = None,
		vmin = None, vmax = None,out_dir=os.getcwd()):
		"""Make a heatmap of a matrix where rows are ratios and columns are magnitudes of the fusion/fission rate

			:param matrix: A numpy matrix, intensities for heatmap. Expect a square matrix of dimension n_points x n_points
			:param zlabel: A string, colorbar label of heatmap
			:param figname: A string, prefix to figure name. If none, do not write to file
			:param xlabel: A string, x-label of heatmap
			:param ylabel: A string, y-label of heatmap
			:param vmin: A float, minimum value for colorbar
			:param vmax: A float, maximum value for colorbar
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

	Params
	----------
	data: A pandas dataframe, contains numeric values in the column `name`
	name: A string, the name of the column to be plotted
	dx: A float, width of the jitter

	Returns
	----------
	datax: A pandas series, containing non-null values of the column to plot
	x: A numpy array, jittered x-values
	"""
	datax = data[name]
	datax = datax[~datax.isnull()]
	x = np.ones(len(datax))+np.random.uniform(-dx,dx,size=len(datax))
	return datax, x

def make_jitter_plots(data, names, ylabel, dx=0.1, ytick_fmt='%.2f', xlabels = None):
	"""
	Make a jitter plot of columns from a pandas dataframe

	Params
	--------
	data: A pandas dataframe, contains numeric values in the columns `names`
	name: An array of strings, the name of the columns to be plotted
	dx: A float, width of the jitter
	ylabel: A string, the y-label for the plot

	Returns
	--------
	fig: A matplotlib figure handle
	ax: A matplotlib axis handle
	"""
	hx_tuples = []
	for name in names:
		hx_tuples.append(get_non_null_and_jitter(data, name, dx))

	fig, ax = plt.subplots(1,1)
	for i in range(len(names)):
		hi = hx_tuples[i][0]
		xi = hx_tuples[i][1]
		ax.plot(i+xi,hi,'.k')

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
	return fig, ax
