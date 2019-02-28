import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import mutual_info_score
import scipy.stats as ss


def bootstrap_2D_PCA(x,y,B=100):
	'''
	Fit a steady state line of the linear feedback control to data in the w,m plane using PCA

	Parameters
	--------------
	x : A numpy array, data for dimension 1
	y : A numpy array, data for dimension 2
	B : Number of bootstrap iterations

	Returns
	-------------
	slopes : A numpy array, bootstrapped slope
	intercept : A numpy array, bootstrapped intercept
	delta_ml : A float, most likely value of slope
	kappa_ml : A float, most likely value of intercept
	'''
	X = np.vstack([x,y]).T
	mu = np.mean(X, axis = 0)

	n_data = len(x)
	if n_data != len(y):
		raise Exception('len(x) != len(y)')

	# Bootstrap PCA
	slopes = []
	intercepts = []
	pca = PCA(n_components=1)
	for i in range(B+1):
		if i == 0:
			idxs = np.arange(n_data) # take all data, unbootstrapped
		else:
			idxs = np.random.choice(n_data,size=n_data,replace=True)
		X_b = X[idxs,:] # bootstrapped data
		pca.fit(X_b)
		Sigma = pca.get_covariance()
		eigval, eigvec = np.linalg.eig(Sigma)
		max_evec = eigvec[:,np.argmax(eigval)]

		if i == 0:
			slope_ml = max_evec[1]/max_evec[0] # deep copy
			intercept_ml = mu[1] - slope_ml*mu[0]
		else:
			slope_b = max_evec[1]/max_evec[0]
			slopes.append(max_evec[1]/max_evec[0])
			intercepts.append(mu[1] - slope_b*mu[0])

	slopes = np.array(slopes)
	intercepts = np.array(intercepts)
	return slopes, intercepts, slope_ml, intercept_ml

def calc_MI(x, y, bins):
	"""
	Calculate the mutual information between two continuous random variables

	Parameters
	--------------
	x : An array of floats
	y : An array of floats
	bins : An int, the number of bins for performing a 2D histogram on (x,y)

	Returns
	--------------
	mi : A float, mutual information

	References
	--------------
	https://stackoverflow.com/questions/20491028/optimal-way-to-compute-pairwise-mutual-information-using-numpy
	https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mutual_info_score.html

	To do
	--------------
	- Find a rational way of picking bins
	- Read up about this some more
	"""
	c_xy = np.histogram2d(x, y, bins)[0]
	mi = mutual_info_score(None, None, contingency=c_xy)
	return mi

def std_error_lr(x,y):
	"""
	Standard errors on parameters in linear regression

	Parameters
	--------------
	x : An array of floats, the independent variable
	y : An array of floats, the dependent variable

	Returns
	-------------
	se_sl : A float, standard error in the slope
	se_int : A float, standard error in the intercept

	References
	-------------
	See Wasserman, All of Statistics, p214
	"""
	lr = ss.linregress(x,y)
	n = len(x)
	mest = lr.slope
	cest = lr.intercept
	sigmahat = np.sqrt((1.0/(n-2.0))*np.sum((y-(mest*x+cest))**2))
	sx = np.sqrt(np.sum((x- np.mean(x))**2)/float(n))

	se_sl = sigmahat/(sx*np.sqrt(n))
	se_int = sigmahat/(sx*np.sqrt(n)) * np.sqrt(np.sum(x**2)/float(n))
	return se_sl, se_int

def one_sample_t_test(estimate, pop_mean, std_err, n, two_sided=True):
	"""
	One sample T-test

	Parameters
	--------------
	estimate : A float, plug-in estimate for the T-distributed quantity under the null hypothesis
	pop_mean : A float, population mean under the null hypothesis
	std_err : A float, plug-in estimate for the standard error of the quantity under the null hypothesis
	n : An int, the number of observations associated with `estimate` and `std_err`
	two_sided : A bool, whether to perform a two-sided test

	Returns
	--------------
	t : A float, the t-statistic
	p : A float, the p-value under the null hypothesis

	References
	---------------
	https://en.wikipedia.org/wiki/Student%27s_t-test#One-sample_t-test
	"""
	t = abs((estimate - pop_mean)/std_err)
	df = n - 2
	if two_sided:
		return t, 2.0*ss.t.sf(t,df)
	else:
		return t, ss.t.sf(t,df)

def bootstrap_lr(x, y, x_sp = None, q_low = 2.5, q_high = 100-2.5, B=1000):
	"""
	Bootstrap linear regression

	Parameters
	-------------

	x : An array of floats, the independent variable
	y : An array of floats, the dependent variable

	x_sp : An array of floats, the space over the independent variable to evaluate the bootstrap
	q_low : The lower quantile for the bootstrap
	q_high : The upper quantile for the bootstrap

	Returns
	-------------
	y_ql : An array of floats, the lower bootstrapped quantile of the dependent variable under linear regression over x_sp
	y_qh : An array of floats, the upper bootstrapped quantile of the dependent variable under linear regression over x_sp
	"""
	if x_sp is None:
		x_sp = np.linspace(min(x), max(x))
	y_arr = np.zeros((B, len(x_sp)))
	for i in range(B):
		idxs = np.random.choice(len(x),size=len(x),replace=True)
		xb = x[idxs]
		yb = y[idxs]
		lr = ss.linregress(xb, yb)
		y_arr[i,:] = lr.slope * x_sp + lr.intercept
	y_ql = np.percentile(y_arr, q_low, axis = 0)
	y_qh = np.percentile(y_arr, q_high, axis = 0)

	if x_sp is None:
		return x_sp, y_ql, y_qh
	else:
		return y_ql, y_qh
