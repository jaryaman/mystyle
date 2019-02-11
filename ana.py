import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import mutual_info_score



def bootstrap_lfc(w,m,B=100):
	'''
	Fit a steady state line of the linear feedback control to data in the w,m plane using PCA

	Parameters
	--------------
	w : A numpy array, data for wild-type copy number
	m : A numpy array, data for mutant copy number
	B : Number of bootstrap iterations

	Returns
	-------------
	deltas : A numpy array, bootstrapped delta
	kappas : A numpy array, bootstrapped kappa
	delta_ml : A float, most likely value of delta
	kappa_ml : A float, most likely value of kappa
	'''
	X = np.vstack([w,m]).T
	mu = np.mean(X, axis = 0)

	n_data = len(w)
	if n_data != len(m):
		raise Exception('len(w) != len(m)')

	# Bootstrap PCA
	deltas = []
	kappas = []
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
		grad = max_evec[1]/max_evec[0]
		intercept = mu[1] - grad*mu[0]

		delta = -1.0/grad
		if i == 0:
			delta_ml = 1.0*delta # deep copy
			kappa_ml = intercept*delta
		else:
			deltas.append(delta)
			kappas.append(intercept*delta) # kappa = intercept

	deltas = np.array(deltas)
	kappas = np.array(kappas)
	return deltas, kappas, delta_ml, kappa_ml

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
