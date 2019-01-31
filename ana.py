import numpy as np
from sklearn.decomposition import PCA



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
