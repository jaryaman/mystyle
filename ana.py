import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import mutual_info_score
import scipy.stats as ss


def bootstrap_2d_pca(x, y, B=100):
    """
    Compute a line of best fit to 2D data and bootstrap confidence interval

    Parameters
    --------------
    x : A numpy array, data for dimension 1
    y : A numpy array, data for dimension 2
    B : Number of bootstrap iterations

    Returns
    -------------
    slopes : A numpy array, bootstrapped slope
    intercept : A numpy array, bootstrapped intercept
    slope_ml : A float, most likely value of slope
    intercept_ml : A float, most likely value of intercept
    """
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
            slopes.append(slope_b)
            intercepts.append(mu[1] - slope_b*mu[0])

    slopes = np.array(slopes)
    intercepts = np.array(intercepts)
    return slopes, intercepts, slope_ml, intercept_ml


def calc_mutual_information(x, y, bins):
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


def std_error_lr(x, y):
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

def one_sample_z_test(estimate, pop_mean, std_err, two_sided=True):
    """
    One sample Z-test

    Parameters
    --------------
    estimate : A float, plug-in estimate for the normally distributed quantity under the null hypothesis
    pop_mean : A float, population mean under the null hypothesis
    std_err : A float, plug-in estimate for the standard error of the quantity under the null hypothesis
    n : An int, the number of observations associated with `estimate` and `std_err`
    two_sided : A bool, whether to perform a two-sided test

    Returns
    --------------
    z : A float, the z-statistic
    p : A float, the p-value under the null hypothesis

    References
    ---------------
    https://en.wikipedia.org/wiki/Z-test
    """
    z = abs((estimate - pop_mean)/std_err)
    if two_sided:
        return z, 2.0*ss.norm.sf(z)
    else:
        return z, ss.norm.sf(z)


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
    x_sp_defined = 1
    if x_sp is None:
        x_sp = np.linspace(min(x), max(x))
        x_sp_defined = 0
    y_arr = np.zeros((B, len(x_sp)))
    for i in range(B):
        idxs = np.random.choice(len(x),size=len(x),replace=True)
        xb = x[idxs]
        yb = y[idxs]
        lr = ss.linregress(xb, yb)
        y_arr[i,:] = lr.slope * x_sp + lr.intercept
    y_ql = np.percentile(y_arr, q_low, axis = 0)
    y_qh = np.percentile(y_arr, q_high, axis = 0)

    if x_sp_defined == 0:
        return x_sp, y_ql, y_qh
    else:
        return y_ql, y_qh

def bootstrap_pearson_corr_coef(x, y, B=1000):
    """
    Bootstrap Pearson's correlation coefficient

    Parameters
    -------------

    x : An array of floats, the independent variable
    y : An array of floats, the dependent variable

    Returns
    -------------
    r_boot : An array of floats, bootstrapped values of Pearson's correlation

    """
    r_boot = np.zeros(B)
    for i in range(B):
        idxs = np.random.choice(len(x),size=len(x),replace=True)
        xb = x[idxs]
        yb = y[idxs]
        lr = ss.linregress(xb, yb)
        r_boot[i] = lr.rvalue
    return r_boot

def multivariate_gaussian_2d(x, y, mu, Sigma):
    """Return the multivariate Gaussian distribution pdf on a lattice

    Parameters
    -----------------
    x : An array of floats, the x-axis to evaluate the multivariate normal
    y : An array of floats, the y-axis to evaluate the multivariate normal
    mu : An array of floats, the mean of the multivariate normal
    Sigma: A 2x2 array of floats, the covariance matrix of the multivariate
           normal

    Returns
    -----------------
    z : A 2D array of floats, the multivariate normal evaluated on the lattice
        defined by x,y

    Example
    ----------------
    import mystyle.ana as ana
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.linspace(-2,2)
    y = np.linspace(-2,2)
    mu=[0.0,0.0]
    Sig=[[1.07,0.63],[0.63,0.64]]
    z = ana.multivariate_gaussian_2D(x, y, mu=mu, Sigma=Sig)
    X = np.random.multivariate_normal(mu,Sig,size=100)

    fig, ax = plt.subplots(1,1)
    ax.plot(X[:,0],X[:,1],'ok',alpha=alpha)
    ax.contour(x, y, z, cmap='Reds', alpha=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    Notes
    --------------------
    Source: https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/
    """
    X, Y = np.meshgrid(x,y)
    mu = np.array(mu)
    Sigma = np.array(Sigma)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
    return np.exp(-fac / 2) / N

def standardize(X):
    """Z-transform an array

    param X: An N x D array where N is the number of examples and D is the number of features

    returns: An N x D array where every column has been rescaled to 0 mean and unit variance
    """
    return (X - X.mean())/X.std(ddof=1)

def benjamini_hochberg_correction(pval_dict, alpha=0.05):
    """
    Perform Bonferroni correction on a dictionary of p-values

    Parameters
    -------------------
    pval_dict : A dictionary, keys are strings of hypothesis names, values are raw p-values
    alpha : A float, the false-discovery rate

    Returns
    ------------------
    A list of significant hypotheses with FDR <= alpha
    """
    n_hypothesis = len(pval_dict)
    hyp_names = list(pval_dict.keys())
    hyp_pvals = list(pval_dict.values())
    hyp_pvals = np.vstack((np.arange(len(hyp_pvals)), hyp_pvals)).T
    hyp_pvals = hyp_pvals[hyp_pvals[:, 1].argsort()]

    sig_hyps = []
    for i in range(len(pval_dict)):
        if hyp_pvals[i,1] < (i+1)*alpha/n_hypothesis:
            sig_hyps.append(int(hyp_pvals[i,0]))
    return list(np.array(hyp_names)[sig_hyps])
