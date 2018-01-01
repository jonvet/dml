import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from math import sqrt
import seaborn as sn
from matplotlib import pyplot as plt
import warnings
import pickle as pkl

def vineBeta(dim, betaparam):
    P = np.zeros([dim, dim])
    S = np.eye(dim)
    for k in range(0, dim-1):
        for i in range(k+1, dim):
            P[k,i] = np.random.beta(betaparam, betaparam)
            P[k,i] = (P[k,i]-0.5)*2
            p = P[k,i]
            for l in range(k-1, -1, -1):
                p = p * np.sqrt((1-np.power(P[l,i],2)) * (1-np.power(P[l,k], 2))) + P[l,i]*P[l,k]
            S[k,i] = p
            S[i,k] = p
    return S

def run_OLS(X, y):
    OLS = linear_model.LinearRegression()
    OLS.fit(X, y)
    return OLS.coef_[0]

def run_naive_DML(z, y, d, max_leaf_nodes_y, max_leaf_nodes_d):
    model_y_naive = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes_y)
    model_d_naive = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes_d)
    model_y_naive.fit(z, y)
    model_d_naive.fit(z, d)
    y_hat_naive = model_y_naive.predict(z)
    d_hat_naive = model_d_naive.predict(z)
    residuals_simple = d - d_hat_naive
    return np.matmul(residuals_simple,(y - y_hat_naive))/np.matmul(residuals_simple,d)

def run_good_DML(z, y, d, max_leaf_nodes_y, max_leaf_nodes_d):
    thetas = []
    tf = KFold(len(z), n_folds=5)
    for I_index, IC_index in tf:
        model_y = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes_y)
        model_d = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes_d)
        model_y.fit(z[I_index], y[I_index])
        model_d.fit(z[I_index], d[I_index])
        y_hat = model_y.predict(z[IC_index])
        d_hat = model_d.predict(z[IC_index])
        residuals = d[IC_index] - d_hat
        theta = np.matmul(residuals, (y[IC_index] - y_hat))/np.matmul(residuals, d[IC_index])
        thetas.append(theta)
    return np.mean(thetas)

def xval_rf(z, y, d, verbose=False):
    max_N = len(z)
    y_dict = {'best_error': np.inf}
    d_dict = {'best_error': np.inf}
    tf = KFold(max_N, n_folds=3)
    z = z[:max_N,:]
    y = y[:max_N]
    d = d[:max_N]
    for nodes in range(10, 30):
        for train, test in tf:
            model_y = RandomForestRegressor(max_leaf_nodes=nodes)
            model_d = RandomForestRegressor(max_leaf_nodes=nodes)
            model_y.fit(z[train], y[train])
            model_d.fit(z[train], d[train])
            y_hat = model_y.predict(z[test])
            d_hat = model_d.predict(z[test])
            error_y = sqrt(mean_squared_error(y[test], y_hat))
            error_d = sqrt(mean_squared_error(d[test], d_hat))
            if y_dict['best_error'] > error_y:
                y_dict['best_nodes'] = nodes
                y_dict['best_error'] = error_y
            if d_dict['best_error'] > error_d:
                d_dict['best_nodes'] = nodes
                d_dict['best_error'] = error_d
    if verbose:
        print(y_dict)
        print(d_dict)
    return y_dict, d_dict

def get_corr_matrix(d, n):
    betaparameter = 1.0
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        runtimeerror = True
        tries = 1
        while runtimeerror:
            if tries % 2==0:
                    betaparameter += 0.1
            try:
                corr = vineBeta(d, betaparameter)
                z = np.random.multivariate_normal(mean=np.zeros(len(corr)), cov=corr, size=n)
                runtimeerror = False
            except Warning as e:
                print('Tried to generate pos sem definite matrix {} times, trying beta parameter {} now'.format(tries, betaparameter))
            tries += 1
    return corr

def run_experiments(Ds, Ns, Ms, output_file=None):
    results = {}
    for D in Ds:
        corr = get_corr_matrix(D, 500)
        for N in Ns:
            print('\n\nRunning experiments with D={} and N={}'.format(D,N))
            b=1/np.arange(1, D+1)
            theta_hat = np.zeros([Ms, 3])
            for i in range(Ms):
                print('\rRun: {}'.format(i), end='    ')
                z = np.random.multivariate_normal(mean=np.zeros(len(corr)), cov=corr, size=N)
                g = np.power(np.cos(np.matmul(z, b)), 2)
                m = np.sin(np.matmul(z, b)) + np.cos(np.matmul(z, b))
                d = m + np.random.randn(N)
                y = theta*d + g + np.random.randn(N)

                # Naive OLS
                theta_hat[i,0] = run_OLS(np.concatenate((np.expand_dims(d, 1), z), axis=1), y)

                # Find best number of leaf nodes
                dict_y, dict_d = xval_rf(z,y,d)
                max_leaf_nodes_y = dict_y['best_nodes']
                max_leaf_nodes_d = dict_d['best_nodes']

                # Naive DML
                theta_hat[i,1] = run_naive_DML(z, y, d, max_leaf_nodes_y, max_leaf_nodes_d)

                # Good DML
                theta_hat[i,2] = run_good_DML(z, y, d, max_leaf_nodes_y, max_leaf_nodes_d)

            print('Mean: {}'.format(np.mean(theta_hat,axis=0)))
            print('Median: {}'.format(np.median(theta_hat,axis=0)))
            df = pd.DataFrame(theta_hat)
            results[len(results)] = {'dict_y':dict_y, 'dict_d': dict_d, 'theta_hat': theta_hat, 'N': N, 'D': D, 
                'mean': np.mean(theta_hat,axis=0), 'median':np.median(theta_hat,axis=0)}

    if output_file is not None:
        with open(output_file, 'wb') as f:
            pkl.dump(results, f)
    return theta_hat, results

def plot_distributions(thetas):
    bw=.02
    plt.close()
    with sn.hls_palette(3, l=.3, s=.8):
        sn.kdeplot(thetas[:,0], label='OLS', shade=True, bw=bw)
        sn.kdeplot(thetas[:,1], label='DML naive', shade=True, bw=bw)
        sn.kdeplot(thetas[:,2], label='DML cross-fitting', shade=True, bw=bw)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
    plt.xlim(0.2, 0.8)
    plt.show()

theta=0.5

plt.close()
theta_hat, results = run_experiments([10, 30, 50, 70], [500, 1000, 1500, 2000], 100, output_file='results_single_figure.pkl')
plot_distributions(theta_hat)
