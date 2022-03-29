import scanpy as sc
from scipy.sparse import issparse
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import numpy as np
import pandas as pd


def Detect_Genes(adata, nper = 1000, quantile = 0.95, permutated="gene"):
    # X must be the count matrix
    print("fit regression and calculate dropout rate")
    X = adata.X
    nspot, ngene = adata.shape
    if issparse(X):
        X = X.A
    mus = X.mean(axis=0)
    vars = X.var(axis=0)

    # fit the regression
    ys = vars - mus
    mus2 = mus**2
    XX = mus2.reshape(len(mus2), 1)
    y = ys.reshape(len(ys), 1)
    lm_fit = LinearRegression(fit_intercept=False)
    lm_model = lm_fit.fit(X=XX, y=y)
    rho = 1/lm_model.coef_
    observedDropouts = np.mean(X==0, axis=0)
    predDropouts = (rho/(mus+rho))**rho
    predDropouts = predDropouts.reshape(-1)

    print("start the bootstrap")
    preds = []
    if permutated =="gene":
        for i in tqdm(range(nper)):
            ids = np.random.choice(ngene, ngene)
            Xnew = XX[ids]
            ynew = y[ids]
            lm_fit = LinearRegression(fit_intercept=False)
            lm_model_new = lm_fit.fit(X=Xnew, y=ynew)
            rho_new = 1 / lm_model_new.coef_
            predDropoutsNew = (rho_new / (mus + rho_new)) ** rho_new
            predDropoutsNew = predDropoutsNew.reshape(-1)
            preds.append(predDropoutsNew)
    else:
        for i in tqdm(range(nper)):
            ids = np.random.choice(nspot, nspot)
            Xmat = X[ids,:]
            New_mus = Xmat.mean(axis=0)
            New_vars = Xmat.var(axis=0)

            New_ys = New_vars-New_mus
            New_mus2 = New_mus**2
            New_xx = New_mus2.reshape(len(mus2), 1)
            New_y=New_ys.reshape(len(New_ys), 1)
            lm_fit = LinearRegression(fit_intercept=False)
            lm_model_new = lm_fit.fit(X=New_xx, y=New_y)
            rho_new = 1 / lm_model_new.coef_
            predDropoutsNew = (rho_new / (mus + rho_new)) ** rho_new
            predDropoutsNew = predDropoutsNew.reshape(-1)
            preds.append(predDropoutsNew)



    preds = pd.DataFrame(np.array(preds), columns=adata.var_names)
    quantile1 = 1 - quantile
    quantile2 = quantile
    df_quantile = preds.quantile([quantile1, 0.5, quantile2])
    lower_bound = df_quantile.iloc[0, :]
    middle_bound = df_quantile.iloc[1, :]
    upper_bound = df_quantile.iloc[2, :]
    adata.var["mu"] = mus
    adata.var["{:.2f}".format(quantile1)] = lower_bound
    adata.var["{:.2f}".format(0.5)] = middle_bound
    adata.var["{:.2f}".format(quantile2)] = upper_bound
    adata.var['Dropout'] = observedDropouts
    adata.var['FittedDropout'] = predDropouts

    from scipy.optimize import fmin_l_bfgs_b as optim
    def opt_f(params, *args):
        rho_opt = params
        middle_bound = np.array(args[0])
        mus = np.array(args[1])
        cost_res = sum((np.array(middle_bound) - (rho_opt / (mus + rho_opt)) ** rho_opt)**2)
        return cost_res

    rho_opt_res = optim(opt_f, x0 = rho, args=(middle_bound,mus), approx_grad=1)
    rho_opt = rho_opt_res[0]
    ix1 = np.where(observedDropouts > upper_bound)[0]
    ix2 = np.where(observedDropouts < lower_bound)[0]
    print("{} of the genes need to be de-noised.\nAmong, {} needs to be imputated"
          " and {} needs to be downgraded".format(len(ix1) + len(ix2), len(ix2), len(ix1)))
    den_indicator = np.zeros(ngene)
    den_indicator[ix1] = 2
    den_indicator[ix2] = 1
    adata.var["is.DEN"] = den_indicator
    adata.uns["optimized_rho"] = rho_opt

    return adata


if __name__=="__main__":
    adata = sc.read_h5ad("data/IDC.h5ad")
    res = Detect_Genes(adata, permutated="spot")
    res.write_h5ad("data/IDC_detectedd_by_spot.h5ad")
    print(res)