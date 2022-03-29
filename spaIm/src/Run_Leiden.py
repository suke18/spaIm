import scanpy as sc
import numpy as np
import leidenalg # a prerequiste for running scanpy leiden clustering
import matplotlib.pyplot as plt
import seaborn as sns

def runLeiden(adata, ncluster = 10, nhvg = 2000):

    # preprocessing of the data
    adata.var_names_make_unique()

    # select the hvgs which might be tricky. However, people use this way all the time.
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=nhvg)

    # prepare for leiden clustering
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)

    adata_copy = adata.copy()
    seq_res = np.arange(0.1, 2, 0.1)
    find_n_clusters = []
    for i in range(len(seq_res)):
        tmp_res = seq_res[i]
        tmp_key = "leiden_" + str(i)
        sc.tl.leiden(adata, key_added=tmp_key, resolution=tmp_res)
        find_n_clusters.append(len(set(adata.obs[tmp_key])))

    ix = [i for i, v in enumerate(find_n_clusters) if v == ncluster]

    if len(ix) > 0:
        find_key = "leiden_" + str(ix[0])
        adata_copy.obs["leiden"] = adata.obs[find_key]

    else:
        close_ix = np.argmin(abs(np.array(find_n_clusters) - ncluster))
        close_res = seq_res[close_ix]
        nlen = find_n_clusters[close_ix]
        res_step = 0.01
        runs = 0
        while nlen != ncluster and runs < 40:
            if nlen < ncluster:
                close_res += res_step
                sc.tl.leiden(adata, key_added="leiden_find", resolution=close_res)
            if nlen > ncluster:
                close_res -= res_step
                sc.tl.leiden(adata, key_added="leiden_find", resolution=close_res)
            clusters = adata.obs["leiden_find"]
            nlen = len(set(clusters))
            runs +=1
        adata_copy.obs["leiden"] = adata.obs["leiden_find"]

    return adata_copy




if __name__ == "__main__":
    adata = sc.read_h5ad("data/IDC.h5ad")
    denoised = sc.read_h5ad("data/IDC_original_denoised.h5ad")

    adata = runLeiden(adata)
    denoised = runLeiden(denoised)

    from sklearn.metrics.cluster import adjusted_rand_score
    adjusted_rand_score(denoised.obs["leiden"], denoised.obs["region"])
    adjusted_rand_score(adata.obs["leiden"], adata.obs["region"])

















