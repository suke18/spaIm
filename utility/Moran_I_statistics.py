import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import numba
from src.Read_data import read_HER2ST
import os
import matplotlib.pyplot as plt

@numba.njit("f4(f4[:], f4[:])")
def euclid_dist(t1, t2):
    sum = 0
    for i in range(t1.shape[0]):
        sum += (t1[i] - t2[i]) ** 2
    return np.sqrt(sum)


@numba.njit("f4[:,:](f4[:,:])", parallel=True, nogil=True)
def pairwise_distance(X):
    n = X.shape[0]
    adj = np.empty((n, n), dtype=np.float32)
    for i in numba.prange(n):
        for j in numba.prange(n):
            adj[i][j] = euclid_dist(X[i], X[j])
    return adj


def calculate_adj_matrix(x, y, x_pixel=None, y_pixel=None, image=None, beta=49, alpha=1, histology=True):
    # x,y,x_pixel, y_pixel are lists
    if histology:
        assert (x_pixel is not None) & (x_pixel is not None) & (image is not None)
        assert (len(x) == len(x_pixel)) & (len(y) == len(y_pixel))
        print("Calculateing adj matrix using histology image...")
        # beta to control the range of neighbourhood when calculate grey vale for one spot
        # alpha to control the color scale
        beta_half = round(beta / 2)
        g = []
        for i in range(len(x_pixel)):
            max_x = image.shape[0]
            max_y = image.shape[1]
            nbs = image[max(0, x_pixel[i] - beta_half):min(max_x, x_pixel[i] + beta_half + 1),
                  max(0, y_pixel[i] - beta_half):min(max_y, y_pixel[i] + beta_half + 1)]
            g.append(np.mean(np.mean(nbs, axis=0), axis=0))
        c0, c1, c2 = [], [], []
        for i in g:
            c0.append(i[0])
            c1.append(i[1])
            c2.append(i[2])
        c0 = np.array(c0)
        c1 = np.array(c1)
        c2 = np.array(c2)
        print("Var of c0,c1,c2 = ", np.var(c0), np.var(c1), np.var(c2))
        c3 = (c0 * np.var(c0) + c1 * np.var(c1) + c2 * np.var(c2)) / (np.var(c0) + np.var(c1) + np.var(c2))
        c4 = (c3 - np.mean(c3)) / np.std(c3)
        z_scale = np.max([np.std(x), np.std(y)]) * alpha
        z = c4 * z_scale
        z = z.tolist()
        print("Var of x,y,z = ", np.var(x), np.var(y), np.var(z))
        X = np.array([x, y, z]).T.astype(np.float32)
    else:
        print("Calculateing adj matrix using xy only...")
        X = np.array([x, y]).T.astype(np.float32)
    return pairwise_distance(X)


def Moran_I(genes_exp, x, y, k=10, knn=True):
    XY_map = pd.DataFrame({"x": x, "y": y})
    if knn:
        XYnbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean').fit(XY_map)
        XYdistances, XYindices = XYnbrs.kneighbors(XY_map)
        W = np.zeros((genes_exp.shape[0], genes_exp.shape[0]))
        for i in range(0, genes_exp.shape[0]):
            W[i, XYindices[i, :]] = 1
        for i in range(0, genes_exp.shape[0]):
            W[i, i] = 0
    else:
        W = calculate_adj_matrix(x=x, y=y, histology=False)
    I = pd.Series(index=genes_exp.columns, dtype="float64")
    from tqdm import tqdm
    for k in tqdm(genes_exp.columns):
        X_minus_mean = np.array(genes_exp[k] - np.mean(genes_exp[k]))
        X_minus_mean = np.reshape(X_minus_mean, (len(X_minus_mean), 1))
        Nom = np.sum(np.multiply(W, np.matmul(X_minus_mean, X_minus_mean.T)))
        Den = np.sum(np.multiply(X_minus_mean, X_minus_mean))
        I[k] = (len(genes_exp[k]) / np.sum(W)) * (Nom / Den)
    return I


def Geary_C(genes_exp, x, y, k=5, knn=True):
    XYmap = pd.DataFrame({"x": x, "y": y})
    if knn:
        XYnbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean').fit(XYmap)
        XYdistances, XYindices = XYnbrs.kneighbors(XYmap)
        W = np.zeros((genes_exp.shape[0], genes_exp.shape[0]))
        for i in range(0, genes_exp.shape[0]):
            W[i, XYindices[i, :]] = 1
        for i in range(0, genes_exp.shape[0]):
            W[i, i] = 0
    else:
        W = calculate_adj_matrix(x=x, y=y, histology=False)
    C = pd.Series(index=genes_exp.columns, dtype="float64")
    for k in genes_exp.columns:
        X = np.array(genes_exp[k])
        X_minus_mean = X - np.mean(X)
        X_minus_mean = np.reshape(X_minus_mean, (len(X_minus_mean), 1))
        Xij = np.array([X, ] * X.shape[0]).transpose() - np.array([X, ] * X.shape[0])
        Nom = np.sum(np.multiply(W, np.multiply(Xij, Xij)))
        Den = np.sum(np.multiply(X_minus_mean, X_minus_mean))
        C[k] = (len(genes_exp[k]) / (2 * np.sum(W))) * (Nom / Den)
    return C


if __name__ == "__main__":
    exprs_path = "simProjects/her2st-master/data/ST-cnts"
    coords_path = "simProjects/her2st-master/data/ST-spotfiles"
    ## Let check A patient first

    ids = ["A1", "B1", "C1", "D1", "E1", "F1", "G1", "H1"]
    for id in ids:
        print(id)
        slice_path = os.path.join(exprs_path, id+".tsv.gz")
        coord_path = os.path.join(coords_path, id+"_selection.tsv")
        slice = read_HER2ST(exprs_path=slice_path, coords_path=coord_path, process_sc=False, subseting=True)
        slice_Spots = slice.obs.index

        genes_exp = slice.X
        genes_exp = pd.DataFrame(genes_exp)

        x = slice.obsm['pixel'][:, 0]
        y = slice.obsm['pixel'][:, 1]
        gene_mis = Moran_I(genes_exp, x, y)

        npz_path = os.path.join("tests/HER2ST/", id, "ViT_B_16_features.npz")
        npz = np.load(npz_path)
        files = npz.files
        img_mat = npz[files[0]]
        spots = npz[files[1]]

        slice_spots = slice.obs.index.to_list()
        spots = spots.tolist()
        mixs = [spots.index(i) for i in slice_spots]
        img_exp = img_mat[mixs, :]
        img_exp = pd.DataFrame(img_exp)
        img_exp.index = slice.obs.index
        img_mis = Moran_I(img_exp, x, y)


        my_dict = {'Image': img_mis, 'Gene': gene_mis}
        fig, ax = plt.subplots()
        ax.boxplot(my_dict.values())
        ax.set_xticklabels(my_dict.keys())
        ax.set_title(id)
        plt.show()


    ### For the LIBD
    ids1 = np.array(range(151669, 151677))
    ids2 = np.array(range(151507, 151511))
    ids = np.append(ids1, ids2)

    ids = [str(id) for id in ids]
    from src.Read_data import read_LIBD

    for id in ids:
        print(id)
        h5ad_path = "data/LIBD/LIBD_" + id + '.h5ad'
        slice = read_LIBD(h5ad_path, process_sc=False, subseting=True)
        genes_exp = slice.X.A
        genes_exp = pd.DataFrame(genes_exp)

        x = slice.obsm['pixel'][:, 0]
        y = slice.obsm['pixel'][:, 1]
        gene_mis = Moran_I(genes_exp, x, y, k=10)

        ResNet_path = os.path.join("tests/LIBD", id, "ResNet.npz")
        npz = np.load(ResNet_path)
        files = npz.files
        img_mat = npz[files[0]]
        spots = npz[files[1]]

        slice_spots = slice.obs.index.to_list()
        spots = spots.tolist()
        mixs = [spots.index(i) for i in slice_spots]
        img_exp = img_mat[mixs, :]
        img_exp = pd.DataFrame(img_exp)
        img_exp.index = slice.obs.index
        img_mis = Moran_I(img_exp, x, y, k=10)

        my_dict = {'Image': img_mis, 'Gene': gene_mis}
        fig, ax = plt.subplots()
        ax.boxplot(my_dict.values())
        ax.set_xticklabels(my_dict.keys())
        ax.set_title(id)
        plt.show()



    ''' for 151669 sample
    '''
    import scanpy as sc
    import pandas as pd
    libd = sc.read_h5ad("tests/LIBD/151673/151673_ViT_ResNet_VGG_Anno.h5ad")
    sc.pp.highly_variable_genes(libd, n_top_genes=500)
    obsm = libd.obsm.items()
    x = libd.obsm['pixel'][:, 0]
    y = libd.obsm['pixel'][:, 1]
    # genes_exp = libd.X.A
    # genes_exp = libd.X.A[:, libd.var['highly_variable']]
    genes_exp = pd.read_csv("tests/LIBD/151673/geneMatrixTop.csv")
    # genes_exp = pd.DataFrame(genes_exp)
    gene_mis = Moran_I(genes_exp, x, y, k=10)
    mis_res = pd.DataFrame({"Gene": gene_mis})
    obsNames = ["ViT","ResNet","VGG"]
    for obs in obsNames:
        print(obs)
        imgMat = libd.obsm[obs]
        imgMat = pd.DataFrame(imgMat)
        Mis = Moran_I(imgMat, x, y, k=10)
        tmp_df = pd.DataFrame({obs: Mis.tolist()})
        mis_res = pd.concat([mis_res, tmp_df], ignore_index=True, axis=1)

    mis_res.to_csv("tests/LIBD/151673/151673_MI_statistics.csv")
    import scipy.stats as stats

    ''' for HERST gene expression predictions'''
    adata = sc.read_h5ad("tests/HER2ST/allData_ann.h5ad")
    testSamples = ["A1","B1", "C1", "D1", "E1", "F1", "G1", "H1"]
    for samp in testSamples:
        samp_path = os.path.join("tests/HER2ST/top1000HighExpresGenes/", samp) + "_HOPE2Net_predicted_top_1000_genes.npz"
        npz = np.load(samp_path)
        Ypreds = pd.DataFrame(npz['Ypreds'])
        Ytests = pd.DataFrame(npz['testY'])
        genes = npz['genes']
        sampleNames = npz['sampleNames']
        sampleids = [i for i, v in enumerate(adata.obs['samples']) if v == samp]
        oneData = adata[sampleids, :]
        x = oneData.obsm["pixel"][:, 0]
        y = oneData.obsm["pixel"][:, 1]
        mis_preds = Moran_I(Ypreds, x, y,  k =10)
        mis_tests = Moran_I(Ytests, x, y,  k =10)
        mis_res = pd.DataFrame({'preds': mis_preds.tolist(), 'tests': mis_tests.tolist()})
        save_path = "tests/HER2ST/MI_res/" + samp + "_top_1000_gene_MI.csv"
        mis_res.to_csv(save_path)



    ''' for LIBD (151676) gene expression predictions'''
    adata = sc.read_h5ad("tests/LIBD/LIBD_allData_ann.h5ad")
    ids1 = np.array(range(151669, 151677))
    ids2 = np.array(range(151507, 151511))
    ids = np.append(ids1, ids2)
    ids = [str(id) for id in ids]
    for id in ids:
        id_path = os.path.join("tests/LIBD/top1000HighExpressions/", id) + "_HOPE2Net_predicted_1000_genes.npz"
        npz = np.load(id_path)
        Ypreds = pd.DataFrame(npz['Ypreds'])
        Ytests = pd.DataFrame(npz['testY'])
        genes = npz['genes']
        sampleNames = npz['sampleNames']
        sampleids = [i for i, v in enumerate(adata.obs['samples']) if v == id]
        oneData = adata[sampleids, :]
        x = oneData.obsm["pixel"][:, 0]
        y = oneData.obsm["pixel"][:, 1]
        mis_preds = Moran_I(Ypreds, x, y,  k =10)
        mis_tests = Moran_I(Ytests, x, y,  k =10)
        mis_res = pd.DataFrame({'preds': mis_preds.tolist(), 'tests': mis_tests.tolist()})
        save_path = "tests/LIBD/MI_res/" + id + "_top_1000_gene_MI.csv"
        mis_res.to_csv(save_path)


