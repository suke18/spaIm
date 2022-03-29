#####
import numpy as np
import os
from ge.models import Node2Vec
import networkx as nx
from sklearn.neighbors import NearestNeighbors

def Graph2Node2Vec(slice, knn=10, walk_length=10, num_walks=10, embed_size=768, niter=100):
    '''not using the sc.pp.neighbor for finding the nearest neighbors.
    Instead only using the coordinates for KNN and build graph.
    '''
    coord_df = slice.obsm['pixel']
    coord_df = np.array(coord_df)
    neigh = NearestNeighbors(n_neighbors=knn)
    neigh.fit(coord_df)
    nb_list = neigh.kneighbors(coord_df, return_distance=False)
    sampleNames = slice.obs.index
    pairs = []
    nNodes = len(nb_list)
    for i in range(nNodes):
        node = nb_list[i]
        nodeLen = len(node)
        for j in range(nodeLen):
            if node[j] > i:
                onePair = [sampleNames[i], sampleNames[node[j]]]
                pairs.append(onePair)
    nxEdges = nx.to_networkx_graph(pairs, create_using=nx.DiGraph)
    # use the Node2Vec embeddings
    model = Node2Vec(nxEdges, walk_length=walk_length, num_walks=num_walks,
                     p=0.25, q=4, workers=1, use_rejection_sampling=0)
    model.train(window_size=5, embed_size=embed_size, iter=niter)
    embeddings = model.get_embeddings()

    sampleNames = slice.obs.index.tolist()
    dictKeys = list(embeddings.keys())
    mix = [dictKeys.index(i) for i in sampleNames]
    embed_pd = np.array(list(embeddings.values()))
    embed_pd = embed_pd[mix]
    slice.obsm["PE"] = embed_pd
    return embed_pd, slice


def Com2HistNode(slice):
    def NormalizeMat(Xmat):
        Means = np.mean(Xmat, axis=0)
        Stds = np.std(Xmat, axis=0)
        NMat = (Xmat - Means) / Stds
        return NMat
    slice.obsm['PE+ViT'] = NormalizeMat(slice.obsm["PE"]) + NormalizeMat(slice.obsm["ViT"])


def CFAmetrix(Xmat, Ymat):
    def HSICmetric(Xmat, Ymat):
        '''
        arg Xmat: one matrix with m (samples) x p (features) in numpy array
        arg Ymat: one matrix with m (samples) x q (features) in numpy array
        '''
        if len(Xmat) != len(Ymat):
            import sys
            sys.exit('two matrices are not within the same sample sizes')
        m = len(Xmat)
        Kmat = Xmat.dot(Xmat.T).astype(np.float64)
        Lmat = Ymat.dot(Ymat.T).astype(np.float64)
        Hmat = np.identity(m) - np.ones((m, m)) / m
        Kmat2 = Hmat.dot(Kmat).dot(Hmat)
        Lmat2 = Hmat.dot(Lmat).dot(Hmat)
        HSIC = np.dot(Kmat2.flatten(), Lmat2.flatten()) / ((m - 1) ** 2)
        return HSIC
    d1 = HSICmetric(Xmat, Ymat)
    d2 = (HSICmetric(Xmat, Xmat) * HSICmetric(Ymat, Ymat)) **(1/2)
    return d1/d2



def naive_pearson_cor(X, Y):
    from tqdm import tqdm
    import scipy.stats
    result = np.zeros(shape=(X.shape[1], Y.shape[1]))
    for i in tqdm(range(X.shape[1])):
        for j in range(Y.shape[1]):
            r, _ = scipy.stats.pearsonr(X[:,i], Y[:,j])
            result[i,j] = r
    return result



if __name__=="__main__":
    import scanpy as sc
    from collections import Counter
    adata = sc.read_h5ad("tests/LIBD/LIBD_allData_ann2.h5ad")
    samp_dict = Counter(adata.obs["samples"])
    testSamples = list(samp_dict.keys())
    # select high expression genes
    allMeans = np.zeros((len(testSamples), adata.shape[1]))
    for j in range(len(testSamples)):
       testSample = testSamples[j]
       sampleids = [i for i, v in enumerate(adata.obs['samples']) if v == testSample]
       oneAdata = adata[sampleids, :]
       oneMeans = oneAdata.X.A.mean(axis=0)
       allMeans[j] = oneMeans

    allMeansMs = allMeans.mean(axis=0)
    ixs = np.argsort(allMeansMs)[::-1]
    geneIds = ixs[:1000]

    CKA_dict = {}
    for sample in testSamples:
        print(sample)
        ids = np.where(np.array(adata.obs['samples']) == sample)
        ids = list(ids[0])
        slice = adata[ids, geneIds]

        sc.pp.normalize_total(slice)
        sc.pp.log1p(slice)
        # sc.pp.highly_variable_genes(slice, n_top_genes=2000)
        # hvgs = slice.var.index[slice.var['highly_variable']].copy()
        Xmat = slice.X.A
        CKA1 = CFAmetrix(Xmat, slice.obsm["ViT"])
        CKA2 = CFAmetrix(Xmat, slice.obsm["PE"])
        CKA3 = CFAmetrix(Xmat, slice.obsm["PE+ViT"])
        CKA_dict[sample] = [CKA1, CKA2, CKA3]

    import pickle
    save_path = os.path.join("tests/LIBD", "CKA/") + "LIBD_CKA.pickle"
    with open(save_path, "wb") as f:
        pickle.dump(CKA_dict, f)

    vit_gene_cor = {}
    for sample in testSamples:
        ids = np.where(np.array(adata.obs['samples'] == sample))
        ids = list(ids[0])
        slice = adata[ids, geneIds]
        sc.pp.normalize_total(slice)
        sc.pp.log1p(slice)
        Xmat = slice.X.A
        vit_r = naive_pearson_cor(Xmat, slice.obsm["PE+ViT"])
        vit_gene_cor[sample] = vit_r

    save_path = os.path.join("tests/LIBD", "CKA/") + "LIBD_PE+ViT_1000_gene_cors.pickle"
    with open(save_path, "wb") as f:
        pickle.dump(vit_gene_cor, f)



    # ids = list(range(151669, 151677))
    # ids.extend(list(range(151507, 151511)))
    # ids = [str(id) for id in ids]
    # CKA_dict = {}
    # for id in ids:
    #     print(id)
    #     h5ad_path = os.path.join("tests/LIBD/", id, id) + "_ViT_ResNet_PE_Anno.h5ad"
    #     slice = sc.read_h5ad(h5ad_path)
    #     import torch
    #     sc.pp.normalize_total(slice)
    #     sc.pp.log1p(slice)
    #     sc.pp.highly_variable_genes(slice, n_top_genes=2000)
    #     hvgs = slice.var.index[slice.var['highly_variable']].copy()
    #     Xmat = slice[:, hvgs].X.A
    #     CKA1 = CFAmetrix(Xmat, slice.obsm["ViT"])
    #     CKA2 = CFAmetrix(Xmat, slice.obsm["PE"])
    #     CKA3 = CFAmetrix(Xmat, slice.obsm["PE+ViT"])
    #     CKA_dict[id] = [CKA1, CKA2, CKA3]



    import scanpy as sc
    adata = sc.read_h5ad("/Users/kenongsu/Dropbox/Mingyao/MultiSlices/tests/HER2ST/allData_ann2.h5ad")
    testSamples = ["A1", "C1", "D1", "E1", "F1", "G1", "H1"]
    allMeans = np.zeros((len(testSamples), adata.shape[1]))
    for j in range(len(testSamples)):
        testSample = testSamples[j]
        sampleids = [i for i, v in enumerate(adata.obs['samples']) if v == testSample]
        oneAdata = adata[sampleids, :]
        oneMeans = oneAdata.X.mean(axis=0)
        allMeans[j] = oneMeans

    allMeansMs = allMeans.mean(axis=0)
    ixs = np.argsort(allMeansMs)[::-1]
    geneIds = ixs[:1000]


    samples = ["A1", "A2", "A3", "A4", "A5", "A6",
               "B1", "B2", "B3", "B4", "B5", "B6",
               "C1", "C2", "C3", "C4", "C5", "C6",
               "D1", "D2", "D3", "D4", "D5", "D6",
               "E1", "E2", "E3", "F1", "F2", "F3",
               "G1", "G2", "G3", "H1", "H2", "H3"]

    import pickle
    CKA_dict ={}

    for sample in samples:
        ids = np.where(np.array(adata.obs['samples'] == sample))
        ids = list(ids[0])
        slice = adata[ids, geneIds]

        sc.pp.normalize_total(slice)
        sc.pp.log1p(slice)
        # sc.pp.highly_variable_genes(slice, n_top_genes=2000)
        # hvgs = slice.var.index[slice.var['highly_variable']].copy()
        Xmat = slice.X

        CKA1 = CFAmetrix(Xmat, slice.obsm["ViT"])
        CKA2 = CFAmetrix(Xmat, slice.obsm["PE"])
        CKA3 = CFAmetrix(Xmat, slice.obsm["PE+ViT"])
        CKA_dict[sample] = [CKA1, CKA2, CKA3]

    save_path = os.path.join("tests/HER2ST", "CKA/") + "HER2ST_CKA.pickle"
    with open(save_path, "wb") as f:
        pickle.dump(CKA_dict, f)

    vit_gene_cor = {}
    samples = ["A1", "B1", "C1", "D1", "E1", "F1", "G1", "H1"]
    for sample in samples:
        ids = np.where(np.array(adata.obs['samples'] == sample))
        ids = list(ids[0])
        slice = adata[ids, geneIds]

        sc.pp.normalize_total(slice)
        sc.pp.log1p(slice)
        Xmat = slice.X
        vit_r = naive_pearson_cor(Xmat, slice.obsm["PE+ViT"])
        vit_gene_cor[sample] = vit_r

    save_path = os.path.join("tests/HER2ST", "CKA/") + "HER2ST_PE+ViT_1000_gene_cors.pickle"
    with open(save_path, "wb") as f:
        pickle.dump(vit_gene_cor, f)




    # ids = list(range(151669, 151677))
    # ids.extend(list(range(151507, 151511)))
    # ids = [str(id) for id in ids]
    # for id in ids:
    #     print(id)
    #     h5ad_path = os.path.join("tests/LIBD/", id, id) + "_ViT_ResNet_Anno.h5ad"
    #     slice = sc.read_h5ad(h5ad_path)
    #     import torch
    #     torch.manual_seed(123)
    #     Graph2Node2Vec(slice, niter=1000)
    #     Com2HistNode(slice)
    #
    #     save_file_path = os.path.join("tests/LIBD/", id, id) + "_ViT_ResNet_PE_Anno.h5ad"
    #     slice.write_h5ad(save_file_path)

        # sc.pp.normalize_total(slice)
        # sc.pp.log1p(slice)
        # sc.pp.highly_variable_genes(slice, n_top_genes=2000)
        # hvgs = slice.var.index[slice.var['highly_variable']].copy()
        # Xmat = slice[:, hvgs].X.A
        # print(CFAmetrix(Xmat, slice.obsm["ViT"]))
        # print(CFAmetrix(Xmat, slice.obsm["PE"]))
        # print(CFAmetrix(Xmat, slice.obsm["PE+ViT"]))




# p1 = naive_pearson_cor(Xmat, slice.obsm["ViT"])
# p2 = naive_pearson_cor(Xmat, slice.obsm["PE"])
# p3 = naive_pearson_cor(Xmat, slice.obsm["PE+ViT"])
#
# import scanpy as sc
# slide = sc.read_h5ad("tests/LIBD/151673/151673_ViT_ResNet_Anno.h5ad")
# import torch
# torch.manual_seed(123)
#
# Graph2Node2Vec(slide, niter=1000)
# Com2HistNode(slide)
# sc.pp.highly_variable_genes(slide, n_top_genes =2000)
# slide2 = slide[:, slide.var["highly_variable"]==1]
# # slice.write_h5ad("tests/HER2ST/A1/A1_ViT_ResNet_PE_Anno.h5ad")
# Xmat = slide2.X.A
#
# CFAmetrix(Xmat, slide.obsm["ViT"])
# CFAmetrix(Xmat, slide.obsm["PE"])
# CFAmetrix(Xmat, slide.obsm["PE+ViT"])
#
# p1 = naive_pearson_cor(Xmat, slice.obsm["ViT"])
# p2 = naive_pearson_cor(Xmat, slice.obsm["PE"])
# p3 = naive_pearson_cor(Xmat, slice.obsm["PE+ViT"])
#
#
#
# import scanpy as sc
# slice = sc.read_h5ad("tests/HER2ST/B1/B1_ViT_ResNet_Anno.h5ad")
# Graph2Node2Vec(slice, niter=1000)
# Com2HistNode(slice)
# slice.write_h5ad("tests/HER2ST/B1/B1_ViT_ResNet_PE_Anno.h5ad")
#
#
