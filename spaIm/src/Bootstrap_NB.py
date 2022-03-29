import torch

from src.get_disp import *
from src.decoder_NBC import *
from src.PE_NODE_Embedding import *
from sklearn.decomposition import PCA


def NormalizeMat(Xmat):
    Means = np.mean(Xmat, axis=0)
    Stds = np.std(Xmat, axis=0)
    NMat = (Xmat - Means) / Stds
    return NMat

def Decoder_NB(adata, min_cells=10, embed_size=64, batch_size=64, n_hidden1=128, n_hidden2=256, detect_outliers = False,
               learning_rate=1e-4, max_epochs=100, min_delta=1e-4, patience=6):
    print("filter out zero expressed genes")
    sc.pp.filter_genes(adata, min_cells = min_cells)
    print("detect the outlier genes")
    adata2 = Detect_Genes(adata)
    # del adata
    print("learn the position embeddings")
    if "PE" not in adata2.obsm.keys():
        coord_df = np.array([adata2.obs["imagerow"], adata2.obs["imagecol"]], dtype=float)
        coord_df = coord_df.T
        adata2.obsm['pixel'] = coord_df
        _, _ = Graph2Node2Vec(adata2, embed_size=embed_size)

    print("prepare the input for de-noising")
    if detect_outliers:
        den_ids = np.where(adata2.var["is.DEN"] !=0)[0]
        den_adata = adata2[:, den_ids]
    else:
        den_ids = np.arange(adata2.shape[1])
        den_adata = adata2

    Ymat = den_adata.X # X must be the count matrix
    Y = torch.IntTensor(Ymat)

    tmpadata = sc.AnnData(Ymat)
    sc.pp.normalize_total(tmpadata)
    sc.pp.log1p(tmpadata)
    pca = PCA(n_components=embed_size)
    pca.fit(tmpadata.X)
    pca_trans = pca.transform(tmpadata.X)
    Norm_X1 = pca_trans
    X1 = torch.FloatTensor(Norm_X1)
    Norm_X2 = den_adata.obsm["PE"]
    X2 = torch.FloatTensor(Norm_X2)
    rho = torch.Tensor(adata2.uns["optimized_rho"])

    n_sample, n_gene = Y.shape
    Trdataset = TrDataset(X1=X1, X2=X2, Y=Y)
    nval = int(len(Y) * 0.2)
    ntrain = len(Y) - nval
    train_set, val_set = random_split(Trdataset, [ntrain, nval])
    Train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    Val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=min_delta, patience=patience, verbose=True, mode="min")
    trainer = pl.Trainer(max_epochs=max_epochs, callbacks=[early_stop_callback])
    plmodel = NBR_plnet(n_embed=embed_size, n_hidden1=n_hidden1, n_hidden2=n_hidden2, n_output=n_gene, rho=rho, learning_rate=learning_rate)
    trainer.fit(plmodel, train_dataloaders=Train_dataloader, val_dataloaders=Val_dataloader)

    print("calculating denoised matrix")
    pred_con, pred_rate = plmodel(X1, X2)
    ypreds = torch.exp(pred_con) / torch.exp(pred_rate)
    ypreds = ypreds.detach().numpy()
    adata2[:, den_ids] = np.round(ypreds)

    return adata2


if __name__ == '__main__':
    adata = sc.read_h5ad("data/IDC_with_PE.h5ad")
    adata2 = Decoder_NB(adata)
    adata2.write_h5ad("data/IDC_denoised_by_klv_by_weight.h5ad")




