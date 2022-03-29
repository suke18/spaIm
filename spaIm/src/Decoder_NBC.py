import torch
from pyro.distributions import GammaPoisson
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def gammapoisson_nll(concentration, rate, y):
    dist = GammaPoisson(concentration=concentration, rate=rate)
    loss = (-1) * dist.log_prob(y)
    return loss.mean()
# n_observations = 100
# n_targets = 1
# y_observed = torch.randint( low=0, high=10, size=(n_observations, n_targets))
# concentration_predicted = torch.exp(torch.randn(n_observations, n_targets))
# rate_predicted = torch.exp(torch.randn(n_observations, n_targets))
# loss = gammapoisson_nll( concentration=concentration_predicted, rate=rate_predicted, y=y_observed)
# print(loss)

class NBR_plnet(pl.LightningModule):
    def __init__(self, n_embed, n_hidden1, n_hidden2, n_output, rho, learning_rate = 1e-5):
        super(NBR_plnet, self).__init__()
        self.learning_rate = learning_rate
        self.hidden1 = torch.nn.Linear(n_embed, n_hidden1)  # hidden layer1
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)  # hidden layer2
        self.predict_con = torch.nn.Linear(n_hidden2, n_output)  # output layer for concentration parameter
        self.predict_rate = torch.nn.Linear(n_hidden2, n_output)  # output layer for rate parameter
        self.rho = rho
        self.w1 = torch.nn.Parameter(torch.ones(n_embed))
        self.w2 = torch.nn.Parameter(torch.ones(n_embed))

    def forward(self, x1, x2):
        x = torch.mul(x1, self.w1) + torch.mul(x2, self.w2)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        con = self.predict_con(x)
        rate = self.predict_rate(x)
        return con, rate

    def training_step(self, batch, batch_idx):
        x1, x2, y = batch
        con, rate = self.forward(x1, x2)
        cal_con = torch.exp(con)
        cal_rate = torch.exp(rate)
        loss1 = gammapoisson_nll(concentration=cal_con, rate=cal_rate, y=y)
        mu = torch.round(cal_con / cal_rate)
        binary_mat=y==0
        binary_mat = binary_mat.type(torch.Tensor)
        dropout = torch.mean(binary_mat, dim=0)
        pred_dropout = torch.pow(self.rho / (torch.mean(mu, dim=0) + self.rho), self.rho)
        # loss2 = F.mse_loss(pred_dropout, dropout)
        loss2 = torch.abs(F.kl_div(pred_dropout, dropout))
        loss = loss1*2 + loss2
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x1, x2, y = batch
        con, rate = self.forward(x1, x2)
        cal_con = torch.exp(con)
        cal_rate = torch.exp(rate)
        loss1 = gammapoisson_nll(concentration=cal_con, rate=cal_rate, y=y)
        mu = torch.round(cal_con / cal_rate)
        binary_mat = y == 0
        binary_mat = binary_mat.type(torch.Tensor)
        dropout = torch.mean(binary_mat, dim=0)
        pred_dropout = torch.pow(self.rho / (torch.mean(mu, dim=0) + self.rho), self.rho)
        # loss2 = F.mse_loss(pred_dropout, dropout)
        loss2 = torch.abs(F.kl_div(pred_dropout, dropout))
        loss = loss1*2 + loss2
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # print(get_lr(optimizer))
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.9, step_size=50)
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, mode="min")
        # scheduler = {"scheduler": lr_scheduler, "reduce_on_plateau": True, "monitor": "val_loss", "patience": 5, "verbose": True}
        # return [optimizer], [scheduler]
        return optimizer

class TrDataset(torch.utils.data.Dataset):
    """ Information about the datasets """
    def __init__(self, X1, X2, Y):
        self.x1 = X1
        self.x2 = X2
        self.y = Y

    def __getitem__(self, index):
        return self.x1[index], self.x2[index], self.y[index]

    def __len__(self):
        return len(self.x1)


if __name__ == '__main__':

    Net = NBR_plnet(n_embed=10000, n_hidden1=128, n_hidden2=64, n_output=10000, rho=2)
    x1 = torch.Tensor(2000, 10000)
    x2 = torch.Tensor(2000, 10000)
    Net(x1, x2)
    gammapoisson_nll(concentration=torch.exp(x1), rate=torch.exp(x2), y = torch.exp(x2+x1))

    import scanpy as sc, numpy as np
    import squidpy as sq
    examp = sq.datasets.visium_hne_adata()
    genes = examp[:, examp.var.highly_variable].var_names.values[:100]
    sq.gr.spatial_neighbors(examp)
    sq.gr.spatial_autocorr(
        examp,
        genes=genes,
        n_perms=100,
        n_jobs=1,
    )
    examp.uns["moranI"].head(10)

    adata = sc.read_h5ad("data/IDC.h5ad")
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    coord_df = np.array([adata.obs["imagerow"], adata.obs["imagecol"]], dtype=float)
    genes = adata.var_names[adata.var["is.HVG"] == 1]


    from src.Moran_I_statistics import *
    genes_exp = pd.DataFrame(adata[:, genes].X)
    genes_exp.columns = genes
    x = coord_df[0]
    y = coord_df[1]
    gene_mis = Moran_I(genes_exp, x, y)
    adata.uns["moran"] = pd.DataFrame({"I": gene_mis})
    adata.write_h5ad("data/IDC2.h5ad")

    down_adata = sc.read_h5ad("data/DownSampling/IDC_down.h5ad")
    sc.pp.normalize_total(down_adata)
    sc.pp.log1p(down_adata)
    down_genes_exp = pd.DataFrame(down_adata[:, genes].X)
    down_genes_exp.columns = genes
    down_gene_mis = Moran_I(down_genes_exp, x, y)
    down_adata.uns["moran"] = pd.DataFrame({"I": down_gene_mis})
    down_adata.write_h5ad("data/DownSampling/IDC_down2.h5ad")


    coord_df = coord_df.T
    adata.obsm['pixel'] = coord_df

    from src.PE_NODE_Embedding import *
    Graph2Node2Vec(adata, embed_size=64)
    adata.write_h5ad("data/DownSampling/IDC_PE_down.h5ad")

    adata = sc.read_h5ad("data/DownSampling/IDC_PE_down.h5ad")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=64)
    pca.fit(adata.X)
    X1 = torch.Tensor(pca.transform(adata.X))
    X2 = torch.Tensor(np.zeros([adata.shape[0], 64]))
    Y = torch.Tensor(adata.X)

    n_sample, n_gene = Y.shape
    batch_size = 128
    Trdataset = TrDataset(X1=torch.Tensor(X1), X2=torch.Tensor(X2), Y=torch.Tensor(Y))
    nval = int(len(Y) * 0.2)
    ntrain = len(Y) - nval
    train_set, val_set = random_split(Trdataset, [ntrain, nval])
    Train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    Val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-3, patience=6, verbose=True, mode="min")
    trainer = pl.Trainer(max_epochs=100, callbacks=[early_stop_callback])
    plmodel = NBR_plnet(n_embed=64, n_hidden1=128, n_hidden2=256, n_output=n_gene, learning_rate=1e-4)
    trainer.fit(plmodel, train_dataloaders=Train_dataloader, val_dataloaders=Val_dataloader)
    plmodel.on_save_checkpoint("data/NBR_checkpoint_only_gene.pt")

    num_params = sum(param.numel() for param in plmodel.parameters())

    for parameter in plmodel.parameters():
        print(parameter.shape)


    torch.load("data/N")
    pred_con, pred_rate = plmodel(X1, X2)

    ypreds = torch.exp(pred_con) / torch.exp(pred_rate)
    ypreds = ypreds.detach().numpy()
    ypreds_adata = sc.AnnData(ypreds)
    ypreds_adata.obs = adata.obs
    ypreds_adata.var = adata.var
    ypreds_adata.write("/Users/kenongsu/Dropbox/Mingyao/spaIm/data/NBR_denoised_by_only_gene.h5ad")
    np.savez_compressed(NBR=ypreds, file="/Users/kenongsu/Dropbox/Mingyao/spaIm/data/NBR_preds.npz")

