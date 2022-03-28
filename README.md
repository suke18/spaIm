<center> <h2> SpaIm </h2> </center>

-------------------
**SpaIm** is a tool for de-noising gene expressions in the spatial transcriptomics (ST) data. All the copyrights are explained by Kenong Su <kenong.su@pennmedicine.upenn.edu> from [Dr. Li's lab](https://transgen.med.upenn.edu/).


### downsampling
Using Dirichlet Multinomial (DM) Mixture Models for simulation the downsampled gene expression data. The DM parameters are estimated by simple method of moments (MoM) approach. It includes two steps: `draw from the Dirichlet distribution`, and `simulate multinomial distribution with a shrinked N`. The example can be found here: 


### de-noising
Using NB decoder to de-noise the original gene expression matrix.
