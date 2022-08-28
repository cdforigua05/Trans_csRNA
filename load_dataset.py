import scanpy as sc
import numpy as np
import pandas as pd
import os

def load_dataset(args):
    adata = None
    if args.dataset=="10X_PBMC":
        adata = sc.read_10x_mtx(args.data_file, var_names='gene_symbols', cache=True)
        adata.X = adata.X.toarray()
        Y = pd.read_csv(os.path.join(args.data_file, "clusters.csv"))
        Y = np.asarray(Y.Cluster)
        assert adata.X.shape[0] == Y.shape[0], "Size don't match"

        adata.obs["Group"] = Y
    return adata

        

    
    return dataset
