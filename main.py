from ast import arg
from json import load
from time import time
import math, os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
from sklearn.cluster import KMeans
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import collections
from sklearn import metrics
import h5py
import scanpy as sc
from preprocess import read_dataset, normalize
import argparse
from load_dataset import load_dataset
from model.select_model import select_model
from utils import cluster_acc, generate_random_pair, heatmap_genes, histogram_weights
import random
import torch
import numpy as np

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)#as reproducibility docs
    torch.manual_seed(seed)# as reproducibility docs
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False# as reproducibility docs
    torch.backends.cudnn.deterministic = True# as reproducibility docs

if __name__ == "__main__":
    seed_torch()
    # setting hyper parameters
    parser = argparse.ArgumentParser(description="Trans scRNA", 
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='10X_PBMC')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--label_cells', default=0.1, type=float)
    parser.add_argument('--label_cells_files', default='label_selected_cells_1.txt')
    parser.add_argument('--data_file', default='./datos/10x PBMC/')
    parser.add_argument('--maxiter', default=2000, type=int)
    parser.add_argument('--pretrain_epochs', default=3000, type=int)
    parser.add_argument('--gamma', default=1., type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--ae_weights', default=None)
    parser.add_argument('--save_dir', default='results/Trans_scRNA/')
    parser.add_argument('--ae_weight_file', default='AE_weights_p0_1.pth.tar')
    parser.add_argument('--name', default='10X_PBCM_scDCCBaseline')

    # Pre-procesing parameters
    parser.add_argument('--top_genes', default=0, type=int, 
                        help='number of selected genes, 0 means using all genes')
    parser.add_argument('--min_counts_cell', default=1, type=int)
    parser.add_argument('--min_count_gene', default=1, type = int)
    parser.add_argument('--n_clusters', default=8, type=int)

    # Model parameters
    parser.add_argument('--model', default='scDCC', help='Model to train')

    #Noise parameters
    parser.add_argument('--sigma', default=2.5, type=int)

    # Attention parameters
    parser.add_argument('--simple_attention', action='store_true')
    parser.add_argument('--complex_attention', action='store_true')
    
    args = parser.parse_args()
    print(args)
    adata = load_dataset(args)
    #top_genes = adata.var["gene_ids"]
    #f = open(os.path.join("results", args.dataset,"total_genes.txt"), "w")
    #for gene in top_genes:
    #    f.write(gene + " \n")
    #f.close()

    # Crear folders necesarios
    if not os.path.exists("./results"):
        os.mkdir("./results")
    if not os.path.exists(os.path.join("results", args.dataset)):
        os.mkdir(os.path.join("results", args.dataset))
    if not os.path.exists(os.path.join("results", args.dataset, args.name)):
        os.mkdir(os.path.join("results", args.dataset, args.name))
        os.mkdir(os.path.join("results", args.dataset, args.name, "checkpoints"))
        os.mkdir(os.path.join("results", args.dataset, args.name, "pretrained"))
    

    # preprocessing scRNA-seq read counts matrix
    adata = read_dataset(adata, transpose=False, test_split=False, copy=True)
    adata = preprocess(adata, args, size_factors=True, normalize_input=True, logtrans_input= True)

    print('### After Preprocessing: {} genes and {} cells.'.format(adata.n_vars, adata.n_obs))
    input_size = adata.n_vars
    if not os.path.exists(os.path.join("results", args.dataset, args.label_cells_files)):
        np.random.seed(0)
        indx = np.arange(len(adata.obs.Group))
        np.random.shuffle(indx)
        label_cell_indx = indx[0:int(np.ceil(args.label_cells*len(adata.obs.Group)))]
    else:
        label_cell_indx = np.loadtxt(os.path.join("results", args.dataset, args.label_cells_files), dtype=np.int)
    
    x_sd = adata.X.std(0) # Desviación estándar de los genes
    x_sd_median = np.median(x_sd) # Desviación media de los genes 
    print("median of gene sd: %.5f" % x_sd_median)
    
    model = select_model(args=args, input_size=input_size)
    # Descomentar para usar PCA
    #sc.tl.pca(adata,  n_comps=32, svd_solver='arpack')
    #X_pca = adata.obsm["X_pca"]
    #print(args.n_clusters)
    #y_pred = KMeans(args.n_clusters, n_init=20, random_state=0).fit_predict(X_pca)
    #eval_cell_y_pred = np.delete(y_pred, label_cell_indx)
    #eval_cell_y = np.delete(np.array(adata.obs.Group), label_cell_indx)
    #acc = np.round(cluster_acc(eval_cell_y, eval_cell_y_pred), 5)
    #nmi = np.round(metrics.normalized_mutual_info_score(eval_cell_y, eval_cell_y_pred), 5)
    #ari = np.round(metrics.adjusted_rand_score(eval_cell_y, eval_cell_y_pred), 5)
    #print('Evaluating cells: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (acc, nmi, ari))

    #adata.write(results_file)
    #sc.pp.neighbors(adata)
    #tl.paga(adata)
    print(str(model))
    t0 = time()
    if args.ae_weights is None:
        model.pretrain_autoencoder(x=adata.X, X_raw=adata.raw.X, size_factor=adata.obs.size_factors, 
                                batch_size=args.batch_size, epochs=args.pretrain_epochs, ae_weights=args.ae_weight_file)
    else:
        if os.path.isfile(args.ae_weights):
            print("==> loading checkpoint '{}'".format(args.ae_weights))
            checkpoint = torch.load(args.ae_weights)
            model.load_state_dict(checkpoint['ae_state_dict'])
        else:
            print("==> no checkpoint found at '{}'".format(args.ae_weights))
            raise ValueError
    if args.simple_attention:
        m =  nn.Sigmoid()
        weights = m(model.att).cpu().detach().numpy()
        histogram_weights(weights, name=os.path.join("results", args.dataset, args.name,"weights_distribution.png"))
        top_weights =  np.argsort(weights)[-100:]
        #top_genes = adata[:,top_weights].var["gene_ids"]
        #f = open(os.path.join("results", args.dataset, args.name, "top_genes.txt"), "w")
        #for gene in top_genes:
        #    f.write(gene + " \n")
        #f.close()
    print('Pretraining time: %d seconds.' % int(time() - t0))
    # TODO: Esto es para scDCC. Hay que adaptar
    if args.model == "scDCC":
        ml_ind1, ml_ind2, cl_ind1, cl_ind2 = np.array([]), np.array([]), np.array([]), np.array([])
        y_pred, _, _, _, _ = model.fit(X=adata.X, X_raw=adata.raw.X, sf=adata.obs.size_factors, y=np.array(adata.obs.Group), batch_size=args.batch_size, num_epochs=args.maxiter, 
                    ml_ind1=ml_ind1, ml_ind2=ml_ind2, cl_ind1=cl_ind1, cl_ind2=cl_ind2,
                    update_interval=args.update_interval, tol=args.tol, save_dir=args.save_dir)
    elif args.model == "SwinIR":
        y_pred, _, _, _, _ = model.fit(X=adata.X, X_raw=adata.raw.X, sf=adata.obs.size_factors, y=np.array(adata.obs.Group), batch_size=args.batch_size,
                                    num_epochs=args.maxiter, update_interval=args.update_interval, tol=args.tol, save_dir=args.save_dir)
        pass
    elif args.model == "scDCCRes":
        ml_ind1, ml_ind2, cl_ind1, cl_ind2 = np.array([]), np.array([]), np.array([]), np.array([])
        y_pred, _, _, _, _ = model.fit(X=adata.X, X_raw=adata.raw.X, sf=adata.obs.size_factors, y=np.array(adata.obs.Group), batch_size=args.batch_size, num_epochs=args.maxiter, 
                    ml_ind1=ml_ind1, ml_ind2=ml_ind2, cl_ind1=cl_ind1, cl_ind2=cl_ind2,
                    update_interval=args.update_interval, tol=args.tol, save_dir=args.save_dir)
    heatmap_genes(adata, y_pred)
    print('Total time: %d seconds.' % int(time() - t0))
    
    eval_cell_y_pred = np.delete(y_pred, label_cell_indx)
    eval_cell_y = np.delete(np.array(adata.obs.Group), label_cell_indx)
    acc = np.round(cluster_acc(eval_cell_y, eval_cell_y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(eval_cell_y, eval_cell_y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(eval_cell_y, eval_cell_y_pred), 5)
    print('Evaluating cells: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (acc, nmi, ari))

    if not os.path.exists(os.path.join("results", args.dataset, args.label_cells_files)):
        np.savetxt(os.path.join("results", args.dataset, args.label_cells_files), label_cell_indx, fmt="%i")

