import os
import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import scanpy as sc
import time
import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.data as data
from scipy.linalg import norm
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import seaborn as sn
import pandas as pd

def cluster_acc(y_true, y_pred, root=None):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    dict_ind = {}
    if root is not None:
        df_cm = pd.DataFrame(w, index = [i for i in range(D)], columns = [i for i in range(D)])
        plt.figure(figsize = (10,7))
        sn.heatmap(df_cm, annot=True)
        plt.savefig(os.path.join(root,"heatmap_unorganized.png"))
        w_order = np.zeros((D, D), dtype=np.int64)
        for i in range(D):
            for j in range(D):
                w_order[i,j] = w[i, ind[1][j]]

        df_cm = pd.DataFrame(w_order, index = [i for i in range(D)], columns = [i for i in ind[1]])
        plt.figure(figsize = (10,7))
        sn.heatmap(df_cm, annot=True, fmt='g')
        plt.ylabel("Prediction")
        plt.xlabel("Ground Truth")
        plt.savefig(os.path.join(root,"heatmap_organized.png"))
    return sum([w[i, j] for i, j in zip(ind[0], ind[1])]) * 1.0 / y_pred.size


def generate_random_pair(y, label_cell_indx, num, error_rate=0):
    """
    Generate random pairwise constraints.
    """
    ml_ind1, ml_ind2 = [], []
    cl_ind1, cl_ind2 = [], []
    y = np.array(y)

    def check_ind(ind1, ind2, ind_list1, ind_list2):
        for (l1, l2) in zip(ind_list1, ind_list2):
                if ind1 == l1 and ind2 == l2:
                    return True
        return False

    error_num = 0
    num0 = num
    while num > 0:
        tmp1 = random.choice(label_cell_indx)
        tmp2 = random.choice(label_cell_indx)
        if tmp1 == tmp2:
            continue
        if check_ind(tmp1, tmp2, ml_ind1, ml_ind2):
            continue
        if y[tmp1] == y[tmp2]:
            if error_num >= error_rate*num0:
                ml_ind1.append(tmp1)
                ml_ind2.append(tmp2)
            else:
                cl_ind1.append(tmp1)
                cl_ind2.append(tmp2)
                error_num += 1
        else:
            if error_num >= error_rate*num0:
                cl_ind1.append(tmp1)
                cl_ind2.append(tmp2)
            else:
                ml_ind1.append(tmp1)
                ml_ind2.append(tmp2) 
                error_num += 1               
        num -= 1
    ml_ind1, ml_ind2, cl_ind1, cl_ind2 = np.array(ml_ind1), np.array(ml_ind2), np.array(cl_ind1), np.array(cl_ind2)
    ml_index = np.random.permutation(ml_ind1.shape[0])
    cl_index = np.random.permutation(cl_ind1.shape[0])
    ml_ind1 = ml_ind1[ml_index]
    ml_ind2 = ml_ind2[ml_index]
    cl_ind1 = cl_ind1[cl_index]
    cl_ind2 = cl_ind2[cl_index]
    return ml_ind1, ml_ind2, cl_ind1, cl_ind2, error_num


def generate_random_pair_from_proteins(latent_embedding, num, ML=0.1, CL=0.9):
    """
    Generate random pairwise constraints.
    """
    ml_ind1, ml_ind2 = [], []
    cl_ind1, cl_ind2 = [], []

    def check_ind(ind1, ind2, ind_list1, ind_list2):
        for (l1, l2) in zip(ind_list1, ind_list2):
                if ind1 == l1 and ind2 == l2:
                    return True
        return False

    latent_dist = euclidean_distances(latent_embedding, latent_embedding)
    latent_dist_tril = np.tril(latent_dist, -1)
    latent_dist_vec = latent_dist_tril.flatten()
    latent_dist_vec = latent_dist_vec[latent_dist_vec>0]
    cutoff_ML = np.quantile(latent_dist_vec, ML)
    cutoff_CL = np.quantile(latent_dist_vec, CL)

    while num > 0:
        tmp1 = random.randint(0, latent_embedding.shape[0] - 1)
        tmp2 = random.randint(0, latent_embedding.shape[0] - 1)
        if tmp1 == tmp2:
            continue
        if check_ind(tmp1, tmp2, ml_ind1, ml_ind2):
            continue
        if norm(latent_embedding[tmp1] - latent_embedding[tmp2], 2) < cutoff_ML:
            ml_ind1.append(tmp1)
            ml_ind2.append(tmp2)
        elif norm(latent_embedding[tmp1] - latent_embedding[tmp2], 2) > cutoff_CL:
            cl_ind1.append(tmp1)
            cl_ind2.append(tmp2)
        else:
            continue
        num -= 1
    ml_ind1, ml_ind2, cl_ind1, cl_ind2 = np.array(ml_ind1), np.array(ml_ind2), np.array(cl_ind1), np.array(cl_ind2)
    ml_index = np.random.permutation(ml_ind1.shape[0])
    cl_index = np.random.permutation(cl_ind1.shape[0])
    ml_ind1 = ml_ind1[ml_index]
    ml_ind2 = ml_ind2[ml_index]
    cl_ind1 = cl_ind1[cl_index]
    cl_ind2 = cl_ind2[cl_index]
    return ml_ind1, ml_ind2, cl_ind1, cl_ind2


def generate_random_pair_from_CD_markers(markers, num, low1=0.4, high1=0.6, low2=0.2, high2=0.8):
    """
    Generate random pairwise constraints.
    """
    ml_ind1, ml_ind2 = [], []
    cl_ind1, cl_ind2 = [], []

    def check_ind(ind1, ind2, ind_list1, ind_list2):
        for (l1, l2) in zip(ind_list1, ind_list2):
                if ind1 == l1 and ind2 == l2:
                    return True
        return False

    gene_low1 = np.quantile(markers[0], low1)
    gene_high1 = np.quantile(markers[0], high1)
    gene_low2 = np.quantile(markers[1], low1)
    gene_high2 = np.quantile(markers[1], high1)

    gene_low1_ml = np.quantile(markers[0], low2)
    gene_high1_ml = np.quantile(markers[0], high2)
    gene_low2_ml = np.quantile(markers[1], low2)
    gene_high2_ml = np.quantile(markers[1], high2)
    gene_low3 = np.quantile(markers[2], low2)
    gene_high3 = np.quantile(markers[2], high2)
    gene_low4 = np.quantile(markers[3], low2)
    gene_high4 = np.quantile(markers[3], high2)

    while num > 0:
        tmp1 = random.randint(0, markers.shape[1] - 1)
        tmp2 = random.randint(0, markers.shape[1] - 1)
        if tmp1 == tmp2:
            continue
        if check_ind(tmp1, tmp2, ml_ind1, ml_ind2):
            continue
        if markers[0, tmp1] < gene_low1 and markers[1, tmp1] > gene_high2 and markers[0, tmp2] > gene_high1 and markers[1, tmp2] < gene_low2:
            cl_ind1.append(tmp1)
            cl_ind2.append(tmp2)
        elif markers[0, tmp2] < gene_low1 and markers[1, tmp2] > gene_high2 and markers[0, tmp1] > gene_high1 and markers[1, tmp1] < gene_low2:
            cl_ind1.append(tmp1)
            cl_ind2.append(tmp2)
        elif markers[1, tmp1] > gene_high2_ml and markers[2, tmp1] > gene_high3 and markers[1, tmp2] > gene_high2_ml and markers[2, tmp2] > gene_high3:
            ml_ind1.append(tmp1)
            ml_ind2.append(tmp2)
        elif markers[1, tmp1] > gene_high2_ml and markers[2, tmp1] < gene_low3 and markers[1, tmp2] > gene_high2_ml and markers[2, tmp2] < gene_low3:
            ml_ind1.append(tmp1)
            ml_ind2.append(tmp2)
        elif markers[0, tmp1] > gene_high1_ml and markers[2, tmp1] > gene_high3 and markers[1, tmp2] > gene_high1_ml and markers[2, tmp2] > gene_high3:
            ml_ind1.append(tmp1)
            ml_ind2.append(tmp2)
        elif markers[0, tmp1] > gene_high1_ml and markers[2, tmp1] < gene_low3 and markers[3, tmp1] > gene_high4 and markers[1, tmp2] > gene_high1_ml and markers[2, tmp2] < gene_low3 and markers[3, tmp2] > gene_high4:
            ml_ind1.append(tmp1)
            ml_ind2.append(tmp2)
        elif markers[0, tmp1] > gene_high1_ml and markers[2, tmp1] < gene_low3 and markers[3, tmp1] < gene_low4 and markers[1, tmp2] > gene_high1_ml and markers[2, tmp2] < gene_low3 and markers[3, tmp2] < gene_low4:
            ml_ind1.append(tmp1)
            ml_ind2.append(tmp2)
        else:
            continue
        num -= 1
    ml_ind1, ml_ind2, cl_ind1, cl_ind2 = np.array(ml_ind1), np.array(ml_ind2), np.array(cl_ind1), np.array(cl_ind2)
    ml_index = np.random.permutation(ml_ind1.shape[0])
    cl_index = np.random.permutation(cl_ind1.shape[0])
    ml_ind1 = ml_ind1[ml_index]
    ml_ind2 = ml_ind2[ml_index]
    cl_ind1 = cl_ind1[cl_index]
    cl_ind2 = cl_ind2[cl_index]
    return ml_ind1, ml_ind2, cl_ind1, cl_ind2


def generate_random_pair_from_embedding_clustering(latent_embedding, num, n_clusters, ML=0.005, CL=0.8):
    """
    Generate random pairwise constraints.
    """
    ml_ind1, ml_ind2 = [], []
    cl_ind1, cl_ind2 = [], []

    def check_ind(ind1, ind2, ind_list1, ind_list2):
        for (l1, l2) in zip(ind_list1, ind_list2):
                if ind1 == l1 and ind2 == l2:
                    return True
        return False

    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    y_pred = kmeans.fit(latent_embedding).labels_

    latent_dist = euclidean_distances(latent_embedding, latent_embedding)
    latent_dist_tril = np.tril(latent_dist, -1)
    latent_dist_vec = latent_dist_tril.flatten()
    latent_dist_vec = latent_dist_vec[latent_dist_vec>0]
    cutoff_ML = np.quantile(latent_dist_vec, ML)
    cutoff_CL = np.quantile(latent_dist_vec, CL)


    while num > 0:
        tmp1 = random.randint(0, latent_embedding.shape[0] - 1)
        tmp2 = random.randint(0, latent_embedding.shape[0] - 1)
        if tmp1 == tmp2:
            continue
        if check_ind(tmp1, tmp2, ml_ind1, ml_ind2):
            continue
        if y_pred[tmp1]==y_pred[tmp2]:
            ml_ind1.append(tmp1)
            ml_ind2.append(tmp2)
        elif y_pred[tmp1]!=y_pred[tmp2] and norm(latent_embedding[tmp1] - latent_embedding[tmp2], 2) > cutoff_CL:
            cl_ind1.append(tmp1)
            cl_ind2.append(tmp2)
        else:
            continue
        num -= 1
    ml_ind1, ml_ind2, cl_ind1, cl_ind2 = np.array(ml_ind1), np.array(ml_ind2), np.array(cl_ind1), np.array(cl_ind2)
    ml_index = np.random.permutation(ml_ind1.shape[0])
    cl_index = np.random.permutation(cl_ind1.shape[0])
    ml_ind1 = ml_ind1[ml_index]
    ml_ind2 = ml_ind2[ml_index]
    cl_ind1 = cl_ind1[cl_index]
    cl_ind2 = cl_ind2[cl_index]
    return ml_ind1, ml_ind2, cl_ind1, cl_ind2

def heatmap_against(y, y_pred, cbar_kw=None, cbarlabel=""):

    fig, ax = plt.subplots()
    # Plot the heatmap
    im = ax.imshow(harvest)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

def repetir(lista, veces):
    salida = []
    for elemento in lista:
        salida.extend([elemento] * veces)
    return salida

def heatmap_genes(adata, pred, markers = None, groupby=False):
    print("jejejej")
    X = adata.X
    gt = np.array(adata.obs.Group)
    matrix = []
    for i in range(len(X)):
        if gt[i] == 7 and pred[i] == 0:
            for k in range(20):
                matrix.append(repetir(X[i], 1))
    for i in range(500):
        matrix.append(repetir(np.zeros(X.shape[1]),1))
    for i in range(len(X)):
        if pred[i] == 0 and gt[i] == 5:
            for k in range(20):
                matrix.append(repetir(X[i], 1))
    plt.figure(figsize = (16,16))
    plt.matshow(matrix, cmap="seismic")
    plt.savefig("comparacion_0_7_5.png")
    return None

def histogram_weights(att_weights, name=None):
    m, bins, patches = plt.hist(x=att_weights, bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Weight')
    plt.ylabel('Frequency')
    plt.title('Gene weights histogram')
    maxfreq = m.max()
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.savefig(name)

def clustering_metrics(z, clusters):
    shilsilhouette =  silhouette_score(z, clusters, metric="euclidean")
    Bk = calinski_harabasz_score(z, clusters)
    DB = davies_bouldin_score(z, clusters)
    return shilsilhouette, Bk, DB

import json
import functools
import operator
import collections
import jgraph
import numpy as np
import scipy.sparse
import tqdm


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def in_ipynb():  # pragma: no cover
    try:
        # noinspection PyUnresolvedReferences
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True   # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


def smart_tqdm():  # pragma: no cover
    if in_ipynb():
        return tqdm.tqdm_notebook
    return tqdm.tqdm


def with_self_graph(fn):
    @functools.wraps(fn)
    def wrapped(self, *args, **kwargs):
        with self.graph.as_default():
            return fn(self, *args, **kwargs)
    return wrapped


# Wraps a batch function into minibatch version
def minibatch(batch_size, desc, use_last=False, progress_bar=True):
    def minibatch_wrapper(func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            total_size = args[0].shape[0]
            if use_last:
                n_batch = np.ceil(
                    total_size / float(batch_size)
                ).astype(np.int)
            else:
                n_batch = max(1, np.floor(
                    total_size / float(batch_size)
                ).astype(np.int))
            for batch_idx in smart_tqdm()(
                range(n_batch), desc=desc, unit="batches",
                leave=False, disable=not progress_bar
            ):
                start = batch_idx * batch_size
                end = min((batch_idx + 1) * batch_size, total_size)
                this_args = (item[start:end] for item in args)
                func(*this_args, **kwargs)
        return wrapped_func
    return minibatch_wrapper


# Avoid sklearn warning
def encode_integer(label, sort=False):
    label = np.array(label).ravel()
    classes = np.unique(label)
    if sort:
        classes.sort()
    mapping = {v: i for i, v in enumerate(classes)}
    return np.array([mapping[v] for v in label]), classes


# Avoid sklearn warning
def encode_onehot(label, sort=False, ignore=None):
    i, c = encode_integer(label, sort)
    onehot = scipy.sparse.csc_matrix((
        np.ones_like(i, dtype=np.int32), (np.arange(i.size), i)
    ))
    if ignore is None:
        ignore = []
    return onehot[:, ~np.in1d(c, ignore)].tocsr()


class CellTypeDAG(object):

    def __init__(self, graph=None, vdict=None):
        self.graph = jgraph.Graph(directed=True) if graph is None else graph
        self.vdict = {} if vdict is None else vdict

    @classmethod
    def load(cls, file):
        if file.endswith(".json"):
            return cls.load_json(file)
        elif file.endswith(".obo"):
            return cls.load_obo(file)
        else:
            raise ValueError("Unexpected file format!")

    @classmethod
    def load_json(cls, file):
        with open(file, "r") as f:
            d = json.load(f)
        dag = cls()
        dag._build_tree(d)
        return dag

    @classmethod
    def load_obo(cls, file):  # Only building on "is_a" relation between CL terms
        import pronto
        ont = pronto.Ontology(file)
        graph, vdict = jgraph.Graph(directed=True), {}
        for item in ont:
            if not item.id.startswith("CL"):
                continue
            if "is_obsolete" in item.other and item.other["is_obsolete"][0] == "true":
                continue
            graph.add_vertex(
                name=item.id, cell_ontology_class=item.name,
                desc=str(item.desc), synonyms=[(
                    "%s (%s)" % (syn.desc, syn.scope)
                 ) for syn in item.synonyms]
            )
            assert item.id not in vdict
            vdict[item.id] = item.id
            assert item.name not in vdict
            vdict[item.name] = item.id
            for synonym in item.synonyms:
                if synonym.scope == "EXACT" and synonym.desc != item.name:
                    vdict[synonym.desc] = item.id
        for source in graph.vs:
            for relation in ont[source["name"]].relations:
                if relation.obo_name != "is_a":
                    continue
                for target in ont[source["name"]].relations[relation]:
                    if not target.id.startswith("CL"):
                        continue
                    graph.add_edge(
                        source["name"],
                        graph.vs.find(name=target.id.split()[0])["name"]
                    )
                    # Split because there are many "{is_infered...}" suffix,
                    # falsely joined to the actual id when pronto parses the
                    # obo file
        return cls(graph, vdict)

    def _build_tree(self, d, parent=None):  # For json loading
        self.graph.add_vertex(name=d["name"])
        v = self.graph.vs.find(d["name"])
        if parent is not None:
            self.graph.add_edge(v, parent)
        self.vdict[d["name"]] = d["name"]
        if "alias" in d:
            for alias in d["alias"]:
                self.vdict[alias] = d["name"]
        if "children" in d:
            for subd in d["children"]:
                self._build_tree(subd, v)

    def get_vertex(self, name):
        return self.graph.vs.find(self.vdict[name])

    def is_related(self, name1, name2):
        return self.is_descendant_of(name1, name2) \
            or self.is_ancestor_of(name1, name2)

    def is_descendant_of(self, name1, name2):
        if name1 not in self.vdict or name2 not in self.vdict:
            return False
        shortest_path = self.graph.shortest_paths(
            self.get_vertex(name1), self.get_vertex(name2)
        )[0][0]
        return np.isfinite(shortest_path)

    def is_ancestor_of(self, name1, name2):
        if name1 not in self.vdict or name2 not in self.vdict:
            return False
        shortest_path = self.graph.shortest_paths(
            self.get_vertex(name2), self.get_vertex(name1)
        )[0][0]
        return np.isfinite(shortest_path)

    def conditional_prob(self, name1, name2):  # p(name1|name2)
        if name1 not in self.vdict or name2 not in self.vdict:
            return 0
        self.graph.vs["prob"] = 0
        v2_parents = list(self.graph.bfsiter(
            self.get_vertex(name2), mode=jgraph.OUT))
        v1_parents = list(self.graph.bfsiter(
            self.get_vertex(name1), mode=jgraph.OUT))
        for v in v2_parents:
            v["prob"] = 1
        while True:
            changed = False
            for v1_parent in v1_parents[::-1]:  # Reverse may be more efficient
                if v1_parent["prob"] != 0:
                    continue
                v1_parent["prob"] = np.prod([
                    v["prob"] / v.degree(mode=jgraph.IN)
                    for v in v1_parent.neighbors(mode=jgraph.OUT)
                ])
                if v1_parent["prob"] != 0:
                    changed = True
            if not changed:
                break
        return self.get_vertex(name1)["prob"]

    def similarity(self, name1, name2, method="probability"):
        if method == "probability":
            return (
                self.conditional_prob(name1, name2) +
                self.conditional_prob(name2, name1)
            ) / 2
        # if method == "distance":
        #     return self.distance_ratio(name1, name2)
        raise ValueError("Invalid method!")  # pragma: no cover

    def count_reset(self):
        self.graph.vs["raw_count"] = 0
        self.graph.vs["prop_count"] = 0  # count propagated from children
        self.graph.vs["count"] = 0

    def count_set(self, name, count):
        self.get_vertex(name)["raw_count"] = count

    def count_update(self):
        origins = [v for v in self.graph.vs.select(raw_count_gt=0)]
        for origin in origins:
            for v in self.graph.bfsiter(origin, mode=jgraph.OUT):
                if v != origin:  # bfsiter includes the vertex self
                    v["prop_count"] += origin["raw_count"]
        self.graph.vs["count"] = list(map(
            operator.add, self.graph.vs["raw_count"],
            self.graph.vs["prop_count"]
        ))

    def best_leaves(self, thresh, retrieve="name"):
        subgraph = self.graph.subgraph(self.graph.vs.select(count_ge=thresh))
        leaves, max_count = [], 0
        for leaf in subgraph.vs.select(lambda v: v.indegree() == 0):
            if leaf["count"] > max_count:
                max_count = leaf["count"]
                leaves = [leaf[retrieve]]
            elif leaf["count"] == max_count:
                leaves.append(leaf[retrieve])
        return leaves


class DataDict(collections.OrderedDict):

    def shuffle(self, random_state=np.random):
        shuffled = DataDict()
        shuffle_idx = None
        for item in self:
            shuffle_idx = random_state.permutation(self[item].shape[0]) \
                if shuffle_idx is None else shuffle_idx
            shuffled[item] = self[item][shuffle_idx]
        return shuffled

    @property
    def size(self):
        data_size = set([item.shape[0] for item in self.values()])
        assert len(data_size) == 1
        return data_size.pop()

    @property
    def shape(self):  # Compatibility with numpy arrays
        return [self.size]

    def __getitem__(self, fetch):
        if isinstance(fetch, (slice, np.ndarray)):
            return DataDict([
                (item, self[item][fetch]) for item in self
            ])
        return super(DataDict, self).__getitem__(fetch)


def densify(arr):
    if scipy.sparse.issparse(arr):
        return arr.toarray()
    return arr


def empty_safe(fn, dtype):
    def _fn(x):
        if x.size:
            return fn(x)
        return x.astype(dtype)
    return _fn


decode = empty_safe(np.vectorize(lambda _x: _x.decode("utf-8")), str)
encode = empty_safe(np.vectorize(lambda _x: str(_x).encode("utf-8")), "S")
upper = empty_safe(np.vectorize(lambda x: str(x).upper()), str)
lower = empty_safe(np.vectorize(lambda x: str(x).lower()), str)
tostr = empty_safe(np.vectorize(str), str)

    




