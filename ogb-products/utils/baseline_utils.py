from torch_geometric.loader import ShaDowKHopSampler
from torch_geometric.data import Data
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_sparse

from .train_utils import row_norm

import math
import numpy as np
import scipy.sparse as sp

import os
import pickle
import time

####################################################################################################
####################################################################################################
####################################################################################################


def get_2_hop_neighbors(data, args):
    #################################################
    data_lite = Data(adj_t=data.adj_t, node_index=data.node_index)
    data_loader = ShaDowKHopSampler(data_lite, 
                                    depth         = 2, 
                                    num_neighbors = args.hop_neighbors, 
                                    batch_size    = args.batch_size, 
                                    num_workers   = 10, 
                                    shuffle       = False)
    #################################################
    batch_all = []
    root_n_id_all = []
    node_index_all = []
    adj_t_row = []
    adj_t_col = []

    num_iters = len(data_loader)
    pbar = tqdm(total=num_iters)
    pbar.set_description('Fetch subgraphs')

    for _, subgraph_data in enumerate(data_loader):
        batch_all.append(subgraph_data.batch.numpy().astype(np.int32))
        root_n_id_all.append(subgraph_data.root_n_id.numpy().astype(np.int32))
        node_index_all.append(subgraph_data.node_index.numpy().astype(np.int32))
        adj_t_row.append(subgraph_data.adj_t.storage.row().numpy().astype(np.int32))
        adj_t_col.append(subgraph_data.adj_t.storage.col().numpy().astype(np.int32))
        pbar.update(1)
    pbar.close()
    
    return batch_all, root_n_id_all, node_index_all, adj_t_row, adj_t_col


####################################################################################################
####################################################################################################
####################################################################################################

def compute_node_features(batch_all, root_n_id_all, node_index_all, adj_t_row, adj_t_col, X_all, Y_all):
    node_feats = []
    
    X_all_cuda = X_all.cuda()
    Y_all_cuda = Y_all.cuda()
    
    num_iters = len(batch_all)
    pbar = tqdm(total=num_iters)
    pbar.set_description('Compute features')
    
    for batch, root_n_id, node_index, row, col in zip(batch_all, root_n_id_all, node_index_all, adj_t_row, adj_t_col):
        
        row = torch.from_numpy(row).to(torch.long)
        col = torch.from_numpy(col).to(torch.long)
        N = len(batch)
        
        subgraph_adj_norm = torch_sparse.SparseTensor(row=row, col=col, sparse_sizes=(N, N), is_sorted=False).cuda()        
        subgraph_adj_norm = row_norm(subgraph_adj_norm)
        
        X = X_all_cuda[node_index]
        Y = Y_all_cuda[node_index]
        Y[root_n_id] = 0
        
        AX  = subgraph_adj_norm.spmm(X)
        A2X = subgraph_adj_norm.spmm(AX)
        A3X = subgraph_adj_norm.spmm(A2X)
        
        AY  = subgraph_adj_norm.spmm(Y)
        A2Y = subgraph_adj_norm.spmm(AY)
        A3Y = subgraph_adj_norm.spmm(A2Y)
        A4Y = subgraph_adj_norm.spmm(A3Y)
        A5Y = subgraph_adj_norm.spmm(A4Y)

        node_feats.append(torch.cat([X + AX + A2X + A3X, AY + A2Y + A3Y + A4Y + A5Y], dim=1)[root_n_id].cpu())
        
        pbar.update(1)
    pbar.close()   
    return torch.cat(node_feats, dim=0)
##################################################################################################
##################################################################################################
##################################################################################################
def get_neighbors(batch_all, root_n_id_all, node_index_all, num_nodes):
    root_nodes = []
    neighbor_nodes = []
    for batch, root_n_id, node_index in zip(batch_all, root_n_id_all, node_index_all):
        root_nodes.append(node_index[root_n_id][batch])
        neighbor_nodes.append(node_index)

    row = np.concatenate(root_nodes)
    col = np.concatenate(neighbor_nodes)
    subgraph_adj = sp.csr_matrix(
        (np.ones_like(row, dtype=np.int16), (row, col)), shape=(num_nodes, num_nodes)
    )
    subgraph_adj.setdiag(0)
    subgraph_adj.eliminate_zeros()

    #############################################
    neighbor_nodes = []
    pbar = tqdm(total=num_nodes)
    pbar.set_description("Compute neighbors")

    for node_i, neighbors_with_i in enumerate(subgraph_adj):
        neighbors_without_i = neighbors_with_i.indices
        neighbor_nodes.append(np.concatenate([np.array([node_i]), neighbors_without_i]))
        pbar.update(1)
    pbar.close()
    #############################################
    subgraph_adj = subgraph_adj.transpose()
    subgraph_relation = []

    pbar = tqdm(total=num_nodes)
    pbar.set_description("Compute subgraph")

    for node_i, neighbors_with_i in enumerate(subgraph_adj):
        neighbors_without_i = neighbors_with_i.indices
        subgraph_relation.append(neighbors_without_i)
        pbar.update(1)
    pbar.close()
    #############################################
    return neighbor_nodes, subgraph_relation

####################################################################################################
####################################################################################################
####################################################################################################



class GNN(nn.Module):
    def __init__(self, W):
        super(GNN, self).__init__()
        self.W = torch.nn.Parameter(W)

    def forward(self, X):
        return X @ self.W

    def ovr_lr_loss(self, X, Y, lam):
        Y[Y == 0] = -1
        Z = (X @ self.W).mul_(Y)
        return -F.logsigmoid(Z).mean(0).sum() + lam * self.W.pow(2).sum() / 2


####################################################################################################
####################################################################################################
####################################################################################################
def lr_grad(w, X, y, lam=0):
    y[y == 0] = -1

    z = torch.sigmoid(y * X.mv(w))
    return X.t().mv((z - 1) * y) + lam * X.size(0) * w


def lr_hessian_inv(w, X, y, lam=0, batch_size=50000):
    y[y == 0] = -1

    z = torch.sigmoid(X.mv(w).mul_(y))
    D = z * (1 - z)
    H = None
    num_batch = int(math.ceil(X.size(0) / batch_size))
    for i in range(num_batch):
        lower = i * batch_size
        upper = min((i + 1) * batch_size, X.size(0))
        X_i = X[lower:upper]
        if H is None:
            H = X_i.t().mm(D[lower:upper].unsqueeze(1) * X_i)
        else:
            H += X_i.t().mm(D[lower:upper].unsqueeze(1) * X_i)

    H = H + lam * X.size(0) * torch.eye(X.size(1)).float()

    try:
        H_inv = torch.linalg.inv(H)
    except:
        H_inv = torch.linalg.pinv(H)

    return H_inv

####################################################################################################
####################################################################################################
####################################################################################################

def get_graph_partitions(num_clusters, data, regen=False):

    save_parts_path = os.path.join('%d_clusters.pkl'%(num_clusters))

    if os.path.exists(save_parts_path) and not regen:
        parts = pickle.load(open(save_parts_path, 'rb'))
    else:
        # convert PyG data structure to scipy's sparse_coo for metis partition
        row = data.adj_t.storage.row().numpy()
        col = data.adj_t.storage.col().numpy()
        num_nodes = data.adj_t.size(0)

        all_nodes = np.arange(num_nodes)
        sparse_coo_adj = sp.csr_matrix((np.ones_like(row), (row, col)), shape=(num_nodes, num_nodes))

        parts = _partition_graph(sparse_coo_adj, all_nodes, num_clusters)

        with open(save_parts_path, 'wb') as f:
            pickle.dump(parts, f)
            
    return parts

def _partition_graph(adj, idx_nodes, num_clusters):
    os.environ['METIS_DLL'] = '/home/weilin/Downloads/metis-5.1.0/build/Linux-x86_64/libmetis/libmetis.so'
    import metis
    
    """partition a graph by METIS."""

    start_time = time.time()
    num_nodes = len(idx_nodes)

    train_adj = adj[idx_nodes, :][:, idx_nodes]
    train_adj_lil = train_adj.tolil()
    train_ord_map = dict()
    train_adj_lists = [[] for _ in range(num_nodes)]
    for i in range(num_nodes):
        rows = train_adj_lil[i].rows[0]
        # self-edge needs to be removed for valid format of METIS
        if i in rows:
            rows.remove(i)
        train_adj_lists[i] = rows
        train_ord_map[idx_nodes[i]] = i

    if num_clusters > 1:
        _, groups = metis.part_graph(train_adj_lists, num_clusters, seed=1)
    else:
        groups = [0] * num_nodes

    parts = [[] for _ in range(num_clusters)]
    for nd_idx in range(num_nodes):
        gp_idx = groups[nd_idx]
        nd_orig_idx = idx_nodes[nd_idx]
        parts[gp_idx].append(nd_orig_idx)
        

    part_size = [len(part) for part in parts]
    print('Partitioning done. %f seconds.'%(time.time() - start_time))
    print('Max part size %d, min part size %d'%(max(part_size), min(part_size)))

    return parts