import argparse
import copy
import pickle
import numpy as np
import os
import random
import scipy.sparse as sp
from tqdm import tqdm
import time

import torch_sparse
from torch_sparse import SparseTensor
from torch_geometric import seed_everything

import torch
import torch.nn as nn
import torch.nn.functional as F


import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.loader import ShaDowKHopSampler

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from utils.train_utils import pred_test
from utils.graph_projector_model_utils import GNN
from utils.graph_eraser_utils import get_graph_partitions

#################################################
#################################################
#################################################

parser = argparse.ArgumentParser(description="OGBN-Products (GNN)")
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--epochs", type=int, default=8000)
parser.add_argument("--eval_epochs", type=int, default=500)
parser.add_argument("--runs", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)

parser.add_argument("--regen_feats", action="store_true")
parser.add_argument("--regen_model", action="store_true")
parser.add_argument("--regen_partition", action="store_true")
parser.add_argument("--num_remove_nodes", type=float, default=0.02)
parser.add_argument("--parallel_unlearning", type=int, default=1)
parser.add_argument("--num_shards", type=int, default=8)
parser.add_argument("--hop_neighbors", type=int, default=25)
parser.add_argument("--lam", type=float, default=1e-6,
                    help="L2 regularization")

parser.add_argument("--x_iters", type=int, default=3)
parser.add_argument("--y_iters", type=int, default=5)

parser.add_argument("--use_adapt_gcs_x", action="store_true")
parser.add_argument("--use_adapt_gcs_y", action="store_true")
parser.add_argument("--use_cross_entropy", action="store_true")

args = parser.parse_args()
args.use_mlp = False
args.require_linear_span = True
args.use_shallow_sampler = True
args.compress_dim = True

args.use_adapt_gcs = args.use_adapt_gcs_x or args.use_adapt_gcs_y


print(args)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# device = 'cpu'
device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

dataset = PygNodePropPredDataset(
    name="ogbn-products", transform=T.ToSparseTensor(), root="../dataset"
)

data = dataset[0]
split_idx = dataset.get_idx_split()

evaluator = Evaluator(name="ogbn-products")

#################################################
#################################################
#################################################
# Generate augment node feats
num_train_nodes = len(split_idx["train"])
train_ind_perm = np.random.permutation(split_idx["train"])
if args.num_remove_nodes < 1:
    args.num_remove_nodes = int(num_train_nodes * args.num_remove_nodes)
else:
    args.num_remove_nodes = int(args.num_remove_nodes)
print('Remove nodes %d/%d' % (args.num_remove_nodes, num_train_nodes))
delete_nodes_all = train_ind_perm[: args.num_remove_nodes]

extra_feats = torch.zeros(data.x.size(0))
extra_feats[delete_nodes_all] = 1

data.x = torch.cat([data.x, extra_feats.view(-1, 1)], dim=1)
data.y[delete_nodes_all] = dataset.num_classes

shard_nodes = get_graph_partitions(args.num_shards, data, args.regen_partition)
for shard_i, shard_node in enumerate(shard_nodes):
    shard_nodes[shard_i] = np.intersect1d(
        np.array(shard_nodes[shard_i]), np.array(split_idx["train"])
    )
    print("Number of nodes in shard %d is %d" % (shard_i, len(shard_nodes[shard_i])))
    
#################################################
#################################################
#################################################
data.adj_t = torch_sparse.fill_diag(data.adj_t.to_symmetric(), 1)

data.y_one_hot_train = F.one_hot(
    data.y.squeeze(), dataset.num_classes + 1).float()
data.y_one_hot_train[split_idx["test"], :] = 0

num_nodes = data.x.size(0)
data.node_inds = torch.arange(data.x.size(0))

all_loader = ShaDowKHopSampler(data,
                               depth=2,
                               num_neighbors=args.hop_neighbors,
                               batch_size=512,
                               num_workers=10,
                               shuffle=False)
is_train = torch.zeros(num_nodes)
is_train[split_idx['train']] = 1

@torch.no_grad()
def compute_eval_feats(model, all_loader):
    all_feats = []
    pbar = tqdm(total=len(all_loader))
    pbar.set_description('Compute eval feats >>>')
    with torch.no_grad():
        for subgraph_data in all_loader:
            feats = model.gen_feats(subgraph_data.to(device))
            all_feats.append(feats.detach().cpu())
            pbar.update(1)
        pbar.close()
    all_feats = torch.cat(all_feats, dim=0)
    return all_feats

def pre_train(shard_i, print_per_epoch=5):
    # create train loader
    cur_shard_nodes = shard_nodes[shard_i]
    adj_csr = data.adj_t.to_scipy().tocsr()
    adj_t_scipy = adj_csr[cur_shard_nodes, :][:, cur_shard_nodes].tocoo()
    
    subgraph_data = Data(adj_t=SparseTensor(row=torch.tensor(adj_t_scipy.row).long(),
                                            col=torch.tensor(
                                                adj_t_scipy.col).long(),
                                            sparse_sizes=adj_t_scipy.shape),
                         x=data.x[cur_shard_nodes],
                         y=data.y[cur_shard_nodes],
                         y_one_hot_train=data.y_one_hot_train[cur_shard_nodes],
                         node_inds=data.node_inds[cur_shard_nodes])

    node_idx = torch.from_numpy(np.intersect1d(
        split_idx["train"], cur_shard_nodes))
    
    train_loader = ShaDowKHopSampler(subgraph_data,
                                     depth=2,
                                     num_neighbors=args.hop_neighbors,
                                     batch_size=512,
                                     num_workers=10,
                                     shuffle=True,
                                     node_idx=is_train[cur_shard_nodes].bool())

    # training
    best_valid_score = 0
    eval_feat_cached = False
    recycle_steps = 500

    model = GNN(x_dims, h_dims, y_dims, device, args).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(1, 1 + args.epochs):
        seed_everything(args.seed + epoch)

        #######################################################
        # save all grads during training (simulate full-batch --> OGB-products + GD)
        model.train()
        if (epoch - 1) % recycle_steps == 0:
            pbar = tqdm(total=len(train_loader))
            pbar.set_description('Epoch %d' % epoch)

            # create a dict to save gradients
            all_grads = dict()
            for n, p in model.named_parameters():
                all_grads[n] = torch.zeros_like(p.data)

            all_train_feats, all_train_labels = [], []
            for iters, subgraph_data in enumerate(train_loader):
                subgraph_data = subgraph_data.to(device)

                train_feats = model.gen_feats(subgraph_data)
                train_labels = model.get_labels(subgraph_data)

                loss = model.loss_with_feats(train_feats, train_labels, args.lam)

                optimizer.zero_grad()
                loss.backward()

                if iters == len(train_loader) - 1:
                    for n, p in model.named_parameters():
                        if p.grad != None:
                            p.grad.data = (
                                p.grad.data + all_grads[n]) / len(train_loader)
                    optimizer.step()
                else:
                    # aggregate all grads
                    for n, p in model.named_parameters():
                        if p.grad != None:
                            all_grads[n] += p.grad.data.clone()

                all_train_feats.append(train_feats.clone().detach().cpu())
                all_train_labels.append(train_labels.clone().detach().cpu())

                pbar.update(1)
            pbar.close()

            all_train_feats = torch.cat(all_train_feats, dim=0).to(device)
            all_train_labels = torch.cat(all_train_labels, dim=0).to(device)

            #######################################################
            if args.use_mlp or args.use_adapt_gcs:
                all_feats = compute_eval_feats(model, all_loader)
                eval_feat_cached = True

        #######################################################
        else:
            loss = model.loss_with_feats(all_train_feats, all_train_labels, args.lam)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #######################################################
        # evaluate
        if eval_feat_cached == False:
            all_feats = compute_eval_feats(model, all_loader)
            eval_feat_cached = True

        if epoch % args.eval_epochs == 0:
            model.eval()
            all_score = all_feats @ model.W.cpu()
            train_acc, val_acc, test_acc = pred_test(
                all_score, data, split_idx, evaluator)

            print(
                f"Epoch: {epoch}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}"
            )

            if val_acc > best_valid_score:
                best_epoch = epoch
                best_valid_score = val_acc
                best_params = copy.deepcopy(model.state_dict())

            if epoch - best_epoch > 1000:
                break
            
    model.load_state_dict(best_params)
    return model

#################################################
#################################################
#################################################
# pre-training

x_dims = data.x.size(1)
y_dims = data.y_one_hot_train.size(1)
h_dims = 128

start_time = time.time()

output_models = []
for shard_i in range(args.num_shards):
    
    single_time = time.time()
    output_models.append(pre_train(shard_i))
    print("train 1x model time", time.time() - single_time)
    
print("train model time", time.time() - start_time)


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

average_params = average_weights([model.state_dict() for model in output_models])
model = GNN(x_dims, h_dims, y_dims, device, args).to(device)
model.load_state_dict(average_params)


@torch.no_grad()
def evaluation_reuse_labels(model):
    model = model.to(device)

    # directly predict
    all_pred = []

    seed_everything(args.seed)
    for subgraph_data in all_loader:
        pred = model(subgraph_data.to(device))
        all_pred.append(pred.detach().cpu())

    all_pred = torch.cat(all_pred, dim=0)
    train_acc, val_acc, test_acc = pred_test(
        all_pred, data, split_idx, evaluator)
    print(
        f"Direct predict >>> Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")

    # reuse predicted labels
    y_one_hot_tmp = copy.deepcopy(data.y_one_hot_train)
    y_one_hot_tmp[split_idx["test"]] = F.one_hot(
        all_pred[split_idx["test"]].argmax(dim=-1, keepdim=True).squeeze(),
        data.y_one_hot_train.size(1)
    ).float()

    # label reuse
    all_pred = []

    seed_everything(args.seed)
    for subgraph_data in all_loader:
        subgraph_data.y_one_hot_train = y_one_hot_tmp[subgraph_data.node_inds]
        pred = model(subgraph_data.to(device))
        all_pred.append(pred.detach().cpu())

    all_pred = torch.cat(all_pred, dim=0)
    train_acc, val_acc, test_acc = pred_test(
        all_pred, data, split_idx, evaluator)
    print(
        f"Label reuse >>> Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")

    print(
        ">>> Number of nodes are predicted as the last category",
        torch.sum(
            all_pred[split_idx["train"]].argmax(dim=1) == dataset.num_classes
        ).item(),
    )
    
evaluation_reuse_labels(model)

retrain_shards = []
for shard_i in range(args.num_shards):
    
    intersect = np.intersect1d(delete_nodes_all, shard_nodes[shard_i])
    if len(intersect) > 0:
        retrain_shards.append(shard_i)
        
print(retrain_shards)

feat_dim = data.x.size(1)
label_dim = data.y_one_hot_train.size(1)

W_optim = model.W.data.cpu().clone()
extra_channel_norm_before = 0
split_dims = [feat_dim for _ in range(args.x_iters+1)] + [label_dim for _ in range(args.y_iters)]
for W_part in torch.split(W_optim, split_dims):
    extra_channel_norm_before += W_part[-1, :].norm(2).item()
print(extra_channel_norm_before)