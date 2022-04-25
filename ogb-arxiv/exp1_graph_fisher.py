import argparse
import copy
import pickle
import numpy as np
import os
import random
import scipy.sparse as sp
from tqdm import tqdm
import math
import time

import torch_sparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch_geometric.transforms as T

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from utils.train_utils import pred_test
from utils.newton_utils import lr_grad, lr_hessian_inv, predict
from utils.baseline_utils import get_2_hop_neighbors, compute_node_feats, GNN

#################################################
#################################################
#################################################

parser = argparse.ArgumentParser(description="OGBN-Arxiv (GNN)")
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--epochs", type=int, default=10000)
parser.add_argument("--runs", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--dropout_times", type=int, default=2)

parser.add_argument("--regen_feats", action="store_true")
parser.add_argument("--regen_model", action="store_true")
parser.add_argument("--num_remove_nodes", type=float, default=0.05)
parser.add_argument("--parallel_unlearning", type=int, default=10)
parser.add_argument("--lam", type=float, default=1e-4,
                    help="L2 regularization")
parser.add_argument("--hop_neighbors", type=int, default=15)

args = parser.parse_args()
args.regen_feats = True
args.regen_neighbors = False
args.regen_model = True
print(args)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

dataset = PygNodePropPredDataset(
    name="ogbn-arxiv", transform=T.ToSparseTensor(), root="../dataset"
)

data = dataset[0]
split_idx = dataset.get_idx_split()

evaluator = Evaluator(name="ogbn-arxiv")

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

if args.parallel_unlearning < 0:
    args.parallel_unlearning = int(args.num_remove_nodes/10)
print('Unlearning by %d iters' % args.parallel_unlearning)
#################################################
#################################################
#################################################
# get adjs
data.adj_t = torch_sparse.fill_diag(data.adj_t.to_symmetric(), 1)

data.y_one_hot_train = F.one_hot(
    data.y.squeeze(), dataset.num_classes + 1).float()
data.y_one_hot_train[split_idx["test"], :] = 0

adj_t_scipy = data.adj_t.to_scipy().tocsr()
num_nodes = adj_t_scipy.shape[0]

#################################################
#################################################
#################################################
# compute 2-hop neighbors
fn = os.path.join(os.getcwd(), "baseline_cache_info",
                  "neighbor_nodes_%d.pkl" % (args.hop_neighbors))
print(fn)

if os.path.exists(fn):
    with open(fn, "rb") as f:
        neighbor_nodes, subgraph_relation = pickle.load(f)
else:
    neighbor_nodes, subgraph_relation = get_2_hop_neighbors(
        num_nodes, adj_t_scipy)
    with open(fn, "wb") as f:
        pickle.dump([neighbor_nodes, subgraph_relation], f)

#################################################
#################################################
#################################################
# compute node feats
fn = os.path.join(os.getcwd(), "baseline_cache_info", "exp1_node_feat_%d_%d.pkl" %
                  (args.hop_neighbors, args.num_remove_nodes))
print(fn)

if os.path.exists(fn) and not args.regen_feats:
    with open(fn, "rb") as f:
        node_feat_all = pickle.load(f)
else:
    node_feat_all = compute_node_feats(
        range(num_nodes),
        neighbor_nodes,
        adj_t_scipy,
        data.x,
        data.y_one_hot_train
    )
    with open(fn, "wb") as f:
        pickle.dump(node_feat_all, f)

#################################################
#################################################
#################################################
# pre-train model


def pre_train(W_init, train_inds, print_per_epoch=100):
    best_valid_score = 0

    model = GNN(W_init).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    X_cuda = node_feat_all.cuda()
    Y_cuda = data.y_one_hot_train.cuda()

    for epoch in range(1, 1 + args.epochs):
        scores = model(X_cuda)

        loss = model.ovr_lr_loss(
            X_cuda[train_inds], Y_cuda[train_inds], args.lam)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc, val_acc, test_acc = pred_test(
            scores.cpu(), data, split_idx, evaluator
        )
        if epoch % print_per_epoch == 0:
            print(
                f"Epoch: {epoch}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}"
            )

        if val_acc > best_valid_score:
            best_valid_score = val_acc
            best_W = model.W.data.clone().cpu()
            best_epoch = epoch

        if epoch - best_epoch > 1000:
            break

    return best_W


#################################################
W_init = torch.rand(node_feat_all.size(1), data.y_one_hot_train.size(1))

start_time = time.time()

fn = os.path.join(os.getcwd(), "baseline_cache_info", "exp1_fisher_%d_%d.pt" % (
    args.hop_neighbors, args.num_remove_nodes))
if os.path.exists(fn) and not args.regen_model:
    W_optim = torch.load(fn)
else:
    W_optim = pre_train(W_init, split_idx['train'])
    torch.save(W_optim, fn)

print("train model time", time.time() - start_time)

#################################################
#################################################
#################################################


def evaluation_reuse_labels(W_optim, regen_feats=False):
    """
    Reuse labels for evaluation
    """
    y_dim = data.y_one_hot_train.size(1)

    # predict labels from test node
    if regen_feats:
        node_feat_all_inf = compute_node_feats(
            range(num_nodes), neighbor_nodes, adj_t_scipy, data.x, data.y_one_hot_train
        )
        scores = predict(node_feat_all_inf, W_optim)
    else:
        scores = predict(node_feat_all, W_optim)

    train_acc, val_acc, test_acc = pred_test(
        scores, data, split_idx, evaluator)
    print(f"Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")

    # reuse predicted labels
    y_one_hot_inference = copy.deepcopy(data.y_one_hot_train)
    y_one_hot_inference[split_idx["test"]] = F.one_hot(
        scores[split_idx["test"]].argmax(dim=-1, keepdim=True).squeeze(), y_dim
    ).float()

    node_feat_all_inf = compute_node_feats(
        split_idx["test"], neighbor_nodes, adj_t_scipy, data.x, y_one_hot_inference
    )

    # predict labels using test node label reuse
    scores[split_idx["test"]] = predict(node_feat_all_inf, W_optim)

    train_acc, val_acc, test_acc = pred_test(
        scores, data, split_idx, evaluator)
    print(f"Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")

    print(
        ">>> Number of nodes are predicted as the last category",
        torch.sum(
            scores[split_idx["train"]].argmax(dim=1) == dataset.num_classes
        ).item(),
    )


# evaluation_reuse_labels(W_optim)


########################################################
########################################################
########################################################


def fisher_removal(delete_node, remain_nodes, W_optim):

    print(">>> Fisher-based method: delete node {}/{} from graph".format(len(delete_node), len(remain_nodes)))

    W_prime = W_optim.clone()

    for k in range(dataset.num_classes):

        # delete nodes
        grad = lr_grad(
            w=W_optim[:, k],
            X=node_feat_all[remain_nodes],
            y=data.y_one_hot_train[remain_nodes, k],
            lam=args.lam,
        )
        # remain nodes
        hessian = lr_hessian_inv(
            w=W_optim[:, k],
            X=node_feat_all[remain_nodes],
            y=data.y_one_hot_train[remain_nodes, k],
            lam=args.lam,
        )

        delta = hessian.mv(grad)
        W_prime[:, k] -= delta

    return W_prime


##############################################################
##############################################################
##############################################################
start_time = time.time()

affect_nodes = []
for node_i in delete_nodes_all:
    affect_nodes.append(node_i)
    affect_nodes.extend(subgraph_relation[node_i])
affect_nodes = np.unique(affect_nodes)
affect_nodes = np.intersect1d(affect_nodes, split_idx['train'])
remain_nodes = np.setdiff1d(split_idx['train'], affect_nodes)

W_unlearn = fisher_removal(
    affect_nodes, remain_nodes, W_optim
)

print("Weight norm changes", (W_optim - W_unlearn).sum())

print("Total time:", time.time() - start_time)

evaluation_reuse_labels(W_unlearn, regen_feats=True)
evaluation_reuse_labels(W_optim, regen_feats=True)


# compute weight norm for augment channels
feat_dim = data.x.size(1)
label_dim = data.y_one_hot_train.size(1)

extra_channel_norm_before = 0
extra_channel_norm_after = 0

all_dims = [feat_dim, label_dim]
# all_dims = [feat_dim for _ in range(4)] + [label_dim for _ in range(3)]
for W_part in torch.split(W_optim, all_dims):
    extra_channel_norm_before += W_part[-1, :].norm(2).item()
for W_part in torch.split(W_unlearn, all_dims):
    extra_channel_norm_after += W_part[-1, :].norm(2).item()
print('extra_channel_norm_before', extra_channel_norm_before,
      'extra_channel_norm_after',  extra_channel_norm_after)