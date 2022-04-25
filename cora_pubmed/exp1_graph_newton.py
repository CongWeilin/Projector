import argparse

import os.path as osp

import torch
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric import seed_everything

import time
import copy
import numpy as np

from model_utils import GNN 
from newton_utils import lr_hessian_inv, lr_grad

################################################################
parser = argparse.ArgumentParser(description="Planetoid (GNN)")
parser.add_argument("--dataset", type=str, default='Cora')
parser.add_argument("--num_remove_nodes", type=float, default=0.02)
parser.add_argument("--lam", type=float, default=0)
parser.add_argument("--is_influence", action="store_true")
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=0)

args = parser.parse_args()

# load dataset
dataset = args.dataset
num_remove_nodes = args.num_remove_nodes

path = osp.join('data', dataset)
if dataset in ['Cora', 'Pubmed']:
    dataset = Planetoid(path, dataset,
                        transform=T.ToSparseTensor(),
                        split='full'
                        )
else:
    dataset = Reddit2(path, dataset,
                      transform=T.ToSparseTensor()
    )
data = dataset[0]
seed_everything(0)


print(dataset, num_remove_nodes)
################################################################
# inject feature + label
train_inds = torch.where(data.train_mask)[0].cpu().numpy()

num_train_nodes = len(train_inds)
train_ind_perm = np.random.permutation(train_inds)
if num_remove_nodes < 1:
    num_remove_nodes = int(num_train_nodes * num_remove_nodes)
else:
    num_remove_nodes = int(num_remove_nodes)
print('Remove nodes %d/%d' % (num_remove_nodes, num_train_nodes))
delete_nodes_all = train_ind_perm[:num_remove_nodes]

extra_feats = torch.zeros(data.x.size(0))
extra_feats[delete_nodes_all] = 1

data.x = torch.cat([data.x, extra_feats.view(-1, 1)], dim=1)
data.y[delete_nodes_all] = dataset.num_classes
data.y_one_hot = F.one_hot(data.y, dataset.num_classes)

################################################################
# training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = GNN(data.x.size(1), dataset.num_classes).to(
    device), data.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

start_time = time.time()
best_val = 0
epoch = 0
while True:
    epoch += 1
    model.train()
    pred = model.pred(data.adj_t, data.x)
    loss = model.ovr_lr_loss(pred[data.train_mask], data.y_one_hot[data.train_mask], args.lam)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    mask = data.train_mask
    train_acc = pred[mask].max(1)[1].eq(
        data.y[mask]).sum().item() / mask.sum().item()
    mask = data.val_mask
    val_acc = pred[mask].max(1)[1].eq(
        data.y[mask]).sum().item() / mask.sum().item()
    mask = data.test_mask
    test_acc = pred[mask].max(1)[1].eq(
        data.y[mask]).sum().item() / mask.sum().item()

    if best_val < val_acc:
        best_train = train_acc
        best_val = val_acc
        best_test = test_acc
        best_epoch = epoch
        best_params = copy.deepcopy(model.state_dict())

    if best_epoch < epoch - 200:
        print('Run epochs', best_epoch)
        break
#     print("train >>>", train_acc, val_acc, test_acc)

model.load_state_dict(best_params)
print("Final", best_train, best_val, best_test, 
      "Time", time.time()- start_time)

################################################################
# unlearning

def influence_removal(delete_node, remain_nodes, W_optim):
    print(">>> Influence-based method: delete node {}/{} from graph".format(len(delete_node), len(remain_nodes)))
    W_prime = W_optim.clone()
    for k in range(dataset.num_classes):
        # delete nodes
        grad = lr_grad(
            w=W_optim[:, k].cpu(),
            X=model(data.adj_t, data.x)[delete_node].detach().cpu(),
            y=data.y_one_hot[delete_node, k].cpu(),
            lam=args.lam,
        )
        # remain nodes
        hessian = lr_hessian_inv(
            w=W_optim[:, k].cpu(),
            X=model(data.adj_t, data.x)[remain_nodes].detach().cpu(),
            y=data.y_one_hot[remain_nodes, k].cpu(),
            lam=args.lam,
        )
        delta = hessian.mv(grad)
        W_prime[:, k] += delta
    return W_prime

def fisher_removal(delete_node, remain_nodes, W_optim):
    print(">>> Fisher-based method: delete node {}/{} from graph".format(len(delete_node), len(remain_nodes)))
    W_prime = W_optim.clone()
    for k in range(dataset.num_classes):
        # delete nodes
        grad = lr_grad(
            w=W_optim[:, k].cpu(),
            X=model(data.adj_t, data.x)[remain_nodes].detach().cpu(),
            y=data.y_one_hot[remain_nodes, k].cpu(),
            lam=args.lam,
        )
        # remain nodes
        hessian = lr_hessian_inv(
            w=W_optim[:, k].cpu(),
            X=model(data.adj_t, data.x)[remain_nodes].detach().cpu(),
            y=data.y_one_hot[remain_nodes, k].cpu(),
            lam=args.lam,
        )
        delta = hessian.mv(grad)
        W_prime[:, k] -= delta
    return W_prime


W_optim = model.W.data.clone().cpu()

start_time = time.time()

# compute affect + remain nodes
adj_t = data.adj_t.to_scipy('csr')
adj_t_dense = data.adj_t.to_scipy('csr').toarray()
adj_t_dense = adj_t@adj_t_dense

affect_nodes = np.where(np.sum(adj_t_dense[delete_nodes_all, :], axis=0) > 0)[0]
affect_nodes = np.intersect1d(affect_nodes, train_inds)
remain_nodes = np.setdiff1d(train_inds, affect_nodes)

if args.is_influence:
    W_unlearn = influence_removal(
        affect_nodes, remain_nodes, W_optim
    )
else:
    W_unlearn = fisher_removal(
        affect_nodes, remain_nodes, W_optim
    )

extra_channel_norm_before = W_optim[-1, :].norm(2).item()
extra_channel_norm_after  = W_unlearn[-1, :].norm(2).item()

print('time', time.time() - start_time, 
      'weight norm', extra_channel_norm_before, extra_channel_norm_after, 
      'weight change', (W_optim - W_unlearn).norm(2))

################################################################
# evaluate
model.W.data = W_unlearn.to(device)
model.eval()
pred = model.pred(data.adj_t, data.x)

mask = data.train_mask
train_acc = pred[mask].max(1)[1].eq(
    data.y[mask]).sum().item() / mask.sum().item()
mask = data.val_mask
val_acc = pred[mask].max(1)[1].eq(
    data.y[mask]).sum().item() / mask.sum().item()
mask = data.test_mask
test_acc = pred[mask].max(1)[1].eq(
    data.y[mask]).sum().item() / mask.sum().item()

print("Unlearn >>>", train_acc, val_acc, test_acc)
