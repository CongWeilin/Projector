# Reaches around 0.7945 ± 0.0059 test accuracy.
import pickle 
import argparse
import scipy.sparse as sp
import numpy as np
import os.path as osp
from torch_geometric import seed_everything

import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch.nn import Linear as Lin
from tqdm import tqdm

from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv

parser = argparse.ArgumentParser(description='OGBN-Products (GraphSAINT)')
parser.add_argument('--seed', type=int, default=10)
parser.add_argument("--num_remove_nodes", type=float, default=0.2)
parser.add_argument("--parallel_unlearning", type=int, default=20)
parser.add_argument('--continue_unlearn_step', action='store_true')
args = parser.parse_args()
print(args)

seed_everything(args.seed)

dataset = PygNodePropPredDataset(name='ogbn-products', root="../dataset")
split_idx = dataset.get_idx_split()
evaluator = Evaluator(name='ogbn-products')
data = dataset[0]

num_train_nodes = len(split_idx["train"])
train_ind_perm = np.random.permutation(split_idx["train"])
if args.num_remove_nodes < 1:
    args.num_remove_nodes = int(num_train_nodes * args.num_remove_nodes)
else:
    args.num_remove_nodes = int(args.num_remove_nodes)

print('Remove nodes %d/%d' % (args.num_remove_nodes, num_train_nodes))
delete_nodes_all = train_ind_perm[: args.num_remove_nodes]

batch = args.parallel_unlearning
delete_node_batch = [[] for _ in range(batch)]
for i, node_i in enumerate(delete_nodes_all):
    delete_node_batch[i % batch].append(node_i)

fn = 'exp3_results/exp3_retrain_graphsage_%d_%f_seed%d.pkl'%(args.parallel_unlearning, args.num_remove_nodes, args.seed)

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    def inference(self, x_all):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all
    
num_nodes = data.x.size(0)
remain_train = split_idx['train']
remain_nodes = np.arange(num_nodes)

row = data.edge_index[0].numpy()
col = data.edge_index[1].numpy()
adj_t_scipy = sp.csr_matrix((np.ones_like(row), (row, col)), shape=(num_nodes, num_nodes))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SAGE(dataset.num_features, 256, dataset.num_classes, num_layers=3)
model = model.to(device)

x = data.x.to(device)
y = data.y.squeeze().to(device)


# def train(epoch, train_idx):
#     model.train()

#     pbar = tqdm(total=train_idx.size(0))
#     pbar.set_description(f'Epoch {epoch:02d}')

#     total_loss = total_correct = 0
#     for batch_size, n_id, adjs in train_loader:
#         # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
#         adjs = [adj.to(device) for adj in adjs]

#         optimizer.zero_grad()
#         out = model(x[n_id], adjs)
#         loss = F.nll_loss(out, y[n_id[:batch_size]])
#         loss.backward()
#         optimizer.step()

#         total_loss += float(loss)
#         total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
#         pbar.update(batch_size)

#     pbar.close()

#     loss = total_loss / len(train_loader)
#     approx_acc = total_correct / train_idx.size(0)

#     return loss, approx_acc


@torch.no_grad()
def test():
    model.eval()

    out = model.inference(x)

    y_true = y.cpu().unsqueeze(-1)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    val_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, val_acc, test_acc

def train_model(remain_train):
    #################################################
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    best_val_acc = 0
    for epoch in range(1, 11):
        ###############
        model.train()

        pbar = tqdm(total=remain_train.size(0))
        pbar.set_description(f'Epoch {epoch:02d}')

        total_loss = total_correct = 0
        for batch_size, n_id, adjs in train_loader:
            # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
            adjs = [adj.to(device) for adj in adjs]

            optimizer.zero_grad()
            out = model(x[n_id], adjs)
            loss = F.nll_loss(out, y[n_id[:batch_size]])
            loss.backward()
            optimizer.step()

            total_loss += float(loss)
            total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
            pbar.update(batch_size)

        pbar.close()

        loss = total_loss / len(train_loader)
        acc = total_correct / remain_train.size(0)
        ###############
        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')

        if epoch > 5:
            result = test()
            train_acc, val_acc, test_acc = result
            print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
                  f'Test: {test_acc:.4f}')

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_result = result
    return best_result
    

train_loader = NeighborSampler(data.edge_index, node_idx=remain_train,
                                   sizes=[15, 10, 5], batch_size=1024,
                                   shuffle=True, num_workers=10)
    
subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                  batch_size=4096, shuffle=False,
                                  num_workers=10)
    
if args.continue_unlearn_step > 0:
    with open(fn, 'rb') as f:
        all_results = pickle.load(f)
else:
    results = train_model(remain_train)
    all_results = [results]
    with open(fn, 'wb') as f:
            pickle.dump(all_results, f)
            
#################################################
#################################################
#################################################
for cnt, delete_node_batch_i in enumerate(delete_node_batch): 
    #################################################
    # edit graph
    remain_nodes = np.setdiff1d(remain_nodes, delete_node_batch_i)
    remain_train = np.setdiff1d(remain_train, delete_node_batch_i)
    remain_train = torch.from_numpy(remain_train)
    print('Unlearn', cnt)
    if cnt < len(all_results) - 1:
        continue
    #################################################
    adj_t_scipy_remain = adj_t_scipy[remain_nodes, :][:, remain_nodes].tocoo()
    edge_index = np.stack([remain_nodes[adj_t_scipy_remain.row], remain_nodes[adj_t_scipy_remain.col]])
    data.edge_index = torch.from_numpy(edge_index).long()
    
    train_loader = NeighborSampler(data.edge_index, node_idx=remain_train,
                                   sizes=[15, 10, 5], batch_size=1024,
                                   shuffle=True, num_workers=10)
    
    subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                      batch_size=4096, shuffle=False,
                                      num_workers=10)
    
    results = train_model(remain_train)
    all_results += [results]
    
    with open(fn, 'wb') as f:
        pickle.dump(all_results, f)
