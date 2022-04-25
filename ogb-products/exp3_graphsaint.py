import argparse
import scipy.sparse as sp
import numpy as np
import pickle 
from torch_geometric import seed_everything

import torch
from tqdm import tqdm
import torch.nn.functional as F

from torch_geometric.data import GraphSAINTRandomWalkSampler, NeighborSampler
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import subgraph

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight)
        return torch.log_softmax(x, dim=-1)

    def inference(self, x_all, subgraph_loader, device):
        pbar = tqdm(total=x_all.size(0) * len(self.convs))
        pbar.set_description('Evaluating')

        for i, conv in enumerate(self.convs):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = conv((x, x_target), edge_index)
                if i != len(self.convs) - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all


def train(model, loader, optimizer, device):
    model.train()

    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        y = data.y.squeeze(1)
        loss = F.nll_loss(out[data.train_mask], y[data.train_mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def test(model, data, evaluator, subgraph_loader, device):
    model.eval()

    y_pred = model.inference(data.x, subgraph_loader, device)

    y_true = data.y
    y_pred = y_pred.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[data.train_mask],
        'y_pred': y_pred[data.train_mask]
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[data.valid_mask],
        'y_pred': y_pred[data.valid_mask]
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[data.test_mask],
        'y_pred': y_pred[data.test_mask]
    })['acc']

    return train_acc, valid_acc, test_acc

def train_model(model):
    #  update loader
    loader = GraphSAINTRandomWalkSampler(data,
                                         batch_size=args.batch_size,
                                         walk_length=args.walk_length,
                                         num_steps=args.num_steps,
                                         sample_coverage=0)

    subgraph_loader = NeighborSampler(data.edge_index, sizes=[-1],
                                      batch_size=4096, shuffle=False,
                                      num_workers=10)
    # train
    best_valid = 0

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1, 1 + args.epochs):
        loss = train(model, loader, optimizer, device)
        if epoch % args.log_steps == 0:
            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}')

        if epoch > 9 and epoch % args.eval_steps == 0:
            result = test(model, data, evaluator, subgraph_loader, device)
            train_acc, valid_acc, test_acc = result

            if best_valid < valid_acc:
                best_valid = valid_acc
                best_result = result

            print(f'Epoch: {epoch:02d}, '
                  f'Train: {100 * train_acc:.2f}%, '
                  f'Valid: {100 * valid_acc:.2f}% '
                  f'Test: {100 * test_acc:.2f}%')
    return best_result
    
#####################################################################################################
#####################################################################################################
#####################################################################################################
parser = argparse.ArgumentParser(description='OGBN-Products (GraphSAINT)')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--log_steps', type=int, default=1)
parser.add_argument('--inductive', action='store_true')
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--hidden_channels', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--batch_size', type=int, default=20000)
parser.add_argument('--walk_length', type=int, default=3)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--num_steps', type=int, default=30)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--eval_steps', type=int, default=2)
parser.add_argument('--seed', type=int, default=10)
parser.add_argument("--num_remove_nodes", type=float, default=0.2)
parser.add_argument("--parallel_unlearning", type=int, default=20)

parser.add_argument('--continue_unlearn_step', action='store_true')

args = parser.parse_args()
print(args)

seed_everything(args.seed)

device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

dataset = PygNodePropPredDataset(name='ogbn-products', root="../dataset")
split_idx = dataset.get_idx_split()

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

batch = args.parallel_unlearning
delete_node_batch = [[] for _ in range(batch)]
for i, node_i in enumerate(delete_nodes_all):
    delete_node_batch[i % batch].append(node_i)

#################################################
#################################################
#################################################
data = dataset[0]
num_nodes = data.x.size(0)

# Convert split indices to boolean masks and add them to `data`.
for key, idx in split_idx.items():
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[idx] = True
    data[f'{key}_mask'] = mask

# We omit normalization factors here since those are only defined for the
# inductive learning setup.

row = data.edge_index[0].numpy()
col = data.edge_index[1].numpy()
adj_t_scipy = sp.csr_matrix((np.ones_like(row), (row, col)), shape=(num_nodes, num_nodes))

#################################################
#################################################
#################################################
fn = 'exp3_results/exp3_retrain_graphsaint_%d_%f_seed%d.pkl'%(args.parallel_unlearning, args.num_remove_nodes, args.seed)

model = SAGE(data.x.size(-1), args.hidden_channels, dataset.num_classes,
             args.num_layers, args.dropout).to(device)
evaluator = Evaluator(name='ogbn-products')

remain_nodes = np.arange(num_nodes)

if args.continue_unlearn_step > 0:
    with open(fn, 'rb') as f:
        all_results = pickle.load(f)
else:
    results = train_model(model)
    all_results = [results]
    with open(fn, 'wb') as f:
            pickle.dump(all_results, f)

for cnt, delete_node_batch_i in enumerate(delete_node_batch): 
    # edit graph
    remain_nodes = np.setdiff1d(remain_nodes, delete_node_batch_i)
    print('>>> Unlearn', cnt)

    if cnt < len(all_results) - 1:
        continue

    adj_t_scipy_remain = adj_t_scipy[remain_nodes, :][:, remain_nodes].tocoo()
    edge_index = np.stack([remain_nodes[adj_t_scipy_remain.row], 
                           remain_nodes[adj_t_scipy_remain.col]])
    data.edge_index = torch.from_numpy(edge_index).long()
    results = train_model(model)
    all_results += [results]

    with open(fn, 'wb') as f:
        pickle.dump(all_results, f)
