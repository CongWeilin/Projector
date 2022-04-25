import argparse
import pickle
import numpy as np

import torch
import torch.nn.functional as F

from torch_sparse import SparseTensor

from torch_geometric import seed_everything
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


def main():
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument("--num_remove_nodes", type=float, default=0.2)
    parser.add_argument("--parallel_unlearning", type=int, default=20)
    args = parser.parse_args()
    print(args)
    
    seed_everything(args.seed)
    
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(
        name="ogbn-arxiv",
        transform=T.ToSparseTensor(),
        root="../dataset"
    )

    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data = data.to(device)

    split_idx = dataset.get_idx_split()
    num_nodes = data.x.size(0)
    
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

    #################################################
    #################################################
    #################################################

    if args.use_sage:
        model = SAGE(data.num_features, args.hidden_channels,
                     dataset.num_classes, args.num_layers,
                     args.dropout).to(device)
    else:
        model = GCN(data.num_features, args.hidden_channels,
                    dataset.num_classes, args.num_layers,
                    args.dropout).to(device)

    evaluator = Evaluator(name='ogbn-arxiv')
    
    #################################################
    #################################################
    #################################################
    # use all train nodes
    
    def train_with_inds(model, train_idx):
        print(data)
        
        print('num train data', len(train_idx))
        model.reset_parameters()
        
        best_valid = 0
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, data, train_idx, optimizer)
            result = test(model, data, split_idx, evaluator)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                # print(f'Epoch: {epoch:02d}, '
                #       f'Loss: {loss:.4f}, '
                #       f'Train: {100 * train_acc:.2f}%, '
                #       f'Valid: {100 * valid_acc:.2f}% '
                #       f'Test: {100 * test_acc:.2f}%')

                if best_valid < valid_acc:
                    best_valid = valid_acc
                    best_results = result
                    
        return best_results
    
    #################################################
    #################################################
    #################################################
    batch = args.parallel_unlearning
    delete_node_batch = [[] for _ in range(batch)]
    for i, node_i in enumerate(delete_nodes_all):
        delete_node_batch[i % batch].append(node_i)
    
    remain_nodes = split_idx['train']
    all_results = [train_with_inds(model, remain_nodes)]
    
    adj_t_scipy = data.adj_t.to_scipy('csr')
    
    for cnt, delete_node_batch_i in enumerate(delete_node_batch): 
        remain_nodes = np.setdiff1d(remain_nodes, delete_node_batch_i)
        
        remain_valid_test_nodes = np.concatenate([remain_nodes, split_idx['valid'], split_idx['test']])
        adj_t_scipy_remain = adj_t_scipy[remain_valid_test_nodes, :][:, remain_valid_test_nodes].tocoo()
        data.adj_t = SparseTensor(
            row=torch.from_numpy(remain_valid_test_nodes[adj_t_scipy_remain.row]).long(),
            col=torch.from_numpy(remain_valid_test_nodes[adj_t_scipy_remain.col]).long(),
            value=torch.from_numpy(adj_t_scipy_remain.data).float(),
            sparse_sizes=(num_nodes, num_nodes)
        ).to(device)
        all_results.append(train_with_inds(model, remain_nodes))
    
    if args.use_sage:
        fn = 'exp3_results/exp3_retrain_graphsage_%d_%f_seed%d.pkl'%(args.parallel_unlearning, args.num_remove_nodes, args.seed)
    else:
        fn = 'exp3_results/exp3_retrain_gcn_%d_%f_seed%d.pkl'%(args.parallel_unlearning, args.num_remove_nodes, args.seed)
        
    with open(fn, 'wb') as f:
        pickle.dump(all_results, f)
    
    
if __name__ == "__main__":
    main()
    
    
# python exp3_gcn_graphsage.py --seed 0
