import pdb
import pickle
import torch

class GraphLoader:

    def __init__(self, node_file, node_adj_file, device):

        self.nodes = pickle.load(open(node_file, 'rb'))
        self.node2id = {n:i for i,n in enumerate(self.nodes.keys())}
        self.load_adj(node_adj_file)
        self.second_order_adj()
        self.device = device
        self.first_adj = self.first_order_adj()
    
    def load_adj(self, node_adj_file):
        tmp = pickle.load(open(node_adj_file, 'rb'))

        self.node_adj = dict()
        for node, id in self.node2id.items():
            self.node_adj.setdefault(id, [])
            for neigh in tmp[node].keys():
                self.node_adj[id].append(self.node2id[neigh])

    def first_order_adj(self):
        source = []
        target = []
        for src, value in self.node_adj.items():
            for trg in value:
                source.append(src)
                target.append(trg)
        
        return torch.LongTensor([source, target]).to(self.device)

    def second_order_adj(self):
        self.second_order_dict = dict()
        for src, value in self.node_adj.items():
            self.second_order_dict.setdefault(src, set()).add(src)
            for m in value:
                for trg in self.node_adj[m]:
                    self.second_order_dict.setdefault(src, set()).add(trg)
        
        self.edge2index = dict()
        for src, value in self.second_order_dict.items():
            for trg in value:
                self.edge2index[(src, trg)] = len(self.edge2index)

    def sample_subgraph(self, node_list):
        device = node_list.device
        points = list(set(node_list.view(-1).cpu().tolist()))
        sample_neighs = []
        for point in points:
            if point in self.node_adj.keys():
                neighs = self.node_adj[point]
            else:
                neighs = []
            sample_neighs.append(set(neighs))
            
        column_indices = [n for sample_neigh in sample_neighs for n in sample_neigh]
        row_indices = [points[i] for i in range(len(points)) for j in range(len(sample_neighs[i]))]
        sub_graph_edges = torch.LongTensor([row_indices, column_indices]).to(device)
        return sub_graph_edges
    
    def sample_second_subgraph(self, node_list):
        device = node_list.device
        points = list(set(node_list.view(-1).cpu().tolist()))
        sample_neighs = []
        for point in points:
            if point in self.second_order_dict.keys():
                neighs = self.second_order_dict[point]
            else:
                neighs = []
            sample_neighs.append(set(neighs))
            
        column_indices = [n for sample_neigh in sample_neighs for n in sample_neigh]
        row_indices = [points[i] for i in range(len(points)) for j in range(len(sample_neighs[i]))]
        sub_graph_edges = torch.LongTensor([column_indices, row_indices]).to(device)
        sub_edge_index = [self.edge2index[(src, trg)] for src, trg in zip(column_indices, row_indices)]
        sub_edge_index = torch.LongTensor(sub_edge_index).to(device)
        return sub_graph_edges, sub_edge_index