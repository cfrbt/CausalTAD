import pymetis
import pickle
import torch
import pdb

class GraphPartition:

    def __init__(self, adj_list: dict, node_num, L, threshold, load_path, load=True) -> None:
        if load:
            content = pickle.load(open(load_path, 'rb'))
            self.membership = content["membership"]
            self.label_start = content["label_start"]
            self.label_end = content["label_end"]
            self.adj_for_each_level = content["adj"]
            self.adj_tree = content["tree"]
        else:
            self.level_num = L-1
            self.threshold = threshold

            self.adj_list = []
            for i in range(node_num):
                self.adj_list.append(adj_list.setdefault(i, []))

            self.membership = [[-1 for _ in range(self.level_num)]+[i] for i in range(len(adj_list))]
            self.label_start = [0 for _ in range(self.level_num)] + [node_num]
            self.label_end = [0 for _ in range(self.level_num)] + [node_num-1]
            node_index = list(range(len(adj_list)))
            self.multi_level_partition(self.adj_list, node_index, 0)
            self.get_adj()
            content = {"membership": self.membership, "label_start": self.label_start, "label_end": self.label_end, "adj": self.adj_for_each_level, "tree": self.adj_tree}
            pickle.dump(content, open(load_path, 'wb'))

    def get_adj(self):
        self.label_start[self.level_num] = 0
        for l in range(self.level_num-1, -1, -1):
            self.label_end[l] = self.label_start[l] + self.label_end[l+1]
            self.label_start[l] = self.label_end[l+1] + 1

        for i in range(len(self.membership)):
            for l in range(self.level_num):
                self.membership[i][l] += self.label_start[l]
        
        self.adj_for_each_level = [dict() for _ in range(self.level_num+1)]
        self.adj_tree = dict()
        for i in range(len(self.membership)):
            for l in range(self.level_num):
                self.adj_tree.setdefault(self.membership[i][l], set()).add(self.membership[i][l+1])
            for neigh in self.adj_list[i]:
                for l in range(self.level_num+1):
                    self.adj_for_each_level[l].setdefault(self.membership[i][l], set()).add(self.membership[neigh][l])                

    def get_self_loop_adj(self, device):
        source = []
        target = []
        for l in range(self.level_num+1):
            for i in range(self.label_start[l], self.label_end[l]+1):
                for j in self.adj_tree.setdefault(i, []):
                    source.append(i)
                    target.append(j)
                for j in self.adj_for_each_level[l].setdefault(i, []):
                    source.append(i)
                    target.append(j)
                source.append(i)
                target.append(i)
        return torch.LongTensor([source, target]).to(device)


    def sample_subgraph(self, node_list):
        device = node_list[0].device
        sub_edge_list = []
        for l in range(self.level_num+1):
            points = list(set(node_list[l].view(-1).cpu().tolist()))
            sample_neighs = []
            for point in points:
                if point in self.adj_for_each_level[l].keys():
                    neighs = self.adj_for_each_level[l][point]
                else:
                    neighs = []
                sample_neighs.append(set(neighs))
                
            column_indices = [n for sample_neigh in sample_neighs for n in sample_neigh]
            row_indices = [points[i] for i in range(len(points)) for j in range(len(sample_neighs[i]))]
            sub_graph_edges = torch.LongTensor([row_indices, column_indices]).to(device)
            sub_edge_list.append(sub_graph_edges)
        return sub_edge_list

    def multi_level_partition(self, adj_list, node_index, L):
        if L==self.level_num:
            return
        
        if len(node_index)<=self.threshold:
            for i in node_index:
                for j in range(L, self.level_num):
                    self.membership[i][j] = self.label_start[j]
                    self.label_start[j] += 1
            return
        
        tmp_adj, index_map = self.index_map(adj_list, node_index)
        n_cuts, membership = pymetis.part_graph(6, adjacency=tmp_adj)
        
        node_index_group = [[] for i in range(6)]
        adj_list_group = [[] for i in range(6)]
        for _i,l in enumerate(membership):
            node_index_group[l].append(node_index[_i])
            adj_item = []
            for v in adj_list[_i]:
                if membership[index_map[v]]==l:
                    adj_item.append(v)
            adj_list_group[l].append(adj_item)
            self.membership[node_index[_i]][L] = self.label_start[L]+l
        self.label_start[L] += 6

        for i in range(6):
            self.multi_level_partition(adj_list_group[i], node_index_group[i], L+1)
    
    
    def index_map(self, adj_list, node_index):
        index_map = {node:_i for _i, node in enumerate(node_index)}
        tmp_adj = []
        for item in adj_list:
            new_item = []
            for i in item:
                new_item.append(index_map[i])
            tmp_adj.append(new_item)
        return tmp_adj, index_map
