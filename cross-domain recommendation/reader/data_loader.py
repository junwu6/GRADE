from reader.base import Dataset
import numpy as np
import pandas as pd
import scipy.sparse as sp
from multiprocessing import Pool
import copy


class data_loader(Dataset):
    def __init__(self, path, name, target, print_summary=False):
        super(data_loader, self).__init__(path, name, target, print_summary)

    def get_train(self, neg_num=1):
        processor_num = 4
        with Pool(processor_num) as pool:
            nargs = [(user, item, self.train_neg_dict[user]) for user, item in self.train_data]
            res_list = pool.map(_add_negtive, nargs)

        out = []
        for batch_n in res_list:
            out += batch_n

        adj, feats = self.construct_g()
        return out, adj, feats

    def construct_g(self):
        num_nodes = self.num_user + self.num_item
        edges = np.array(copy.deepcopy(self.train_data))
        edges[:, 1] += self.num_user
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(num_nodes, num_nodes),
                            dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj, feats = self.normalize_adj(adj + sp.eye(adj.shape[0]))
        return adj, feats

    def normalize_adj(self, adj):
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        feats = self.one_hot_embedding((rowsum-1).astype(int).flatten())
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo(), feats.astype(np.float)

    def one_hot_embedding(self, degree, num_classes=5000):
        y = np.eye(num_classes)
        return y[degree]


# @staticmethod
def _add_negtive(args):
    user, item, neg_dict = args
    neg_pair = [[user, item, 1]]

    neg_num = 4
    neg_sample_list = np.random.choice(neg_dict, neg_num, replace=False).tolist()
    for neg_sample in neg_sample_list:
        neg_pair.append([user, neg_sample, 0])

    return neg_pair
