import networkx as nx
import numpy as np
import torch
from collections import defaultdict
from dgl import DGLGraph


def constructDGL_s(graph, labels):
    new_g = DGLGraph()
    new_g.add_nodes(len(labels))
    for i in range(len(labels)):
        new_g.add_edge(i, i)
    for edge in graph.edges():
        new_g.add_edge(edge[0], edge[1])
        new_g.add_edge(edge[1], edge[0])

    return new_g


def constructDGL(graph, labels):
    node_mapping = defaultdict(int)
    relabels = []
    for node in sorted(list(graph.nodes())):
        node_mapping[node] = len(node_mapping)
        relabels.append(labels[node])
    assert len(node_mapping) == len(labels)
    new_g = DGLGraph()
    # new_g = DGLGraphStale()
    new_g.add_nodes(len(node_mapping))
    for i in range(len(node_mapping)):
        new_g.add_edge(i, i)
    for edge in graph.edges():
        new_g.add_edge(node_mapping[edge[0]], node_mapping[edge[1]])
        new_g.add_edge(node_mapping[edge[1]], node_mapping[edge[0]])

    return new_g, relabels


def read_struct_net(file_path, label_path):
    g = nx.Graph()
    with open(file_path) as IN:
        for line in IN:
            tmp = line.strip().split()
            g.add_edge(int(tmp[0]), int(tmp[1]))
    labels = dict()
    with open(label_path) as IN:
        IN.readline()
        for line in IN:
            tmp = line.strip().split(' ')
            labels[int(tmp[0])] = int(tmp[1])
    return g, labels


def degree_bucketing(graph, max_degree):
    features = torch.zeros([graph.number_of_nodes(), max_degree])
    for i in range(graph.number_of_nodes()):
        try:
            features[i][min(graph.in_degree(i), max_degree-1)] = 1
        except:
            features[i][0] = 1
    return features


def createTraining(labels, valid_mask=None, train_ratio=0.8):
    train_mask = torch.zeros(labels.shape, dtype=torch.bool)
    test_mask = torch.ones(labels.shape, dtype=torch.bool)

    num_train = int(labels.shape[0] * train_ratio)
    all_node_index = list(range(labels.shape[0]))
    np.random.shuffle(all_node_index)
    train_mask[all_node_index[:num_train]] = 1
    test_mask[all_node_index[:num_train]] = 0
    if valid_mask is not None:
        train_mask *= valid_mask
        test_mask *= valid_mask
    return train_mask, test_mask


def evaluate(model, features, labels):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        # logits = logits[mask]
        # labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)
