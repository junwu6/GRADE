from __future__ import division
from __future__ import print_function
import argparse
from utils import *
from GRADE_train import GRADE_main

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default='airport', choices='airport|social|citation')
parser.add_argument("--model_name", type=str, default='GRADE')
parser.add_argument('--n_hidden', type=int, default=8)
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument("--dropout", type=float, default=0.)
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--cuda', type=int, default=0)
args = parser.parse_args()


def set_random_seed(seed=0):
    # seed setting
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_data(name, d="s"):
    g, labels = read_struct_net(file_path='data/{}-airports.edgelist'.format(name),
                                label_path='data/labels-{}-airports.txt'.format(name))
    g.remove_edges_from(nx.selfloop_edges(g))
    g, labels = constructDGL(g, labels)
    labels = torch.LongTensor(labels)
    features = degree_bucketing(g, args.n_hidden)
    if d == "t":
        features += 0.1

    return g, labels, features


if __name__ == '__main__':
    args.device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")
    print("Results on {} data set".format(args.data))
    if args.data == "airport":
        domains = ["usa", "brazil", "europe"]

    all_acc = []
    for s_domain in domains:
        for t_domain in domains:
            set_random_seed(seed=args.seed)
            if s_domain == t_domain:
                continue
            data_s = get_data(s_domain)
            data_t = get_data(t_domain, d="t")
            acc = GRADE_main(args, data_s, data_t)
            all_acc.append(acc)

    print("\n")
    print("Model: {}, Accuracy: {}, Mean: {}".format(args.model_name, all_acc, np.mean(all_acc)))

