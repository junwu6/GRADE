from __future__ import print_function
import argparse
import torch
import numpy as np
from metric.metric import ndcg_at_k
from reader.data_loader import data_loader
from models.GRADE import GRADE

# Command setting
parser = argparse.ArgumentParser(description='Cross-Domain Recommendation')
parser.add_argument('-model_name', type=str, default='GRADE', help='model name')
parser.add_argument('-data_dir', type=str, default='data/nonoverlapping/CD_Music', help='domain path')
parser.add_argument('-s_domain', type=str, default='music', help='source domain')
parser.add_argument('-t_domain', type=str, default='cd', help='target domain')
parser.add_argument('-edim', type=int, default=8, help='embedding dimensionality')
parser.add_argument('-lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('-reg', type=float, default=0.0001, help='weight decay')
parser.add_argument('-batch_size', type=int, default=1024, help='batch size')
parser.add_argument('-epochs', type=int, default=100, help='batch size')
parser.add_argument('-cuda', type=int, default=1, help='cuda id')
args = parser.parse_args()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def evaluate_test_at_k(model, test_dict, test_input_dict, k=10):
    with torch.no_grad():
        relevant = 0
        selected = 0
        hit = 0
        mrr = 0
        ndcg = 0
        n = 0
        for user in test_dict.keys():
            items = torch.tensor(test_input_dict[user])
            users = torch.tensor([user] * len(items))
            output = model.inference(users, items)
            indices = torch.argsort(output, dim=0, descending=True)[0:k].tolist()
            pred = []
            for idx in indices:
                pred.append(items[idx])
            actual = test_dict[user]
            # print(pred)
            # print(actual)
            reward = 0
            for item in pred:
                if item in actual:
                    reward += 1

            n += reward
            relevant += len(actual)
            selected += len(pred)
            if reward > 0:
                hit += 1

                r = []
                for i in pred:
                    if i in actual:
                        r.append(1)
                    else:
                        r.append(0)
                rf = np.asarray(r).nonzero()[0]
                mrr += 1 / (rf[0] + 1)
                ndcg += ndcg_at_k(r, k)

        print("HIT RATIO@{}: {:.4f}".format(k, hit / len(test_dict.keys())))
        print("MRR@{}: {:.4f}".format(k, mrr / len(test_dict.keys())))
        print("NDCG@{}: {:.4f}".format(k, ndcg / len(test_dict.keys())))
        print("PRECISION@{}: {:.4f}".format(k, n / selected))
        print("RECALL@{}: {:.4f}".format(k, n / relevant))
        print()


if __name__ == '__main__':
    args.device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")

    dataset_ds = data_loader(args.data_dir, args.s_domain, target=False, print_summary=True)
    dataset_dt = data_loader(args.data_dir, args.t_domain, target=True, print_summary=True)
    test_dict_ds, test_input_dict_ds, num_user_ds, num_item_ds = dataset_ds.get_data()
    test_dict_dt, test_input_dict_dt, num_user_dt, num_item_dt = dataset_dt.get_data()
    data_ds, adj_s, feats_s = dataset_ds.get_train()
    adj_s = sparse_mx_to_torch_sparse_tensor(adj_s).to(device=args.device)
    data, adj_t, feats_t = dataset_dt.get_train()
    adj_t = sparse_mx_to_torch_sparse_tensor(adj_t).to(device=args.device)
    model = GRADE(args, num_item_ds, num_user_ds, num_item_dt, num_user_dt, adj_s, adj_t, feats_s, feats_t)
    train_data_s = torch.tensor(data_ds, device=args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.reg)
    print("Start training...{}".format(args.model_name))
    for epoch in range(1, 1+args.epochs):
        model.train()
        train_data = torch.tensor(data, device=args.device)
        permutation = torch.randperm(train_data.shape[0])
        max_idx = int((len(permutation) // (args.batch_size / 2) - 1) * (args.batch_size / 2))

        loss = 0.
        for batch in range(0, max_idx, args.batch_size):
            optimizer.zero_grad()
            idx = permutation[batch: batch + args.batch_size]
            idx_s = np.random.choice(train_data_s.shape[0], args.batch_size)
            loss = model(train_data_s[idx_s, :], train_data[idx])
            loss.backward()
            optimizer.step()

        if epoch % 1 == 0:
            model.eval()
            print("epoch {} loss: {:.4f}".format(epoch, loss))
            evaluate_test_at_k(model, test_dict_dt, test_input_dict_dt, k=10)

    print("Results:")
    model.eval()
    evaluate_test_at_k(model, test_dict_dt, test_input_dict_dt, k=10)
