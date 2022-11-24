from __future__ import division
from __future__ import print_function
from GRADE import *
# from GRADEv2 import *


def train(args, g_s, features_s, labels_s, g_t, features_t, labels_t):
    in_feats = features_s.shape[1]
    n_classes = labels_s.max().item() + 1
    features_s = features_s.to(device=args.device)
    labels_s = labels_s.to(device=args.device)
    features_t = features_t.to(device=args.device)
    model = GRADE(g_s, g_t, in_feats, args.n_hidden, n_classes, args.n_layers, args.dropout).to(device=args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        alpha = 2 / (1 + np.exp(- 10 * epoch / args.epochs)) - 1
        loss = model(features_s, labels_s, features_t, alpha)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        labels_t = labels_t.to(device=args.device)
        logits = model.inference(features_t)
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels_t)
    return correct.item() * 1.0 / len(labels_t)


def GRADE_main(args, data_s, data_t):
    g_s, labels_s, features_s = data_s
    g_t, labels_t, features_t = data_t

    acc = train(args, g_s, features_s, labels_s, g_t, features_t, labels_t)
    return acc
