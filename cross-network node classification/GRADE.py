import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Function
import numpy as np
from dgl.nn.pytorch import GraphConv, SAGEConv, GINConv


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)


def mmd_rbf_noaccelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class GRADE(nn.Module):
    def __init__(self, g_s, g_t, in_feats, n_hidden, n_classes, n_layers, dropout, activation=F.relu, disc="JS"):
        super(GRADE, self).__init__()
        self.disc = disc
        self.g_s = g_s
        self.g_t = g_t
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # self.layers.append(GraphConv(n_hidden, n_classes, activation=None))
        self.fc = nn.Linear(n_hidden, n_classes)
        self.dropout = nn.Dropout(p=dropout)

        if disc == "JS":
            self.discriminator = nn.Sequential(
                nn.Linear(n_hidden*n_layers+n_classes, 2)
            )
        else:
            self.discriminator = nn.Sequential(
                nn.Linear(n_hidden * n_layers + n_classes * 2, 2)
            )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, features_s, labels_s, features_t, alpha=1.0):
        s_f = []
        t_f = []
        for i, layer in enumerate(self.layers):
            features_s = self.dropout(features_s)
            features_t = self.dropout(features_t)
            features_s = layer(self.g_s, features_s)
            features_t = layer(self.g_t, features_t)
            s_f.append(features_s)
            t_f.append(features_t)
        features_s = self.fc(features_s)
        features_t = self.fc(features_t)
        s_f.append(features_s)
        t_f.append(features_t)
        preds_s = torch.log_softmax(features_s, dim=-1)
        class_loss = F.nll_loss(preds_s, labels_s)

        s_f = torch.cat(s_f, dim=1)
        t_f = torch.cat(t_f, dim=1)
        domain_loss = 0.
        if self.disc == "JS":
            domain_preds = self.discriminator(ReverseLayerF.apply(torch.cat([s_f, t_f], dim=0), alpha))
            domain_labels = np.array([0] * features_s.shape[0] + [1] * features_t.shape[0])
            domain_labels = torch.tensor(domain_labels, requires_grad=False, dtype=torch.long, device=features_s.device)
            domain_loss = self.criterion(domain_preds, domain_labels)
        elif self.disc == "MMD":
            mind = min(s_f.shape[0], t_f.shape[0])
            domain_loss = mmd_rbf_noaccelerate(s_f[:mind], t_f[:mind])
        elif self.disc == "C":
            ratio = 8
            s_l_f = torch.cat([s_f, ratio * self.one_hot_embedding(labels_s)], dim=1)
            t_l_f = torch.cat([t_f, ratio * F.softmax(features_t, dim=1)], dim=1)
            domain_preds = self.discriminator(ReverseLayerF.apply(torch.cat([s_l_f, t_l_f], dim=0), alpha))
            domain_labels = np.array([0] * features_s.shape[0] + [1] * features_t.shape[0])
            domain_labels = torch.tensor(domain_labels, requires_grad=False, dtype=torch.long, device=features_s.device)
            domain_loss = self.criterion(domain_preds, domain_labels)
        loss = class_loss + domain_loss * 0.01
        return loss

    def inference(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(self.g_t, x)
        x = self.fc(x)
        return torch.log_softmax(x, dim=-1)

    def one_hot_embedding(self, labels):
        y = torch.eye(self.n_classes, device=labels.device)
        return y[labels]

