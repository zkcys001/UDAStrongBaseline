import torch
from torch import nn
import torch.nn.functional as F

eps = 1e-7


class NCECriterion(nn.Module):
    """
    Eq. (12): L_{NCE}
    """
    def __init__(self, n_data):
        super(NCECriterion, self).__init__()
        self.n_data = n_data

    def forward(self, x):
        bsz = x.shape[0]
        m = x.size(1) - 1

        # noise distribution
        Pn = 1 / float(self.n_data)

        # loss for positive pair
        P_pos = x.select(1, 0)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()

        # loss for K negative pair
        P_neg = x.narrow(1, 1, m)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()

        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz

        return loss


class NCESoftmaxLoss(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""
    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, is_pos=None):

        bsz = x.shape[0]
        x = x.squeeze()
        label = torch.zeros([bsz]).cuda().long()
        loss = self.criterion(x, label)
        return loss


class MultiSoftmaxLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.criterion = nn.KLDivLoss(reduction='batchmean')
        # self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.NLLLoss(reduction='mean')

    def forward(self, x, is_neg):
        bsz = x.shape[0]
        # ce_loss = self.criterion(x, torch.zeros([bsz]).cuda().long())
        x = x.squeeze()
        x = torch.exp(x)

        is_neg = is_neg.float()
        is_need = torch.cat((torch.ones([bsz, 1], dtype=torch.float).cuda(), is_neg), dim=1)

        neg_div = (x * is_need).sum(dim=1, keepdim=True)
        x_logit = x[:,0] / neg_div
        x_logit = -torch.log(x_logit)
        loss = x_logit.mean()

        # x_mask = x_logit * is_pos.float()
        # num_pos = is_pos.sum(dim=1, keepdim=True).float()
        # x_mask = x_mask / num_pos
        # loss = x_logit.sum(dim=1).mean(dim=0)
        return loss

        # loss = 0
        # for i in range(bsz):
        #     tmp_loss = 0
        #     pos_inds = torch.where(is_pos[i] == 1)[0].tolist()
        #     num_pos = len(pos_inds)
        #     for j in pos_inds:
        #         tmp_loss -= torch.log(x[i, j] / (neg_div[i][0] + x[i, j]))
        #     loss += (tmp_loss / num_pos)
        # loss = loss / bsz
        #
        # print(loss)
        # print(fast_loss)
        # from ipdb import set_trace; set_trace()

        # print(ce_loss)
        # print(loss)

    # def forward(self, x, is_pos):
    #     is_pos = is_pos.float()
    #     bsz = x.shape[0]
    #     x = x.squeeze()
    #
    #     label = torch.zeros([bsz]).cuda().long()
    #     # loss = self.criterion1(x, ce_label)
    #
    #     # from ipdb import set_trace; set_trace()
    #     # is_neg = 1 - is_pos[:, 1:]
    #     x = F.softmax(x, dim=1)
    #     x = (x * is_pos).sum(dim=1, keepdim=True)
    #     # neg_logit = (x * is_neg)
    #     # x = torch.cat((pos_logit, x[:, 1:]), dim=1)  # [bsz, 16385]
    #     # x = torch.log(x)
    #
    #     loss = self.criterion(x.log(), label)
    #     return loss

        # x = F.softmax(x, dim=1)
        # label = torch.cat((torch.ones([bsz, 1], dtype=torch.float32).cuda(), is_pos), dim=1)  # (bsz, dim)
        # label = F.softmax(label, dim=1)
        # label = label / label.sum(dim=1, keepdim=True)

        # loss = torch.sum(x * torch.log(1e-9 + x / (label + 1e-9)), dim=1).mean(dim=0)
        # loss = torch.sum(x * (1e-9 + torch.log(x) - torch.log(label + 1e-9)), dim=1).mean(dim=0)
        # from ipdb import set_trace; set_trace()
        # loss = self.criterion(x, label)
        # return loss
