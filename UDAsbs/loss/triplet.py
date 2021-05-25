from __future__ import absolute_import

import torch
from torch import nn
import torch.nn.functional as F


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def cosine_dist(x, y):
    bs1, bs2 = x.size(0), y.size(0)
    frac_up = torch.matmul(x, y.transpose(0, 1))
    frac_down = (torch.sqrt(torch.sum(torch.pow(x, 2), 1))).view(bs1, 1).repeat(1, bs2) * \
                (torch.sqrt(torch.sum(torch.pow(y, 2), 1))).view(1, bs2).repeat(bs1, 1)
    cosine = frac_up / frac_down
    return 1 - cosine


from functools import reduce


def _batch_hard(mat_distance, mat_similarity, indice=False):
    # mat_similarity=reduce(lambda x, y: x * y, mat_similaritys)
    # mat_similarity=mat_similaritys[0]*mat_similaritys[1]*mat_similaritys[2]*mat_similaritys[3]
    sorted_mat_distance, positive_indices = torch.sort(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1,
                                                       descending=True)
    hard_p = sorted_mat_distance[:, 0]
    hard_p_indice = positive_indices[:, 0]
    sorted_mat_distance, negative_indices = torch.sort(mat_distance + (9999999.) * (mat_similarity), dim=1,
                                                       descending=False)
    hard_n = sorted_mat_distance[:, 0]
    hard_n_indice = negative_indices[:, 0]
    if (indice):
        return hard_p, hard_n, hard_p_indice, hard_n_indice
    return hard_p, hard_n


class TripletLoss(nn.Module):
    '''
	Compute Triplet loss augmented with Batch Hard
	Details can be seen in 'In defense of the Triplet Loss for Person Re-Identification'
	'''

    def __init__(self, margin, normalize_feature=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.normalize_feature = normalize_feature
        self.margin_loss = nn.MarginRankingLoss(margin=margin).cuda()

    def forward(self, emb, label):
        if self.normalize_feature:
            # equal to cosine similarity
            emb = F.normalize(emb)
        mat_dist = euclidean_dist(emb, emb)
        # mat_dist = cosine_dist(emb, emb)
        assert mat_dist.size(0) == mat_dist.size(1)
        N = mat_dist.size(0)
        mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()

        dist_ap, dist_an = _batch_hard(mat_dist, mat_sim)
        assert dist_an.size(0) == dist_ap.size(0)
        y = torch.ones_like(dist_ap)
        loss = self.margin_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        return loss, prec


def logsumexp(value, weight=1, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(weight * torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(weight * torch.exp(value - m))

        return m + torch.log(sum_exp)


class SoftTripletLoss(nn.Module):

    def __init__(self, margin=None, normalize_feature=False, uncer_mode=0):
        super(SoftTripletLoss, self).__init__()
        self.margin = margin
        self.normalize_feature = normalize_feature
        self.uncer_mode = uncer_mode

    def forward(self, emb1, emb2, label, uncertainty):
        if self.normalize_feature:
            # equal to cosine similarity
            emb1 = F.normalize(emb1)
            emb2 = F.normalize(emb2)

        mat_dist = euclidean_dist(emb1, emb1)
        assert mat_dist.size(0) == mat_dist.size(1)
        N = mat_dist.size(0)

        # mat_sims=[]
        # for label in labels:
        # 	mat_sims.append(label.expand(N, N).eq(label.expand(N, N).t()).float())
        # mat_sim=reduce(lambda x, y: x + y, mat_sims)
        mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()
        dist_ap, dist_an, ap_idx, an_idx = _batch_hard(mat_dist, mat_sim, indice=True)
        assert dist_an.size(0) == dist_ap.size(0)
        triple_dist = torch.stack((dist_ap, dist_an), dim=1)
        triple_dist = F.log_softmax(triple_dist, dim=1)

        # mat_dist_ref = euclidean_dist(emb2, emb2)
        # dist_ap_ref = torch.gather(mat_dist_ref, 1, ap_idx.view(N,1).expand(N,N))[:,0]
        # dist_an_ref = torch.gather(mat_dist_ref, 1, an_idx.view(N,1).expand(N,N))[:,0]
        # triple_dist_ref = torch.stack((dist_ap_ref, dist_an_ref), dim=1)
        # triple_dist_ref = F.softmax(triple_dist_ref, dim=1).detach()

        # torch.gather
        if self.uncer_mode == 0:
            uncer_ap_ref = torch.gather(uncertainty, 0, ap_idx) + uncertainty
            uncer_an_ref = torch.gather(uncertainty, 0, an_idx) + uncertainty
        elif self.uncer_mode == 1:
            uncer_ap_ref = max(torch.gather(uncertainty, 0, ap_idx), uncertainty)
            uncer_an_ref = max(torch.gather(uncertainty, 0, an_idx), uncertainty)
        else:
            uncer_ap_ref = min(torch.gather(uncertainty, 0, ap_idx), uncertainty)
            uncer_an_ref = min(torch.gather(uncertainty, 0, an_idx), uncertainty)
        uncer = torch.stack((uncer_ap_ref, uncer_an_ref), dim=1).detach() / 2.0

        loss = (-uncer * triple_dist).mean(0).sum()#(uncer * triple_dist)[:,0].mean(0).sum()-(uncer * triple_dist)[:,1].mean(0).sum() #- triple_dist[:,1].mean()
        return loss


class SoftTripletLoss_vallia(nn.Module):

    def __init__(self, margin=None, normalize_feature=False):
        super(SoftTripletLoss_vallia, self).__init__()
        self.margin = margin
        self.normalize_feature = normalize_feature

    def forward(self, emb1, emb2, label):
        if self.normalize_feature:
            # equal to cosine similarity
            emb1 = F.normalize(emb1)
            emb2 = F.normalize(emb2)

        mat_dist = euclidean_dist(emb1, emb1)
        assert mat_dist.size(0) == mat_dist.size(1)
        N = mat_dist.size(0)

        mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()

        dist_ap, dist_an, ap_idx, an_idx = _batch_hard(mat_dist, mat_sim, indice=True)
        assert dist_an.size(0) == dist_ap.size(0)
        triple_dist = torch.stack((dist_ap, dist_an), dim=1)
        triple_dist = F.log_softmax(triple_dist, dim=1)
        if (self.margin is not None):
            loss = (- self.margin * triple_dist[:, 0] - (1 - self.margin) * triple_dist[:, 1]).mean()
            return loss

        mat_dist_ref = euclidean_dist(emb2, emb2)
        dist_ap_ref = torch.gather(mat_dist_ref, 1, ap_idx.view(N, 1).expand(N, N))[:, 0]
        dist_an_ref = torch.gather(mat_dist_ref, 1, an_idx.view(N, 1).expand(N, N))[:, 0]
        triple_dist_ref = torch.stack((dist_ap_ref, dist_an_ref), dim=1)
        triple_dist_ref = F.softmax(triple_dist_ref, dim=1).detach()

        loss = (- triple_dist_ref * triple_dist)[:, 1].mean(0).sum()
        return loss

      
