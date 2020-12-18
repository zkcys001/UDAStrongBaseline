import torch
from torch import nn
from torch.nn import functional as F
import math
from numpy.testing import assert_almost_equal


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def sigmoid(tensor, temp=1.0):
    exponent = -tensor / temp
    exponent = torch.clamp(exponent, min=-50, max=50)
    y = 1.0 / (1.0 + torch.exp(exponent))
    return y


def logsumexp(value, weight = 1, dim=None, keepdim=False):
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


class onlinememory(nn.Module):
    """Fixed-size queue with momentum encoder"""

    def __init__(self, inputSize, outputSize, sour_numclass, K, index2label, choice_c=1, T=0.07, use_softmax=False,
                 cluster_num=0):
        super(onlinememory, self).__init__()
        self.outputSize = outputSize
        self.inputSize = inputSize
        self.sour_numclass = sour_numclass
        self.queueSize = K
        self.T = T
        self.index = 0
        self.use_softmax = use_softmax
        self.register_buffer('params', torch.tensor([-1]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory', torch.rand(self.queueSize, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('index_memory', torch.ones(self.queueSize, dtype=torch.long).fill_(-1))
        self.register_buffer('uncer', torch.ones(self.queueSize, dtype=torch.float).fill_(1))
        print('Using queue shape: ({},{})'.format(self.queueSize, inputSize))

        # self.register_buffer('sour_memory', torch.rand(self.sour_numclass, inputSize).mul_(2 * stdv).add_(-stdv))
        # self.register_buffer('sour_index_memory', torch.ones(self.sour_numclass, dtype=torch.long).fill_(-1))
        # print('Using queue shape: ({},{})'.format(self.sour_numclass, inputSize))

        self.choice_c = choice_c
        self.index_pl = -1  # 3-cluster_num if cluster_num<=3 else 0
        self.index2label = index2label

        self.m = 0.25
        self.gamma = 128

        self.momentum = 0.2
        ################
        # his loss
        num_steps = 151
        self.step = 2 / (num_steps - 1)
        self.eps = 1 / num_steps
        self.t = torch.arange(-1, 1 + self.step, self.step).view(-1, 1).cuda()
        self.tsize = self.t.size()[0]
        ###############
        # smooth ap loss
        self.anneal = 0.01
        self.num_id = 16



    def memo_circle_loss(self, index, q1, uncer):
        batchSize = q1.shape[0]
        # import ipdb;ipdb.set_trace()
        pseudo_label = torch.tensor([self.index2label[self.choice_c][i.item()] for i in index],
                                    dtype=torch.long).cuda()
        pseudo_label = pseudo_label.unsqueeze(1).expand(batchSize, self.queueSize)

        memory_label = torch.tensor(
            [self.index2label[self.choice_c][i.item()] if i.item() != -1 else -1 for i in self.index_memory],
            dtype=torch.long).cuda()
        memory_label = memory_label.unsqueeze(0).expand(batchSize, self.queueSize)

        is_pos = pseudo_label.eq(memory_label).float()
        is_neg = pseudo_label.ne(memory_label).float()

        queue = self.memory.clone()
        l_logist = torch.matmul(queue.detach(), (q1).transpose(1, 0))
        l_logist = l_logist.transpose(0, 1).contiguous()
        sim_mat = l_logist

        s_p = sim_mat * is_pos
        s_n = sim_mat * is_neg

        if uncer:
            exp_variance = (uncer.unsqueeze(1).expand(batchSize, self.queueSize) +self.uncer.clone().unsqueeze(0).expand(batchSize, self.queueSize)) / 2.0
        else:
            exp_variance=1
        alpha_p = F.relu(-s_p.detach() + 1 + self.m)
        alpha_n = F.relu(s_n.detach() + self.m)
        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - self.gamma * alpha_p * (s_p - delta_p)
        logit_n = self.gamma * alpha_n * (s_n - delta_n)
        # ,weight=exp_variance
        loss = (F.softplus(logsumexp(logit_p - 99999.0 * is_neg, weight=exp_variance, dim=1) +
                           logsumexp(logit_n - 99999.0 * is_pos, weight=exp_variance, dim=1))).mean()/ 18.0
        return loss

    def memo_center_circle_loss(self, index, q1):
        batchSize = q1.shape[0]

        pseudo_label = torch.tensor([self.index2label[self.choice_c][i.item()] for i in index],
                                    dtype=torch.long).cuda()
        pseudo_label = pseudo_label.unsqueeze(1).expand(batchSize, self.sour_numclass)
        # pseudo_label = index.expand(batchSize, self.sour_numclass)
        memory_label = torch.tensor(
            [self.index2label[self.choice_c][i] for i in range(self.sour_numclass)],
            dtype=torch.long).cuda()
        memory_label = memory_label.unsqueeze(0).expand(batchSize, self.sour_numclass)

        is_pos = pseudo_label.eq(memory_label).float()
        is_neg = pseudo_label.ne(memory_label).float()

        queue = self.memory[:self.sour_numclass, :].clone()
        l_logist = torch.matmul(queue.detach(), (q1).transpose(1, 0))
        l_logist = l_logist.transpose(0, 1).contiguous()
        sim_mat = l_logist

        s_p = sim_mat * is_pos

        s_n = sim_mat * is_neg

        alpha_p = F.relu(-s_p.detach() + 1 + self.m)
        alpha_n = F.relu(s_n.detach() + self.m)
        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - self.gamma * alpha_p * (s_p - delta_p)
        logit_n = self.gamma * alpha_n * (s_n - delta_n)
        loss = F.softplus(logsumexp(logit_p - 99999.0 * is_neg, dim=1) +
                          logsumexp(logit_n - 99999.0 * is_pos, dim=1)).mean() / 18.0
        return loss



    def forward(self, q1, q2, index, tar_tri, tar_tri_ema, sour_labels, uncer=None, epoch=0):
        batchSize = q1.shape[0]

        # tar_tri = normalize(tar_tri, axis=-1)
        q1 = normalize(q1, axis=-1)
        q2 = normalize(q2, axis=-1)
        loss_q1 = self.memo_circle_loss(index + self.sour_numclass, q1, uncer)
        loss_q2 = self.memo_center_circle_loss(sour_labels, q2)

        with torch.no_grad():
            q1 = q1.detach()
            out_ids = torch.arange(batchSize).cuda()
            out_ids += self.index
            out_ids = torch.fmod(out_ids, self.queueSize - self.sour_numclass)
            out_ids = (out_ids + self.sour_numclass).long()
            self.memory.index_copy_(0, out_ids, q1)
            self.index_memory.index_copy_(0, out_ids, index + self.sour_numclass)
            if uncer:
                self.uncer.index_copy_(0, out_ids, uncer)
            self.index = (self.index + batchSize) % (self.queueSize - self.sour_numclass)
            for x, y in zip(q2, sour_labels):
                self.memory[y] = self.momentum * self.memory[y] + (1. - self.momentum) * x
                self.memory[y] /= self.memory[y].norm()

        return loss_q1, loss_q2, None, None
