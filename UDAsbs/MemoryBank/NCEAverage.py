import torch
from torch import nn
from torch.nn import functional as F
from .alias_multinomial import AliasMethod
import math


class NCEAverage(nn.Module):

    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5, use_softmax=False):
        super(NCEAverage, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K
        self.use_softmax = use_softmax

        self.register_buffer('params', torch.tensor([K, T, -1, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory_l', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_ab', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, l, ab, y, idx=None):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_l = self.params[2].item()
        Z_ab = self.params[3].item()

        momentum = self.params[4].item()
        batchSize = l.size(0)
        outputSize = self.memory_l.size(0)
        inputSize = self.memory_l.size(1)

        # score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)
        # sample
        weight_l = torch.index_select(self.memory_l, 0, idx.view(-1)).detach()
        weight_l = weight_l.view(batchSize, K + 1, inputSize)
        out_ab = torch.bmm(weight_l, ab.view(batchSize, inputSize, 1))
        # sample
        weight_ab = torch.index_select(self.memory_ab, 0, idx.view(-1)).detach()
        weight_ab = weight_ab.view(batchSize, K + 1, inputSize)
        out_l = torch.bmm(weight_ab, l.view(batchSize, inputSize, 1))

        if self.use_softmax:
            out_ab = torch.div(out_ab, T)
            out_l = torch.div(out_l, T)
            out_l = out_l.contiguous()
            out_ab = out_ab.contiguous()
        else:
            out_ab = torch.exp(torch.div(out_ab, T))
            out_l = torch.exp(torch.div(out_l, T))
            # set Z_0 if haven't been set yet,
            # Z_0 is used as a constant approximation of Z, to scale the probs
            if Z_l < 0:
                self.params[2] = out_l.mean() * outputSize
                Z_l = self.params[2].clone().detach().item()
                print("normalization constant Z_l is set to {:.1f}".format(Z_l))
            if Z_ab < 0:
                self.params[3] = out_ab.mean() * outputSize
                Z_ab = self.params[3].clone().detach().item()
                print("normalization constant Z_ab is set to {:.1f}".format(Z_ab))
            # compute out_l, out_ab
            out_l = torch.div(out_l, Z_l).contiguous()
            out_ab = torch.div(out_ab, Z_ab).contiguous()

        # # update memory
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_l, 0, y.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(l, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_l = l_pos.div(l_norm)
            self.memory_l.index_copy_(0, y, updated_l)

            ab_pos = torch.index_select(self.memory_ab, 0, y.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(ab, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_ab = ab_pos.div(ab_norm)
            self.memory_ab.index_copy_(0, y, updated_ab)

        return out_l, out_ab


# =========================
# InsDis and MoCo
# =========================

class MemoryInsDis(nn.Module):
    """Memory bank with instance discrimination"""
    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5, use_softmax=False):
        super(MemoryInsDis, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K
        self.use_softmax = use_softmax

        self.register_buffer('params', torch.tensor([K, T, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, x, y, idx=None):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z = self.params[2].item()
        momentum = self.params[3].item()

        batchSize = x.size(0)
        outputSize = self.memory.size(0)
        inputSize = self.memory.size(1)

        # score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)

        # sample
        weight = torch.index_select(self.memory, 0, idx.view(-1))
        weight = weight.view(batchSize, K + 1, inputSize)
        out = torch.bmm(weight, x.view(batchSize, inputSize, 1))

        if self.use_softmax:
            out = torch.div(out, T)
            out = out.squeeze().contiguous()
        else:
            out = torch.exp(torch.div(out, T))
            if Z < 0:
                self.params[2] = out.mean() * outputSize
                Z = self.params[2].clone().detach().item()
                print("normalization constant Z is set to {:.1f}".format(Z))
            # compute the out
            out = torch.div(out, Z).squeeze().contiguous()

        # # update memory
        with torch.no_grad():
            weight_pos = torch.index_select(self.memory, 0, y.view(-1))
            weight_pos.mul_(momentum)
            weight_pos.add_(torch.mul(x, 1 - momentum))
            weight_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_weight = weight_pos.div(weight_norm)
            self.memory.index_copy_(0, y, updated_weight)

        return out

class MemoryMoCo(nn.Module):
    """Fixed-size queue with momentum encoder"""
    def __init__(self, inputSize, outputSize, K, T=0.07, use_softmax=False):
        super(MemoryMoCo, self).__init__()
        self.outputSize = outputSize
        self.inputSize = inputSize
        self.queueSize = K
        self.T = T
        self.index = 0
        self.use_softmax = use_softmax
        self.register_buffer('params', torch.tensor([-1]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory', torch.rand(self.queueSize, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('index_memory', torch.ones((len(self.target_label),self.queueSize),
                                                        dtype=torch.long).fill_(-1))
        print('using queue shape: ({},{})'.format(self.queueSize, inputSize))

    def update(self, target_label):
        pass
    def forward(self, q, k):
        batchSize = q.shape[0]
        k = k.detach()

        Z = self.params[0].item()
        q = F.normalize(q, 2)
        k = F.normalize(k, 2)
        # pos logit
        l_pos = torch.bmm(q.view(batchSize, 1, -1), k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)
        # neg logit
        queue = self.memory.clone()
        l_neg = torch.mm(queue.detach(), q.transpose(1, 0))
        l_neg = l_neg.transpose(0, 1)

        out = torch.cat((l_pos, l_neg), dim=1)

        if self.use_softmax:
            out = torch.div(out, self.T)
            out = out.squeeze().contiguous()
        else:
            out = torch.exp(torch.div(out, self.T))
            if Z < 0:
                self.params[0] = out.mean() * self.outputSize
                Z = self.params[0].clone().detach().item()
                print("normalization constant Z is set to {:.1f}".format(Z))
            # compute the out
            out = torch.div(out, Z).squeeze().contiguous()

        # # update memory
        with torch.no_grad():
            out_ids = torch.arange(batchSize).cuda()
            out_ids += self.index
            out_ids = torch.fmod(out_ids, self.queueSize)
            out_ids = out_ids.long()
            self.memory.index_copy_(0, out_ids, k)
            self.index = (self.index + batchSize) % self.queueSize

        return out

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x
from functools import reduce
def logsumexp(value, weight=1,dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(weight*torch.exp(value - m))
        # if isinstance(sum_exp, numpy):
        #     return m + math.log(sum_exp)
        # else:
        return m + torch.log(sum_exp)


class MemoryMoCo_id(nn.Module):
    """Fixed-size queue with momentum encoder"""
    def __init__(self, inputSize, outputSize, K, index2label, choice_c=1, T=0.07, use_softmax=False, cluster_num=0):
        super(MemoryMoCo_id, self).__init__()
        self.outputSize = outputSize
        self.inputSize = inputSize
        self.queueSize = K
        self.T = T
        self.index = 0
        self.use_softmax = use_softmax
        self.register_buffer('params', torch.tensor([-1]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory', torch.rand(self.queueSize, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('index_memory', torch.ones(self.queueSize, dtype=torch.long).fill_(-1))
        print('Using queue shape: ({},{})'.format(self.queueSize, inputSize))

        self.choice_c=choice_c
        self.index_pl = -1#3-cluster_num if cluster_num<=3 else 0
        self.index2label = index2label
        self.m = 0.25
        self.gamma = 128

    def posandneg(self, index,batchSize,index_choice):

        # pseudo logit
        # pseudo_label = [torch.tensor([self.index2label[j][i.item()] for i in index], dtype=torch.long).cuda()
        #                 for j in range(4)]
        # pseudo_label = sum(pseudo_label) / 4.0
        # pseudo_label=reduce(lambda x, y: x * y, pseudo_label)
        pseudo_label = torch.tensor([self.index2label[index_choice][i.item()] for i in index], dtype=torch.long).cuda()
        pseudo_label = pseudo_label.unsqueeze(1).expand(batchSize, self.queueSize)

        # memory_label = [
        #     torch.tensor([self.index2label[j][i.item()] if i.item() != -1 else -1 for i in self.index_memory],
        #                  dtype=torch.long).cuda()
        #     for j in range(4)]
        # memory_label = sum(memory_label) / 4.0
        # memory_label = reduce(lambda x, y: x * y, memory_label)
        memory_label = torch.tensor([self.index2label[index_choice][i.item()] if i.item() != -1 else -1
                                     for i in self.index_memory], dtype=torch.long).cuda()
        memory_label = memory_label.unsqueeze(0).expand(batchSize, self.queueSize)

        is_pos = pseudo_label.eq(memory_label).float()
        # is_pos_weight = torch.cat((torch.ones([batchSize, 1], dtype=torch.float).cuda(), is_pos), dim=1)
        # weight = torch.cat(
        #     (torch.ones([batchSize, 1], dtype=torch.float).cuda(), is_pos / is_pos_weight.sum(1, keepdim=True)), dim=1)
        # is_pos = is_pos_weight
        is_pos = torch.cat((torch.ones([batchSize, 1], dtype=torch.float).cuda(), is_pos), dim=1)
        is_neg = pseudo_label.ne(memory_label).float()
        is_neg = torch.cat((torch.zeros([batchSize, 1], dtype=torch.float).cuda(), is_neg), dim=1)
        # is_neg = torch.cat((torch.zeros([batchSize, 1], dtype=torch.float).cuda(), is_neg), dim=1)
        return is_pos, is_neg


    def update(self,q1, q2, index):
        batchSize = q1.shape[0]
        with torch.no_grad():
            q1 = q1.detach()
            q2 = q2.detach()
            out_ids = torch.arange(batchSize).cuda()
            out_ids += self.index
            out_ids = torch.fmod(out_ids, self.queueSize)
            out_ids = out_ids.long()
            self.memory.index_copy_(0, out_ids, (q1+q2)/2.0)
            self.index_memory.index_copy_(0, out_ids, index)
            self.index = (self.index + batchSize) % self.queueSize

    def forward(self, q1, q2, index, epoch=0):
        batchSize = q1.shape[0]

        q1 = normalize(q1, axis=-1)
        q2 = normalize(q2, axis=-1)

        #is_pos0, is_neg0 = self.posandneg(index, batchSize, 0)
        is_pos1, is_neg1 = self.posandneg(index, batchSize, self.choice_c)
        #is_pos2, is_neg2 = self.posandneg(index, batchSize, 2)
        #is_pos3, is_neg3 = self.posandneg(index, batchSize, 3)
        is_pos =is_pos1# (is_pos0 + is_pos1 + is_pos2 + is_pos3)/4.0
        is_neg =is_neg1# (is_neg0 + is_neg1 + is_neg2 + is_neg3)/4.0

        queue = self.memory.clone()
        l_logist = torch.matmul(queue.detach(), ((q1+q2)/2.0).transpose(1, 0))
        l_logist = l_logist.transpose(0, 1).contiguous()  # (bs, queue_size)

        # pos logit for self
        l_pos = torch.bmm(q1.view(batchSize, 1, -1), q2.view(batchSize, -1, 1))
        l_pos_self = l_pos.contiguous().view(batchSize, 1)

        sim_mat = torch.cat((l_pos_self, l_logist), dim=1)

        s_p = sim_mat * is_pos#0#[is_pos].contiguous()#.view(batchSize, -1)
        # s_p = torch.div(s_p, self.T)
        
        s_n = sim_mat * is_neg#3#[is_neg].contiguous()#.view(batchSize, -1)

        alpha_p = F.relu(-s_p.detach() + 1 + self.m)
        alpha_n = F.relu(s_n.detach() + self.m)
        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - self.gamma * alpha_p * (s_p - delta_p)
        logit_n = self.gamma * alpha_n * (s_n - delta_n)

        # logit_p_1 = torch.exp(logit_p * is_pos) * is_pos
        # logit_p_2 = logit_p_1.sum(1)
        # logit_p_3 = torch.log(logit_p_2+ 1e-16)
        # logit_n = torch.log((torch.exp(logit_n) * is_neg).sum(1) + 1e-16)
        # loss = F.softplus(logit_p+logit_n).mean()
        loss = F.softplus(logsumexp(logit_p - 99999.0 * is_neg, dim=1) +
                          logsumexp(logit_n - 99999.0 * is_pos, dim=1)).mean() / 18.0  # weight,
        # loss = F.softplus(logsumexp(logit_p-99999.0*is_neg0,is_pos, dim=1) +
        #                   logsumexp(logit_n-99999.0*is_pos3,is_neg, dim=1)).mean()/18.0#weight,

        # update memory
        with torch.no_grad():
            q1 = q1.detach()
            q2 = q2.detach()
            out_ids = torch.arange(batchSize).cuda()
            out_ids += self.index
            out_ids = torch.fmod(out_ids, self.queueSize)
            out_ids = out_ids.long()
            self.memory.index_copy_(0,  out_ids, (q1+q2)/2.0)
            self.index_memory.index_copy_(0, out_ids, index)
            self.index = (self.index + batchSize) % self.queueSize

        return loss
