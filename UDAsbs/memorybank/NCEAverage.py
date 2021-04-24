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
        return m + torch.log(torch.sum(weight*torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(weight*torch.exp(value - m))

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

    def posandneg(self, index, batchSize, index_choice):

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
        #loss= F.softplus(logsumexp(logit_p - 99999.0 * is_neg0,is_pos, dim=1) +
        #                 logsumexp(logit_n - 99999.0 * is_pos3,is_neg, dim=1)).mean()/18.0#weight,

        # update memory
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

        return loss

class onlinememory(nn.Module):
    """Fixed-size queue with momentum encoder"""

    def __init__(self, inputSize, outputSize, sour_numclass, K, index2label, choice_c=1, T=0.07, use_softmax=False, cluster_num=0):
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

        self.momentum=0.2
        ################
        #his loss
        num_steps=151
        self.step = 2 / (num_steps - 1)
        self.eps = 1 / num_steps
        self.t = torch.arange(-1, 1 + self.step, self.step).view(-1, 1).cuda()
        self.tsize = self.t.size()[0]
        ###############
        # smooth ap loss
        self.anneal = 0.01
        self.num_id=16

    def memo_contr_loss(self,index,q1):
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


        outputs = F.log_softmax(sim_mat, dim=1)
        loss = - (is_pos * outputs)
        loss = loss.sum(dim=1)
        loss = loss.mean(dim=0)
        return loss

    def memo_circle_loss(self,index,q1,uncer):
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

        # exp_variance = exp_variance.detach()
        # exp_variance = exp_variance.unsqueeze(1).expand(batchSize, self.queueSize)

        s_p = sim_mat * is_pos
        #s_p = torch.div(s_p, self.T)
        s_n = sim_mat * is_neg #* exp_variance

        exp_variance = 1#(uncer.unsqueeze(1).expand(batchSize, self.queueSize) + self.uncer.clone().unsqueeze(0).expand(batchSize, self.queueSize))/2.0

        alpha_p = F.relu(-s_p.detach() + 1 + self.m)
        alpha_n = F.relu(s_n.detach() + self.m)
        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - self.gamma * alpha_p * (s_p - delta_p)
        logit_n = self.gamma * alpha_n * (s_n - delta_n)
        # ,weight=exp_variance
        loss = (F.softplus(logsumexp(logit_p - 99999.0 * is_neg,weight=exp_variance, dim=1) +
                          logsumexp(logit_n - 99999.0 * is_pos,weight=exp_variance, dim=1))).mean()
        return loss

    def memo_center_circle_loss(self,index,q1):
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

        queue = self.memory[:self.sour_numclass,:].clone()
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

    def his_loss(self,classes,features):
        classes = torch.tensor([self.index2label[0][i.item()] for i in classes],
                                    dtype=torch.long).cuda()
        def histogram(inds, size):
            s_repeat_ = s_repeat.clone()
            indsa = (s_repeat_floor - (self.t - self.step) > -self.eps) & (
                        s_repeat_floor - (self.t - self.step) < self.eps) & inds
            assert indsa.nonzero().size()[0] == size, ('Another number of bins should be used')
            zeros = torch.zeros((1, indsa.size()[1])).byte()
            if self.cuda:
                zeros = zeros.cuda()
            indsb = torch.cat((indsa, zeros))[1:, :]
            s_repeat_[~(indsb | indsa)] = 0
            # indsa corresponds to the first condition of the second equation of the paper
            s_repeat_[indsa] = (s_repeat_ - self.t + self.step)[indsa] / self.step
            # indsb corresponds to the second condition of the second equation of the paper
            s_repeat_[indsb] = (-s_repeat_ + self.t + self.step)[indsb] / self.step

            return s_repeat_.sum(1) / size
        classes_size = classes.size()[0]
        classes_eq = (classes.repeat(classes_size, 1) == classes.view(-1, 1).repeat(1, classes_size)).data
        dists = torch.mm(features, features.transpose(0, 1))
        assert ((dists > 1 + self.eps).sum().item() + (
                    dists < -1 - self.eps).sum().item()) == 0, 'L2 normalization should be used'
        s_inds = torch.triu(torch.ones(classes_eq.size()), 1).byte()
        if self.cuda: s_inds = s_inds.cuda()
        pos_inds = classes_eq[s_inds].repeat(self.tsize, 1)#18001,2016
        neg_inds = ~classes_eq[s_inds].repeat(self.tsize, 1)#18001,2016
        pos_size = classes_eq[s_inds].sum().item()
        neg_size = (~classes_eq[s_inds]).sum().item()
        s = dists[s_inds].view(1, -1)
        s_repeat = s.repeat(self.tsize, 1)
        s_repeat_floor = (torch.floor(s_repeat.data / self.step) * self.step).float()

        histogram_pos = histogram(pos_inds, pos_size)
        assert_almost_equal(histogram_pos.sum().item(), 1, decimal=1,
                            err_msg='Not good positive histogram', verbose=True)
        histogram_neg = histogram(neg_inds, neg_size)
        assert_almost_equal(histogram_neg.sum().item(), 1, decimal=1,
                            err_msg='Not good negative histogram', verbose=True)
        histogram_pos_repeat = histogram_pos.view(-1, 1).repeat(1, histogram_pos.size()[0])
        histogram_pos_inds = torch.tril(torch.ones(histogram_pos_repeat.size()), -1).byte()
        if self.cuda:
            histogram_pos_inds = histogram_pos_inds.cuda()
        histogram_pos_repeat[histogram_pos_inds] = 0
        histogram_pos_cdf = histogram_pos_repeat.sum(0)
        loss = torch.sum(histogram_neg * histogram_pos_cdf)

        return loss

    def smooth_ap(self, targets,embedding):
        targets= torch.tensor([self.index2label[0][i.item()] for i in targets],
                                    dtype=torch.long).cuda()
        # For distributed training, gather all features from different process.

        all_embedding = self.memory.clone().detach()
        all_targets = torch.tensor(
            [self.index2label[0][i.item()] if i.item() != -1 else -1 for i in self.index_memory],
            dtype=torch.long).cuda()

        sim_dist = torch.matmul(embedding, all_embedding.t())
        N, M = sim_dist.size()

        # Compute the mask which ignores the relevance score of the query to itself
        mask_indx = 1.0 - torch.eye(M, device=sim_dist.device)
        mask_indx = mask_indx.unsqueeze(dim=0).repeat(N, 1, 1)  # (N, M, M)

        # sim_dist -> N, 1, M -> N, M, N
        sim_dist_repeat = sim_dist.unsqueeze(dim=1).repeat(1, M, 1)  # (N, M, M)
        # sim_dist_repeat_t = sim_dist.t().unsqueeze(dim=1).repeat(1, N, 1)  # (N, N, M)

        # Compute the difference matrix
        sim_diff = sim_dist_repeat - sim_dist_repeat.permute(0, 2, 1)  # (N, M, M)

        # Pass through the sigmoid
        sim_sg = sigmoid(sim_diff, temp=self.anneal) * mask_indx

        # Compute all the rankings
        sim_all_rk = torch.sum(sim_sg, dim=-1) + 1  # (N, N)

        pos_mask = targets.view(N, 1).expand(N, M).eq(all_targets.view(M, 1).expand(M, N).t()).float()  # (N, M)

        pos_mask_repeat = pos_mask.unsqueeze(1).repeat(1, M, 1)  # (N, M, M)

        # Compute positive rankings
        pos_sim_sg = sim_sg * pos_mask_repeat
        sim_pos_rk = torch.sum(pos_sim_sg, dim=-1) + 1  # (N, N)

        # sum the values of the Smooth-AP for all instances in the mini-batch
        ap = 0
        group = N // self.num_id
        for ind in range(self.num_id):
            pos_divide = torch.sum(
                sim_pos_rk[(ind * group):((ind + 1) * group), (ind * group):((ind + 1) * group)] / (
                sim_all_rk[(ind * group):((ind + 1) * group), (ind * group):((ind + 1) * group)]))
            ap += pos_divide / torch.sum(pos_mask[ind * group]) / N
        return 1 - ap

    def _smooth_ap(self, targets,embedding):
        """Forward pass for all input predictions: preds - (batch_size x feat_dims) """

        # ------ differentiable ranking of all retrieval set ------
        embedding = F.normalize(embedding, dim=1)

        # For distributed training, gather all features from different process.
        sim_dist = torch.matmul(embedding, self.memory[:self.queueSize-self.sour_numclass,:].t().detach())
        N, M = sim_dist.size()

        # Compute the mask which ignores the relevance score of the query to itself
        mask_indx = 1.0 - torch.eye(M, device=sim_dist.device)
        mask_indx = mask_indx.unsqueeze(dim=0).repeat(N, 1, 1)  # (N, M, M)

        # sim_dist -> N, 1, M -> N, M, N
        sim_dist_repeat = sim_dist.unsqueeze(dim=1).repeat(1, M, 1)  # (N, M, M)

        # Compute the difference matrix
        sim_diff = sim_dist_repeat - sim_dist_repeat.permute(0, 2, 1)  # (N, M, M)

        # Pass through the sigmoid
        sim_sg = sigmoid(sim_diff, temp=self.anneal) * mask_indx

        # Compute all the rankings
        sim_all_rk = torch.sum(sim_sg, dim=-1) + 1  # (N, N)r

        targets = torch.tensor([self.index2label[0][i.item()] for i in targets],
                               dtype=torch.long).cuda()

        queue_label = torch.tensor([self.index2label[0][i.item()] if i.item() != -1
                                    else -1 for i in self.index_memory],
            dtype=torch.long).cuda()[self.sour_numclass:]

        pos_mask = targets.view(N, 1).expand(N, M).eq(queue_label.view(M, 1).expand(M, N).t()).float()  # (N, M)

        pos_mask_repeat = pos_mask.unsqueeze(1).repeat(1, M, 1)  # (N, M, M)

        # Compute positive rankings
        pos_sim_sg = sim_sg * pos_mask_repeat
        sim_pos_rk = torch.sum(pos_sim_sg, dim=-1) + 1  # (N, N)

        # sum the values of the Smooth-AP for all instances in the mini-batch
        ap = 0
        group = N // self.num_id
        for ind in range(self.num_id):
            pos_divide = torch.sum(
                sim_pos_rk[(ind * group):((ind + 1) * group), (ind * group):((ind + 1) * group)] / (
                sim_all_rk[(ind * group):((ind + 1) * group), (ind * group):((ind + 1) * group)]))
            ap += pos_divide / torch.sum(pos_mask[ind * group]) / N
        return 1 - ap

    def forward(self, q1, q2, tar_tri, tar_tri_ema, index, sour_labels, uncer=None, epoch=0):
        batchSize = q1.shape[0]

        # tar_tri = normalize(tar_tri, axis=-1)
        q1 = normalize(q1, axis=-1)
        q2 = normalize(q2, axis=-1)

        # loss_q1 = self.memo_contr_loss(index+self.sour_numclass, q1)
        loss_q1 = self.memo_circle_loss(index + self.sour_numclass, q1, uncer)
        # loss_q1 = self._smooth_ap(index + self.sour_numclass, q1)
        loss_q2 = self.memo_center_circle_loss(sour_labels, q2)

        # with torch.no_grad():
        #     queue = self.memory[:self.sour_numclass, :].clone()
        #     ml_sour = torch.matmul(tar_tri,queue.transpose(1, 0).detach())
        #     ml_sour_ema = torch.matmul(tar_tri_ema, queue.transpose(1, 0).detach())
        # update memory
        with torch.no_grad():
            q1 = q1.detach()
            out_ids = torch.arange(batchSize).cuda()
            out_ids += self.index
            out_ids = torch.fmod(out_ids, self.queueSize-self.sour_numclass)
            out_ids = (out_ids+self.sour_numclass).long()
            self.memory.index_copy_(0, out_ids, q1)
            self.index_memory.index_copy_(0, out_ids, index + self.sour_numclass)
            self.uncer.index_copy_(0, out_ids, uncer)
            self.index = (self.index + batchSize) % (self.queueSize-self.sour_numclass)
            for x, y in zip(q2, sour_labels):
                self.memory[y] = self.momentum * self.memory[y] + (1. - self.momentum) * x
                self.memory[y] /= self.memory[y].norm()

        return loss_q1, loss_q2, None, None
