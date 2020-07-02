from __future__ import print_function, absolute_import
import time

import torch
import torch.nn as nn
from torch.nn import functional as F

from .evaluation_metrics import accuracy
from .loss import TripletLoss, CrossEntropyLabelSmooth, SoftTripletLoss, SoftEntropy, CrossBatchMemory
from .MemoryBank.NCECriterion import MultiSoftmaxLoss
from .utils.meters import AverageMeter


#
# class PreTrainer_personx(object):
#     def __init__(self, model, num_classes, margin=0.0):
#         super(PreTrainer_personx, self).__init__()
#         self.model = model
#         self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
#         self.criterion_triple = SoftTripletLoss(margin=margin).cuda()
#
#     def train(self, epoch, data_loader_source, optimizer, train_iters=200, print_freq=1):
#         self.model.train()
#
#         batch_time = AverageMeter()
#         data_time = AverageMeter()
#         losses_ce = AverageMeter()
#         losses_tr = AverageMeter()
#         precisions = AverageMeter()
#
#         end = time.time()
#
#         for i in range(train_iters):
#             # import ipdb
#             # ipdb.set_trace()
#             source_inputs = data_loader_source.next()
#
#             data_time.update(time.time() - end)
#
#             s_inputs, targets = self._parse_data(source_inputs)
#
#             s_features, s_cls_out,_ = self.model(s_inputs,training=True)
#             # target samples: only forward
#
#
#             # backward main #
#             loss_ce, loss_tr, prec1 = self._forward(s_features, s_cls_out[0], targets)
#             loss = loss_ce + loss_tr
#
#             losses_ce.update(loss_ce.item())
#             losses_tr.update(loss_tr.item())
#             precisions.update(prec1)
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             batch_time.update(time.time() - end)
#             end = time.time()
#
#
#             if ((i + 1) % print_freq == 0):
#                 print('Epoch: [{}][{}/{}]\t'
#                       'Time {:.3f} ({:.3f})\t'
#                       'Data {:.3f} ({:.3f})\t'
#                       'Loss_ce {:.3f} ({:.3f})\t'
#                       'Loss_tr {:.3f} ({:.3f})\t'
#                       'Prec {:.2%} ({:.2%})'
#                       .format(epoch, i + 1, train_iters,
#                               batch_time.val, batch_time.avg,
#                               data_time.val, data_time.avg,
#                               losses_ce.val, losses_ce.avg,
#                               losses_tr.val, losses_tr.avg,
#                               precisions.val, precisions.avg))
#
#     def _parse_data(self, inputs):
#         imgs, _, pids,_, _ = inputs#, pids, index
#         inputs = imgs.cuda()
#         targets = pids.cuda()
#         return inputs, targets
#
#
#     def _forward(self, s_features, s_outputs, targets):
#         loss_ce = self.criterion_ce(s_outputs, targets)
#         if isinstance(self.criterion_triple, SoftTripletLoss):
#             loss_tr = self.criterion_triple(s_features, s_features, targets)
#         elif isinstance(self.criterion_triple, TripletLoss):
#             loss_tr, _ = self.criterion_triple(s_features, targets)
#         prec, = accuracy(s_outputs.data, targets.data)
#         prec = prec[0]
#
#         return loss_ce, loss_tr, prec
class PreTrainer(object):
    def __init__(self, model, num_classes, margin=0.0):
        super(PreTrainer, self).__init__()
        self.model = model
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
        self.criterion_triple = SoftTripletLoss(margin=margin).cuda()

    def train(self, epoch, data_loader_source, data_loader_target, optimizer, train_iters=200, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_tr = AverageMeter()
        precisions = AverageMeter()

        end = time.time()

        for i in range(train_iters):
            # import ipdb
            # ipdb.set_trace()
            source_inputs = data_loader_source.next()
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            s_inputs, targets = self._parse_data(source_inputs)
            t_inputs, _ = self._parse_data(target_inputs)
            s_features, s_cls_out,_ = self.model(s_inputs,training=True)
            # target samples: only forward
            _,_,_= self.model(t_inputs,training=True)

            # backward main #
            loss_ce, loss_tr, prec1 = self._forward(s_features, s_cls_out[0], targets)
            loss = loss_ce + loss_tr

            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tr.item())
            precisions.update(prec1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if ((i ) % print_freq == 0):

                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'lr is {:.7f}\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_tr {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              optimizer.state_dict()['param_groups'][0]['lr'],
                              losses_ce.val, losses_ce.avg,
                              losses_tr.val, losses_tr.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, _, pids,_, _ = inputs#, pids, index
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets


    def _forward(self, s_features, s_outputs, targets):
        loss_ce = self.criterion_ce(s_outputs, targets)
        if isinstance(self.criterion_triple, SoftTripletLoss):
            loss_tr = self.criterion_triple(s_features, s_features, targets)
        elif isinstance(self.criterion_triple, TripletLoss):
            loss_tr, _ = self.criterion_triple(s_features, targets)
        prec, = accuracy(s_outputs.data, targets.data)
        prec = prec[0]

        return loss_ce, loss_tr, prec


class MMTTrainer(object):
    def __init__(self, model_1, model_2,contrast,
                       model_1_ema, model_2_ema, num_cluster=500, alpha=0.999):
        super(MMTTrainer, self).__init__()
        self.model_1 = model_1
        # self.model_2 = model_2
        self.num_cluster = num_cluster

        self.model_1_ema = model_1_ema
        # self.model_2_ema = model_2_ema
        self.alpha = alpha

        self.criterion_ce = CrossEntropyLabelSmooth(num_cluster).cuda()
        # self.criterion_ce_soft = SoftEntropy().cuda()
        self.criterion_tri = SoftTripletLoss(margin=0.0).cuda()
        # self.criterion_tri_soft = SoftTripletLoss(margin=None).cuda()

        self.criterion_multice = MultiSoftmaxLoss().cuda()#NCESoftmaxLoss().cuda()#

        self.contrast=contrast

    def train(self, epoch, data_loader_target, optimizer,
            contrast, index2label, ce_soft_weight=0.5, tri_soft_weight=0.5, print_freq=100, train_iters=200):

        self.model_1.train()
        # self.model_2.train()
        self.model_1_ema.train()
        # self.model_2_ema.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_ce = [AverageMeter(),AverageMeter()]
        losses_tri = [AverageMeter(),AverageMeter()]
        losses_ce_soft = AverageMeter()
        losses_tri_soft = AverageMeter()
        contra_loss = AverageMeter()
        precisions = [AverageMeter(),AverageMeter()]

        end = time.time()
        for i in range(train_iters):
            # import ipdb
            # ipdb.set_trace()

            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs_1, inputs_2, targets, index = self._parse_data(target_inputs)
            # target_ = [index2label[i.item()] for i in index]
            # import ipdb
            # ipdb.set_trace()
            # forward

            f_out_t1, p_out_t1, memory_f_t1 = self.model_1(inputs_1,training=True)
            # f_out_t2, p_out_t2 = self.model_2(inputs_2)
            # f_out_t1_ema, p_out_t1_ema, memory_f_t1_ema = self.model_1_ema(inputs_1, training=True)
            # f_out_t2_ema, p_out_t2_ema, memory_f_t2_ema = self.model_1_ema(inputs_2, training=True)
            shuffle_ids, reverse_ids = self.get_shuffle_ids(inputs_2.shape[0])
            with torch.no_grad():
                inputs_1 = inputs_1[shuffle_ids]
                inputs_2 = inputs_2[shuffle_ids]
                #feat_k, _ = model_ema(x2)
                # f_out_t2_ema, p_out_t2_ema = self.model_2_ema(inputs_2)
                f_out_t1_ema, p_out_t1_ema, memory_f_t1_ema = self.model_1_ema(inputs_1, training=True)
                f_out_t2_ema, p_out_t2_ema, memory_f_t2_ema = self.model_1_ema(inputs_2, training=True)
                memory_f_t1_ema = memory_f_t1_ema[reverse_ids]
                memory_f_t2_ema = memory_f_t2_ema[reverse_ids]
            #f_out_t2_ema, p_out_t2_ema = self.model_1_ema(inputs_2,training=True)

            #feat_q, feat_k = f_out_t1, f_out_t1_ema
            out, is_neg =self.contrast(memory_f_t1, memory_f_t1_ema, memory_f_t2_ema,  targets, index, index2label)

            contra_loss1 = self.criterion_multice(out, is_neg)

            loss_ce_1 = self.criterion_ce(p_out_t1, targets)
            # loss_ce_2 = self.criterion_ce(p_out_t2, targets)

            loss_tri_1 = self.criterion_tri(f_out_t1, f_out_t1, targets)
            # loss_tri_2 = self.criterion_tri(f_out_t2, f_out_t2, targets)

            # loss_ce_soft = self.criterion_ce_soft(p_out_t1, p_out_t2_ema) + self.criterion_ce_soft(p_out_t2, p_out_t1_ema)
            # loss_tri_soft = self.criterion_tri_soft(f_out_t1, f_out_t2_ema, targets) + \
            #                 self.criterion_tri_soft(f_out_t2, f_out_t1_ema, targets)

            # loss = (loss_ce_1 + loss_ce_2)*(1-ce_soft_weight) + \
            #          (loss_tri_1 + loss_tri_2)*(1-tri_soft_weight) + \
            #          loss_ce_soft*ce_soft_weight + loss_tri_soft*tri_soft_weight +contra_loss1
            loss = loss_ce_1 + loss_tri_1 + contra_loss1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            self._update_ema_variables(self.model_1, self.model_1_ema, self.alpha, epoch*len(data_loader_target)+i)
            # self._update_ema_variables(self.model_2, self.model_2_ema, self.alpha, epoch*len(data_loader_target)+i)

            prec_1, = accuracy(p_out_t1.data, targets.data)
            # prec_2, = accuracy(p_out_t2.data, targets.data)

            losses_ce[0].update(loss_ce_1.item())
            # losses_ce[1].update(loss_ce_2.item())
            losses_tri[0].update(loss_tri_1.item())
            # losses_tri[1].update(loss_tri_2.item())
            # losses_ce_soft.update(loss_ce_soft.item())
            # losses_tri_soft.update(loss_tri_soft.item())
            contra_loss.update(contra_loss1.item())
            precisions[0].update(prec_1[0])
            # precisions[1].update(prec_2[0])

            # print log #
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} / {:.3f}\t'
                      'Loss_tri {:.3f} / {:.3f}\t'
                      'Loss_ce_soft {:.3f}\t'
                      'Loss_tri_soft {:.3f}\t'
                      'contra_loss{:.3f}\t'
                      'Prec {:.2%} / {:.2%}\t'
                      .format(epoch, i + 1, len(data_loader_target),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce[0].avg, losses_ce[1].avg,
                              losses_tri[0].avg, losses_tri[1].avg,
                              losses_ce_soft.avg, losses_tri_soft.avg, contra_loss.avg,
                              precisions[0].avg, precisions[1].avg))

    def get_shuffle_ids(self, bsz):
        """generate shuffle ids for shufflebn"""
        forward_inds = torch.randperm(bsz).long().cuda()
        backward_inds = torch.zeros(bsz).long().cuda()
        value = torch.arange(bsz).long().cuda()
        backward_inds.index_copy_(0, forward_inds, value)
        return forward_inds, backward_inds


    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def _parse_data(self, inputs):
        imgs_1, imgs_2, pids, index = inputs
        inputs_1 = imgs_1.cuda()
        inputs_2 = imgs_2.cuda()
        targets = pids.cuda()
        index = index.cuda()
        return inputs_1, inputs_2, targets, index


class MMTTrainer_single(object):
    def __init__(self, model_1, model_1_ema, contrast, ecn, num_cluster=None,c_name=None, alpha=0.999):
        super(MMTTrainer_single, self).__init__()
        self.model_1 = model_1
        # self.model_2 = model_2
        self.num_cluster = num_cluster
        self.c_name = c_name if c_name!=None else num_cluster
        self.model_1_ema = model_1_ema
        # self.model_2_ema = model_2_ema
        self.alpha = alpha
        for i,cl in enumerate(self.num_cluster):
            exec("self.criterion_ce{}_{} = CrossEntropyLabelSmooth({}).cuda()".format(i,self.c_name[i],cl))

        self.criterion_tri = SoftTripletLoss(margin=0.0).cuda()

        self.ecn = ecn
        # self.criterion_ce_soft = SoftEntropy().cuda()
        # self.criterion_tri_soft = SoftTripletLoss(margin=None).cuda()

        # self.criterion_multice = MultiSoftmaxLoss().cuda()
        # self.criterion_multice = NCESoftmaxLoss().cuda()
        # self.criterion_multice = MultiSoftmaxLoss().cuda()
        self.contrast = contrast

        # self.cross_batch=CrossBatchMemory()

    def train(self, epoch, data_loader_target, optimizer, choice_c, ce_soft_weight=0.5,
              tri_soft_weight=0.5, print_freq=100, train_iters=200):

        self.model_1.train()
        # self.model_2.train()
        self.model_1_ema.train()
        # self.model_1_ema.eval()
        # self.model_2_ema.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_ce = [AverageMeter(),AverageMeter()]
        losses_tri = [AverageMeter(),AverageMeter()]
        losses_ce_soft = AverageMeter()
        losses_tri_soft = AverageMeter()
        contra_loss = AverageMeter()
        precisions = [AverageMeter(),AverageMeter()]

        end = time.time()
        for i in range(train_iters):
            # import ipdb
            # ipdb.set_trace()

            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            items = self._parse_data(target_inputs)

            inputs_1=items[0]
            inputs_2=items[1]
            index=items[-1]

            # target_ = [index2label[i.item()] for i in index]
            # forward

            f_out_t1, p_out_t1, memory_f_t1 = self.model_1(inputs_1, training=True)
            f_out_t2, p_out_t2, memory_f_t2 = self.model_1(inputs_2, training=True)

            with torch.no_grad():
                f_out_t1_ema, p_out_t1_ema, memory_f_t1_ema = self.model_1_ema(inputs_1, training=True)
                f_out_t2_ema, p_out_t2_ema, memory_f_t2_ema = self.model_1_ema(inputs_2, training=True)
            # loss_un = 0.1*self.ecn(memory_f_t1, index, epoch=epoch)

            #
            #contra_loss1 = self.cross_batch(memory_f_t1,items[3])#self.criterion_multice(out, is_neg)

            '''
            shuffle_ids, reverse_ids = self.get_shuffle_ids(inputs_2.shape[0])
            with torch.no_grad():
                inputs_1 = inputs_1[shuffle_ids]
                inputs_2 = inputs_2[shuffle_ids]
                #feat_k, _ = model_ema(x2)
                # f_out_t2_ema, p_out_t2_ema = self.model_2_ema(inputs_2)
                f_out_t1_ema, _, memory_f_t1_ema = self.model_1_ema(inputs_1, training=True)
                f_out_t2_ema, _, memory_f_t2_ema = self.model_1_ema(inputs_2, training=True)
                memory_f_t1_ema = memory_f_t1_ema[reverse_ids]
                memory_f_t2_ema = memory_f_t2_ema[reverse_ids]
            '''
            # f_out_t2_ema, p_out_t2_ema = self.model_1_ema(inputs_2,training=True)

            # feat_q, feat_k = f_out_t1, f_out_t1_ema

            loss_ce_1 = []#torch.tensor(0).cuda()
            for k,nc in enumerate(self.num_cluster):
                exec("loss_ce_1.append(self.criterion_ce{}_{}(p_out_t1[{}], items[{}]))".format(k, self.c_name[k],k, k+2))
            loss_ce_1 = sum(loss_ce_1)/len(self.num_cluster)
            # loss_ce_1 = self.criterion_ce500(p_out_t1[0], items[2])#+self.criterion_ce700(p_out_t1[0], items[2])


            # loss_ce_2 = self.criterion_ce(p_out_t2, targets)
            # loss_tri_1 = []  # torch.tensor(0).cuda()
            # for k, nc in enumerate(self.num_cluster):
            #     exec("loss_tri_1.append(self.criterion_tri(f_out_t1, f_out_t1, items[2+{}]))".format(k))
            # loss_tri_1 = sum(loss_tri_1) / 4.0


            loss_tri_1 = self.criterion_tri(f_out_t1, f_out_t1, items[choice_c+2])
            # loss_tri_2 = self.criterion_tri(f_out_t2, f_out_t2, targets)

            #loss_ce_soft = self.criterion_ce_soft(p_out_t1, p_out_t2_ema) + self.criterion_ce_soft(p_out_t2, p_out_t1_ema)
            #loss_tri_soft = self.criterion_tri_soft(f_out_t1, f_out_t2_ema, targets) + \
            #                self.criterion_tri_soft(f_out_t2, f_out_t1_ema, targets)

            #contra_loss1 = self.contrast(memory_f_t1, memory_f_t2, index, epoch=epoch)
            loss = loss_ce_1 + loss_tri_1 #+ contra_loss1


            # contra_loss1 = torch.tensor(0).cuda()
            # loss = loss_ce_1 + loss_tri_1
            # loss = (loss_ce_1 + loss_ce_2)*(1-ce_soft_weight) + \
            #          (loss_tri_1 + loss_tri_2)*(1-tri_soft_weight) + \
            #          loss_ce_soft*ce_soft_weight + loss_tri_soft*tri_soft_weight #+contra_loss1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self._update_ema_variables(self.model_1, self.model_1_ema, self.alpha, epoch*len(data_loader_target)+i)


            prec_1, = accuracy(p_out_t1[choice_c].data, items[choice_c+2].data)
            # prec_2, = accuracy(p_out_t2.data, targets.data)


            losses_ce[0].update(loss_ce_1.item())
            # losses_ce[1].update(loss_ce_2.item())
            losses_tri[0].update(loss_tri_1.item())
            # losses_tri[1].update(loss_tri_2.item())
            # losses_ce_soft.update(loss_ce_soft.item())
            # losses_tri_soft.update(loss_tri_soft.item())
            #contra_loss.update(contra_loss1.item())
            precisions[0].update(prec_1[0])
            # precisions[1].update(prec_2[0])

            # print log #
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 1:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} / {:.3f}\t'
                      'Loss_tri {:.3f} / {:.3f}\t'
                      'Loss_ce_soft {:.3f}\t'
                      'Loss_tri_soft {:.3f}\t'
                      'contra_loss {:.3f}\t'
                      'Prec {:.2%} / {:.2%}\t'
                      .format(epoch, i + 1, len(data_loader_target),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce[0].avg, losses_ce[1].avg,
                              losses_tri[0].avg, losses_tri[1].avg,
                              losses_ce_soft.avg, losses_tri_soft.avg, contra_loss.avg,
                              precisions[0].avg, precisions[1].avg))


    def get_shuffle_ids(self, bsz):
        """generate shuffle ids for shufflebn"""
        forward_inds = torch.randperm(bsz).long().cuda()
        backward_inds = torch.zeros(bsz).long().cuda()
        value = torch.arange(bsz).long().cuda()
        backward_inds.index_copy_(0, forward_inds, value)
        return forward_inds, backward_inds

    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def _parse_data(self, inputs):
        #imgs_1, imgs_2, pids,...,pids2, index = inputs
        inputs_1 = inputs[0].cuda()
        inputs_2 = inputs[1].cuda()
        pids=[]
        for i,pid in enumerate(inputs[3:-2]):
            pids.append(pid.cuda())
        index = inputs[-1].cuda()
        return [inputs_1,inputs_2]+ pids+[index]


class DbscanBaseTrainer(object):
    def __init__(self, model_1, model_1_ema, contrast, ecn, num_cluster=None, c_name=None, alpha=0.999,fc_len=3000):
        super(DbscanBaseTrainer, self).__init__()
        self.model_1 = model_1
        # self.model_2 = model_2
        self.num_cluster = num_cluster
        self.c_name = [fc_len for _ in range(len(num_cluster))]
        self.model_1_ema = model_1_ema
        # self.model_2_ema = model_2_ema
        self.alpha = alpha
        for i,cl in enumerate(self.num_cluster):
            exec("self.criterion_ce{}_{} = CrossEntropyLabelSmooth({}).cuda()".format(i,self.c_name[i],cl))

        self.criterion_tri = SoftTripletLoss(margin=0.0).cuda()


        self.ecn = ecn
        # self.criterion_ce_soft = SoftEntropy().cuda()
        # self.criterion_tri_soft = SoftTripletLoss(margin=None).cuda()

        # self.criterion_multice = MultiSoftmaxLoss().cuda()
        # self.criterion_multice = NCESoftmaxLoss().cuda()
        # self.criterion_multice = MultiSoftmaxLoss().cuda()
        self.contrast = contrast

        # self.cross_batch=CrossBatchMemory()

    def train(self, epoch, data_loader_target, optimizer, choice_c, ce_soft_weight=0.5,
              tri_soft_weight=0.5, print_freq=100, train_iters=200):

        self.model_1.train()
        # self.model_2.train()
        self.model_1_ema.train()
        # self.model_1_ema.eval()
        # self.model_2_ema.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_ce = [AverageMeter(),AverageMeter()]
        losses_tri = [AverageMeter(),AverageMeter()]
        losses_ce_soft = AverageMeter()
        losses_tri_soft = AverageMeter()
        contra_loss = AverageMeter()
        precisions = [AverageMeter(),AverageMeter()]

        end = time.time()
        for i in range(train_iters):
            # import ipdb
            # ipdb.set_trace()

            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            items = self._parse_data(target_inputs)

            inputs_1=items[0]
            inputs_2=items[1]
            index=items[-1]

            # target_ = [index2label[i.item()] for i in index]
            # forward

            f_out_t1, p_out_t1, memory_f_t1 = self.model_1(inputs_1, training=True)
            # f_out_t2, p_out_t2, memory_f_t2 = self.model_1(inputs_2, training=True)

            with torch.no_grad():
                f_out_t1_ema, p_out_t1_ema, memory_f_t1_ema = self.model_1_ema(inputs_1, training=True)
                # f_out_t2_ema, p_out_t2_ema, memory_f_t2_ema = self.model_1_ema(inputs_2, training=True)
            # loss_un = 0.1*self.ecn(memory_f_t1, index, epoch=epoch)


            #contra_loss1 = self.cross_batch(memory_f_t1,items[3])#self.criterion_multice(out, is_neg)

            '''
            shuffle_ids, reverse_ids = self.get_shuffle_ids(inputs_2.shape[0])
            with torch.no_grad():
                inputs_1 = inputs_1[shuffle_ids]
                inputs_2 = inputs_2[shuffle_ids]
                #feat_k, _ = model_ema(x2)
                # f_out_t2_ema, p_out_t2_ema = self.model_2_ema(inputs_2)
                f_out_t1_ema, _, memory_f_t1_ema = self.model_1_ema(inputs_1, training=True)
                f_out_t2_ema, _, memory_f_t2_ema = self.model_1_ema(inputs_2, training=True)
                memory_f_t1_ema = memory_f_t1_ema[reverse_ids]
                memory_f_t2_ema = memory_f_t2_ema[reverse_ids]
            '''
            # f_out_t2_ema, p_out_t2_ema = self.model_1_ema(inputs_2,training=True)

            # feat_q, feat_k = f_out_t1, f_out_t1_ema

            loss_ce_1 = []#torch.tensor(0).cuda()
            for k,nc in enumerate(self.num_cluster):
                exec("loss_ce_1.append(self.criterion_ce{}_{}(p_out_t1[{}], items[{}]))".format(k, self.c_name[k],k, k+2))
            loss_ce_1 = sum(loss_ce_1)/len(self.num_cluster)
            # loss_ce_1 = self.criterion_ce500(p_out_t1[0], items[2])#+self.criterion_ce700(p_out_t1[0], items[2])


            # loss_ce_2 = self.criterion_ce(p_out_t2, targets)
            # loss_tri_1 = []  # torch.tensor(0).cuda()
            # for k, nc in enumerate(self.num_cluster):
            #     exec("loss_tri_1.append(self.criterion_tri(f_out_t1, f_out_t1, items[2+{}]))".format(k))
            # loss_tri_1 = sum(loss_tri_1) / 4.0


            loss_tri_1 = self.criterion_tri(f_out_t1, f_out_t1, items[choice_c+2])
            # loss_tri_2 = self.criterion_tri(f_out_t2, f_out_t2, targets)

            #loss_ce_soft = self.criterion_ce_soft(p_out_t1, p_out_t2_ema) + self.criterion_ce_soft(p_out_t2, p_out_t1_ema)
            #loss_tri_soft = self.criterion_tri_soft(f_out_t1, f_out_t2_ema, targets) + \
            #                self.criterion_tri_soft(f_out_t2, f_out_t1_ema, targets)

            # if None:#epoch %10!=1:
            #     contra_loss1 = self.contrast(memory_f_t1, memory_f_t2, index, epoch=epoch)#TODO
            # else:
            #     contra_loss1= torch.tensor(0).cuda()
            contra_loss1 = torch.tensor(0).cuda()#self.contrast(memory_f_t1, memory_f_t1, index, epoch=epoch)
            loss = loss_ce_1 + loss_tri_1 + contra_loss1


            # contra_loss1 = torch.tensor(0).cuda()
            # loss = loss_ce_1 + loss_tri_1
            # loss = (loss_ce_1 + loss_ce_2)*(1-ce_soft_weight) + \
            #          (loss_tri_1 + loss_tri_2)*(1-tri_soft_weight) + \
            #          loss_ce_soft*ce_soft_weight + loss_tri_soft*tri_soft_weight #+contra_loss1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self._update_ema_variables(self.model_1, self.model_1_ema, self.alpha, epoch*len(data_loader_target)+i)


            prec_1, = accuracy(p_out_t1[choice_c].data, items[choice_c+2].data)
            # prec_2, = accuracy(p_out_t2.data, targets.data)


            losses_ce[0].update(loss_ce_1.item())
            # losses_ce[1].update(loss_ce_2.item())
            losses_tri[0].update(loss_tri_1.item())
            # losses_tri[1].update(loss_tri_2.item())
            # losses_ce_soft.update(loss_ce_soft.item())
            # losses_tri_soft.update(loss_tri_soft.item())
            contra_loss.update(contra_loss1.item())
            precisions[0].update(prec_1[0])
            # precisions[1].update(prec_2[0])

            # print log #
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 1:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'LR {:.7f} \t'
                      'Loss_ce {:.3f} / {:.3f}\t'
                      'Loss_tri {:.3f} / {:.3f}\t'
                      'Loss_ce_soft {:.3f}\t'
                      'Loss_tri_soft {:.3f}\t'
                      'contra_loss {:.3f}\t'
                      'Prec {:.2%} / {:.2%}\t'
                      .format(epoch, i , len(data_loader_target),
                              batch_time.val, batch_time.avg,
                              optimizer.state_dict()['param_groups'][0]['lr'],
                              losses_ce[0].avg, losses_ce[1].avg,
                              losses_tri[0].avg, losses_tri[1].avg,
                              losses_ce_soft.avg, losses_tri_soft.avg, contra_loss.avg,
                              precisions[0].avg, precisions[1].avg))


    def get_shuffle_ids(self, bsz):
        """generate shuffle ids for shufflebn"""
        forward_inds = torch.randperm(bsz).long().cuda()
        backward_inds = torch.zeros(bsz).long().cuda()
        value = torch.arange(bsz).long().cuda()
        backward_inds.index_copy_(0, forward_inds, value)
        return forward_inds, backward_inds

    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def _parse_data(self, inputs):
        #imgs_1, imgs_2, pids,...,pids2, index = inputs
        inputs_1 = inputs[0].cuda()
        inputs_2 = inputs[1].cuda()
        pids=[]
        for i,pid in enumerate(inputs[3:-2]):
            pids.append(pid.cuda())
        index = inputs[-1].cuda()
        return [inputs_1,inputs_2]+ pids+[index]

class MMTTrainer_double(object):
    def __init__(self, model_1, model_1_ema, model_2,model_2_ema, contrast,ecn, num_cluster=None, alpha=0.999):
        super(MMTTrainer_double, self).__init__()
        self.model_1 = model_1
        self.model_2 = model_2
        self.num_cluster = num_cluster

        self.model_1_ema = model_1_ema
        self.model_2_ema = model_2_ema
        self.alpha = alpha
        for i in self.num_cluster:
            exec("self.criterion_ce{} = CrossEntropyLabelSmooth({}).cuda()".format(i,i))

        self.criterion_tri = SoftTripletLoss(margin=0.0).cuda()
        self.ecn=ecn
        self.criterion_ce_soft = SoftEntropy().cuda()
        self.criterion_tri_soft = SoftTripletLoss(margin=None).cuda()

        # self.criterion_multice = MultiSoftmaxLoss().cuda()
        # self.criterion_multice = NCESoftmaxLoss().cuda()
        # self.criterion_multice = MultiSoftmaxLoss().cuda()
        # self.contrast = contrast

    def train(self, epoch, data_loader_target, optimizer, index2label, ce_soft_weight=0.5,
              tri_soft_weight=0.5, print_freq=100, train_iters=200):

        self.model_1.train()
        self.model_2.train()
        self.model_1_ema.train()
        self.model_2_ema.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_ce = [AverageMeter(),AverageMeter()]
        losses_tri = [AverageMeter(),AverageMeter()]
        losses_ce_soft = AverageMeter()
        losses_tri_soft = AverageMeter()
        contra_loss = AverageMeter()
        precisions = [AverageMeter(),AverageMeter()]

        end = time.time()
        for i in range(train_iters):
            # import ipdb
            # ipdb.set_trace()

            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            items = self._parse_data(target_inputs)
            inputs_1=items[0]
            inputs_2=items[1]
            index=items[-1]

            # target_ = [index2label[i.item()] for i in index]
            # forward

            f_out_t1, p_out_t1, memory_f_t1 = self.model_1(inputs_1, training=True)
            f_out_t2, p_out_t2, memory_f_t2 = self.model_2(inputs_2, training=True)

            with torch.no_grad():
                f_out_t1_ema, p_out_t1_ema, memory_f_t1_ema = self.model_1_ema(inputs_1, training=True)
                f_out_t2_ema, p_out_t2_ema, memory_f_t2_ema = self.model_2_ema(inputs_2, training=True)
            #loss_un = 0.1*self.ecn(memory_f_t1, index, epoch=epoch)

            '''
            shuffle_ids, reverse_ids = self.get_shuffle_ids(inputs_2.shape[0])
            with torch.no_grad():
                inputs_1 = inputs_1[shuffle_ids]
                inputs_2 = inputs_2[shuffle_ids]
                #feat_k, _ = model_ema(x2)
                # f_out_t2_ema, p_out_t2_ema = self.model_2_ema(inputs_2)
                f_out_t1_ema, _, memory_f_t1_ema = self.model_1_ema(inputs_1, training=True)
                f_out_t2_ema, _, memory_f_t2_ema = self.model_1_ema(inputs_2, training=True)
                memory_f_t1_ema = memory_f_t1_ema[reverse_ids]
                memory_f_t2_ema = memory_f_t2_ema[reverse_ids]
            '''
            # f_out_t2_ema, p_out_t2_ema = self.model_1_ema(inputs_2,training=True)

            # feat_q, feat_k = f_out_t1, f_out_t1_ema
            '''
            out, is_neg =self.contrast(memory_f_t1, memory_f_t1_ema, memory_f_t2_ema,
                                       targets, index, index2label)

            contra_loss1 = self.criterion_multice(out, is_neg)'''
            loss_ce_1=[]#torch.tensor(0).cuda()
            for k,nc in enumerate(self.num_cluster):
                exec("loss_ce_1.append(self.criterion_ce{}(p_out_t1[{}], items[{}]))".format( nc,k, k+2))
            loss_ce_1 = sum(loss_ce_1) / 5.0
            # loss_ce_1 = self.criterion_ce500(p_out_t1[0], items[2])#+self.criterion_ce700(p_out_t1[0], items[2])
            loss_ce_2 = []  # torch.tensor(0).cuda()
            for k, nc in enumerate(self.num_cluster):
                exec("loss_ce_2.append(self.criterion_ce{}(p_out_t2[{}], items[{}]))".format(nc, k, k + 2))
            loss_ce_2 = sum(loss_ce_2) / 5.0
            # loss_ce_2 = self.criterion_ce(p_out_t2, targets)

            targets = items[3]
            loss_tri_1 = self.criterion_tri(f_out_t1, f_out_t1, targets)
            loss_tri_2 = self.criterion_tri(f_out_t2, f_out_t2, targets)

            loss_ce_soft=[]
            for k, nc in enumerate(self.num_cluster):
                exec("loss_ce_soft.append(self.criterion_ce_soft(p_out_t1[{}], p_out_t2_ema[{}]) + self.criterion_ce_soft(p_out_t2[{}], p_out_t1_ema[{}]))".format(k,k,k,k))
            loss_ce_soft = sum(loss_ce_soft) / 10.0

            loss_tri_soft = self.criterion_tri_soft(f_out_t1, f_out_t2_ema, targets) + \
                            self.criterion_tri_soft(f_out_t2, f_out_t1_ema, targets)

            # loss = loss_ce_1 + loss_tri_1 #+ loss_un#+ contra_loss1

            loss = (loss_ce_1 + loss_ce_2)*(1-ce_soft_weight) + \
                   (loss_tri_1 + loss_tri_2)*(1-tri_soft_weight) + \
                   loss_ce_soft*ce_soft_weight + loss_tri_soft*tri_soft_weight #+contra_loss1


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self._update_ema_variables(self.model_1, self.model_1_ema, self.alpha, epoch*len(data_loader_target)+i)
            self._update_ema_variables(self.model_2, self.model_2_ema, self.alpha, epoch*len(data_loader_target)+i)

            prec_1, = accuracy(p_out_t1[1].data, items[3].data)
            # prec_2, = accuracy(p_out_t2.data, targets.data)

            losses_ce[0].update(loss_ce_1.item())
            losses_ce[1].update(loss_ce_2.item())
            losses_tri[0].update(loss_tri_1.item())
            losses_tri[1].update(loss_tri_2.item())
            losses_ce_soft.update(loss_ce_soft.item())
            losses_tri_soft.update(loss_tri_soft.item())
            # contra_loss.update(contra_loss1.item())
            precisions[0].update(prec_1[0])
            # precisions[1].update(prec_2[0])

            # print log #
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 1:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} / {:.3f}\t'
                      'Loss_tri {:.3f} / {:.3f}\t'
                      'Loss_ce_soft {:.3f}\t'
                      'Loss_tri_soft {:.3f}\t'
                      'contra_loss{:.3f}\t'
                      'Prec {:.2%} / {:.2%}\t'
                      .format(epoch, i + 1, len(data_loader_target),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce[0].avg, losses_ce[1].avg,
                              losses_tri[0].avg, losses_tri[1].avg,
                              losses_ce_soft.avg, losses_tri_soft.avg, contra_loss.avg,
                              precisions[0].avg, precisions[1].avg))


    def get_shuffle_ids(self, bsz):
        """generate shuffle ids for shufflebn"""
        forward_inds = torch.randperm(bsz).long().cuda()
        backward_inds = torch.zeros(bsz).long().cuda()
        value = torch.arange(bsz).long().cuda()
        backward_inds.index_copy_(0, forward_inds, value)
        return forward_inds, backward_inds

    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def _parse_data(self, inputs):
        #imgs_1, imgs_2, pids,...,pids2, index = inputs
        inputs_1 = inputs[0].cuda()
        inputs_2 = inputs[1].cuda()
        pids=[]
        for i,pid in enumerate(inputs[3:-2]):
            pids.append(pid.cuda())
        index = inputs[-1].cuda()
        return [inputs_1,inputs_2]+ pids+[index]


