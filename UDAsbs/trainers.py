from __future__ import print_function, absolute_import
import time

import torch
import torch.nn as nn
from torch.nn import functional as F

from .evaluation_metrics import accuracy
from .loss import TripletLoss, CrossEntropyLabelSmooth, SoftTripletLoss, SoftTripletLoss_uncer
from .utils.meters import AverageMeter


class PreTrainer_multi(object):
    def __init__(self, model, num_classes, margin=0.0):
        super(PreTrainer_multi, self).__init__()
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
        losses_ce_3 = AverageMeter()
        losses_tr_3 = AverageMeter()
        precisions_3 = AverageMeter()

        end = time.time()

        for i in range(train_iters):
            # import ipdb
            # ipdb.set_trace()
            source_inputs = data_loader_source.next()
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            s_inputs, targets = self._parse_data(source_inputs)
            t_inputs, _ = self._parse_data(target_inputs)
            s_features, s_cls_out,_,_,s_cls_out_3,s_features_3 = self.model(s_inputs,training=True)
            # target samples: only forward
            self.model(t_inputs,training=True)

            # backward main #
            loss_ce, loss_tr, prec1 = self._forward(s_features, s_cls_out[0], targets)
            loss_ce_3, loss_tr_3, prec1_3 = self._forward(s_features_3, s_cls_out_3[0], targets)
            loss = loss_ce + loss_tr + loss_ce_3 + loss_tr_3

            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tr.item())
            precisions.update(prec1)
            losses_ce_3.update(loss_ce_3.item())
            losses_tr_3.update(loss_tr_3.item())
            precisions_3.update(prec1_3)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()


            if ((i + 1) % print_freq == 0):
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_tr {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})'
                      'Loss_ce_3 {:.3f} ({:.3f})\t'
                      'Loss_tr_3 {:.3f} ({:.3f})\t'
                      'Prec_3 {:.2%} ({:.2%})'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tr.val, losses_tr.avg,
                              precisions.val, precisions.avg,
                              losses_ce_3.val, losses_ce_3.avg,
                              losses_tr_3.val, losses_tr_3.avg,
                              precisions_3.val, precisions_3.avg))

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


            if ((i + 1) % print_freq == 0):
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_tr {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
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

class DbscanBaseTrainer_multi(object):
    def __init__(self, model_1, model_1_ema, contrast, contrast_center, contrast_center_sour, num_cluster=None,
                 c_name=None, alpha=0.999,source_classes=702,uncer_mode=0):

        super(DbscanBaseTrainer_multi, self).__init__()
        self.model_1 = model_1

        self.num_cluster = num_cluster
        self.c_name = num_cluster#[fc_len for _ in range(len(num_cluster))]
        self.model_1_ema = model_1_ema
        self.uncer_mode=uncer_mode
        self.alpha = alpha
        self.criterion_ce = CrossEntropyLabelSmooth(self.num_cluster[0],False).cuda()

        # self.criterion_tri = SoftTripletLoss(margin=0.0).cuda()
        self.criterion_tri_uncer = SoftTripletLoss_uncer(margin=None,uncer_mode=self.uncer_mode).cuda()
        self.source_classes = source_classes

        self.contrast = contrast
        # self.kl = nn.KLDivLoss()
        self.sm = torch.nn.Softmax(dim=1)
        self.log_sm = torch.nn.LogSoftmax(dim=1)
        # self.cross_batch=CrossBatchMemory()
        self.kl_distance = nn.KLDivLoss(reduction='none')

    def train(self, epoch, data_loader_target, data_loader_source, optimizer, choice_c, lambda_tri=1.0
              , lambda_ct=1.0, lambda_reg=0.06, print_freq=100, train_iters=200):


        self.model_1.train()
        self.model_1_ema.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_ce = [AverageMeter(), AverageMeter()]
        losses_tri = [AverageMeter(), AverageMeter()]
        loss_kldiv = AverageMeter()
        loss_s = AverageMeter()
        losses_tri_unc = AverageMeter()
        contra_loss = AverageMeter()
        precisions = [AverageMeter(), AverageMeter()]

        end = time.time()
        for i in range(train_iters):


            target_inputs = data_loader_target.next()
            source_inputs = data_loader_source.next()

            data_time.update(time.time() - end)

            # process inputs
            items = self._parse_data(target_inputs)
            items_source = self._parse_data(source_inputs)

            inputs_1_t, inputs_2_t, index_t = items[0], items[1], items[-1]
            inputs_1_s, inputs_2_s, index_s = items_source[0], items_source[1], items_source[-1]

            inputs = self.range_spbn(inputs_1_s, inputs_1_t)

            f_out, p_out, memory_f, _, p_out_3, f_out_3 = self.model_1(inputs, training=True)

            f_out_s1, f_out_t1 = self.derange_spbn(f_out)
            _, p_out_t1 = self.derange_spbn(p_out[0])
            _, memory_f_t1 = self.derange_spbn(memory_f)
            # _, ml_x_t1 = self.derange_spbn(ml_x)
            _, p_out_3_t1 = self.derange_spbn(p_out_3[0])
            _, f_out_3_t1 = self.derange_spbn(f_out_3)
            # _, sim_out_t1 =self.derange_spbn(sim_out)
            with torch.no_grad():
                f_out_ema, p_out_ema, memory_f_ema, _, p_out_3_ema, f_out_3_ema \
                    = self.model_1_ema(inputs, training=True)
            f_out_s1_ema, f_out_t1_ema = self.derange_spbn(f_out_ema)
            _, p_out_t1_ema = self.derange_spbn(p_out_ema[0])
            _, memory_f_t1_ema = self.derange_spbn(memory_f_ema)
            # _, ml_x_t1_ema = self.derange_spbn(ml_x_ema)
            _, p_out_3_t1_ema = self.derange_spbn(p_out_3_ema[0])
            _, f_out_3_t1_ema = self.derange_spbn(f_out_3_ema)
            # _, sim_out_t1_ema = self.derange_spbn(sim_out_ema)

            with torch.no_grad():
                queue = self.contrast.memory[:self.contrast.sour_numclass, :].clone()
                ml_sour = torch.matmul(f_out_t1, queue.transpose(1, 0).detach())
                ml_sour_ema = torch.matmul(f_out_t1_ema, queue.transpose(1, 0).detach())

            ########## [memory center]-level uncertainty
            loss_ce_1, loss_reg, exp_variance = self.update_variance(items[2], p_out_t1, p_out_3_t1, p_out_t1_ema,
                                                                     p_out_3_t1_ema, ml_sour,ml_sour_ema,f_out_t1,f_out_t1_ema)
            # loss_ce_1_3, exp_variance_3 = self.update_variance(items[2], p_out_3_t1, p_out_t1, p_out_t1_ema,
            #                                                    ml_sour, ml_sour_ema)
            loss_ce_1 = loss_ce_1#(loss_ce_1+loss_ce_1_3)/2.0

            # exp_variance=torch.tensor(0)
            loss_kl = exp_variance.mean()

            contra_loss_instance, contra_loss_center, _, _ = \
                self.contrast(memory_f_t1, f_out_s1, index_t, f_out_t1, f_out_t1_ema,  items_source[2], uncer=exp_variance, epoch=epoch)

            ########## feature-level uncertainty
            # loss_ce_1, exp_variance = self.update_variance_self(items[2], p_out_t1, f_out_t1, f_out_t1_ema )

            ########## vallia ce loss
            loss_ce_1_norm = torch.tensor(0)#(self.criterion_ce(p_out_t1, items[2]) +self.criterion_ce(p_out_3_t1, items[2])) / 2.0

            ########## uncertainty hard triplet loss
            loss_tri_unc = self.criterion_tri_uncer(f_out_t1, f_out_t1_ema, items[2], exp_variance)

            if epoch % 6 != 0:
                loss = loss_ce_1 + lambda_tri*loss_tri_unc + lambda_reg*loss_reg + lambda_ct*contra_loss_instance + contra_loss_center
            else:
                loss = loss_ce_1 + lambda_tri*loss_tri_unc + lambda_reg*loss_reg + contra_loss_center

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self._update_ema_variables(self.model_1, self.model_1_ema, self.alpha, epoch * len(data_loader_target) + i)

            prec_1, = accuracy(p_out_t1.data, items[choice_c + 2].data)

            losses_ce[0].update(loss_ce_1.item())
            losses_ce[1].update(loss_ce_1_norm.item())
            # losses_tri[0].update(loss_tri_1.item())
            loss_s.update(contra_loss_center.item())
            loss_kldiv.update(loss_kl.item())
            losses_tri_unc.update(loss_tri_unc.item())
            contra_loss.update(contra_loss_instance.item())
            precisions[0].update(prec_1[0])

            # print log #
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 1:
                # if precisions[0].avg<=0.15:
                #     import ipdb
                #     ipdb.set_trace()
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce  {:.3f} / {:.3f}\t'
                      'loss_kldiv {:.3f}\t'
                      'Loss_tri {:.3f} / losses_tri_unc {:.3f} \t'
                      'contra_loss_center {:.3f}\t'
                      'contra_loss {:.3f}\t'
                      'Prec {:.2%} / {:.2%}\t'
                      .format(epoch, i, len(data_loader_target),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce[0].avg, losses_ce[1].avg, loss_kldiv.avg,
                              losses_tri[0].avg, losses_tri_unc.avg, loss_s.avg, contra_loss.avg,
                              precisions[0].avg, precisions[1].avg))

    def update_variance(self, labels, pred1, pred2, pred_ema, pred2_ema, ml_sour, ml_sour_ema,f_out_t1,f_out_t1_ema):
                       #(items[2], p_out_t1, p_out_3_t1, p_out_t1_ema,p_out_3_t1_ema, ml_sour, ml_sour_ema, f_out_t1, f_out_t1_ema)
        loss_4layer = self.criterion_ce(pred1, labels)
        loss_3layer = self.criterion_ce(pred2, labels)

        only_sour=False
        if only_sour:
            variance = torch.sum(self.kl_distance(self.log_sm(ml_sour), self.sm(ml_sour_ema.detach())), dim=1)
        else:
            variance = torch.sum(self.kl_distance(self.log_sm(torch.cat((pred1, ml_sour), 1)),
                                                  self.sm(torch.cat((pred2_ema, ml_sour_ema), 1).detach())), dim=1)
            # variance = torch.sum(self.kl_distance(self.log_sm(pred1),
            #                                        self.sm(pred2_ema.detach())), dim=1)
            # variance = (torch.sum(self.kl_distance(self.log_sm(torch.cat((pred1, ml_sour), 1)),
            #                                       self.sm(torch.cat((pred2_ema, ml_sour_ema), 1).detach())), dim=1) +
            #             torch.sum(self.kl_distance(self.log_sm(torch.cat((pred1, ml_sour), 1)),
            #                                       self.sm(torch.cat((pred_ema, ml_sour_ema), 1).detach())), dim=1))/2.0

        exp_variance = torch.exp(-variance)
        loss = torch.mean(loss_4layer * exp_variance) + torch.mean(loss_3layer* exp_variance)
        loss_reg=torch.mean(variance)

        return loss,loss_reg,exp_variance


    def softmax_kl_loss(self, input_logits, target_logits):
        """Takes softmax on both sides and returns KL divergence

        Note:
        - Returns the sum over all examples. Divide by the batch size afterwards
          if you want the mean.
        - Sends gradients to inputs but not the targets.
        """

        assert input_logits.size() == target_logits.size()
        input_log_softmax = F.log_softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits / 0.2, dim=1)
        return F.kl_div(input_log_softmax, target_softmax, size_average=False)

    def range_spbn(self, inputs_1_s, inputs_1_t):
        # arrange batch for domain-specific BN
        device_num = torch.cuda.device_count()
        B, C, H, W = inputs_1_s.size()

        def reshape(inputs):
            return inputs.view(device_num, -1, C, H, W)

        inputs_1_s, inputs_1_t = reshape(inputs_1_s), reshape(inputs_1_t)
        inputs = torch.cat((inputs_1_s, inputs_1_t), 1).view(-1, C, H, W)
        return inputs

    def derange_spbn(self, f_out):
        device_num = torch.cuda.device_count()
        # de-arrange batch
        f_out = f_out.view(device_num, -1, f_out.size(-1))
        f_out_s, f_out_t = f_out.split(f_out.size(1) // 2, dim=1)
        f_out_s, f_out_t = f_out_s.contiguous().view(-1, f_out.size(-1)), f_out_t.contiguous().view(-1, f_out.size(-1))
        return f_out_s, f_out_t

    def get_shuffle_ids(self, bsz):
        """generate shuffle ids for shufflebn"""
        forward_inds = torch.randperm(bsz).long().cuda()
        backward_inds = torch.zeros(bsz).long().cuda()
        value = torch.arange(bsz).long().cuda()
        backward_inds.index_copy_(0, forward_inds, value)
        return forward_inds, backward_inds

    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        # for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        #    ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

        for (ema_name, ema_param), (model_name, param) in zip(ema_model.named_parameters(), model.named_parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def _parse_data(self, inputs):
        # imgs_1, imgs_2, fname, pids,...,pids2, index = inputs

        inputs_1 = inputs[0].cuda()
        inputs_2 = inputs[1].cuda()
        pids = []
        for i, pid in enumerate(inputs[3:-2]):
            pids.append(pid.cuda())
        index = inputs[-1].cuda()
        return [inputs_1, inputs_2] + pids + [index]

class DbscanBaseTrainer(object):
    def __init__(self, model_1, model_1_ema, contrast, contrast_center, contrast_center_sour, num_cluster=None,
                 c_name=None, alpha=0.999, source_classes=702, uncer_mode=None):
        super(DbscanBaseTrainer, self).__init__()
        self.model_1 = model_1

        self.num_cluster = num_cluster
        self.c_name = num_cluster#[fc_len for _ in range(len(num_cluster))]
        self.model_1_ema = model_1_ema

        self.alpha = alpha
        self.criterion_ce = CrossEntropyLabelSmooth(self.num_cluster[0],False).cuda()

        self.criterion_tri = SoftTripletLoss(margin=0.0).cuda()

        self.source_classes = 751

        self.contrast = contrast

    def train(self, epoch, data_loader_target, data_loader_source, optimizer, choice_c, lambda_tri=1.0
              , lambda_ct=1.0, lambda_reg=0.06, print_freq=100, train_iters=200, uncertainty_d=None):

        self.model_1.train()
        # self.model_2.train()
        self.model_1_ema.train()
        # self.model_1_ema.eval()
        # self.model_2_ema.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_ce = [AverageMeter(), AverageMeter()]
        losses_tri = [AverageMeter(), AverageMeter()]
        loss_kldiv = AverageMeter()
        loss_s = AverageMeter()
        losses_tri_unc = AverageMeter()
        contra_loss = AverageMeter()
        precisions = [AverageMeter(), AverageMeter()]

        end = time.time()
        for i in range(train_iters):
            # import ipdb
            # ipdb.set_trace()

            target_inputs = data_loader_target.next()
            source_inputs = data_loader_source.next()

            data_time.update(time.time() - end)

            # process inputs
            items = self._parse_data(target_inputs)
            items_source = self._parse_data(source_inputs)

            inputs_1_t, inputs_2_t, index_t = items[0], items[1], items[-1]
            inputs_1_s, inputs_2_s, index_s = items_source[0], items_source[1], items_source[-1]

            inputs = self.range_spbn(inputs_1_s, inputs_1_t)

            f_out, p_out, memory_f = self.model_1(inputs, training=True)

            f_out_s1, f_out_t1 = self.derange_spbn(f_out)
            _, p_out_t1 = self.derange_spbn(p_out[0])
            _, memory_f_t1 = self.derange_spbn(memory_f)


            with torch.no_grad():
                f_out_ema, p_out_ema, memory_f_ema = self.model_1_ema(inputs, training=True)
            f_out_s1_ema, f_out_t1_ema = self.derange_spbn(f_out_ema)
            _, p_out_t1_ema = self.derange_spbn(p_out_ema[0])
            _, memory_f_t1_ema = self.derange_spbn(memory_f_ema)


            loss_tri_1 = self.criterion_tri(f_out_t1, f_out_t1, items[choice_c + 2])

            loss_ce_1=self.criterion_ce(p_out_t1, items[2])

            contra_loss_instance, contra_loss_center, ml_sour, ml_sour_ema = torch.tensor(0),torch.tensor(0),torch.tensor(0),torch.tensor(0)
                # self.contrast(memory_f_t1, f_out_s1, index_t, f_out_t1, f_out_t1_ema, items_source[2], epoch=epoch)

            loss_kl = loss_tri_unc= torch.tensor(0)#self.softmax_kl_loss(ml_x_t1, ml_sour)


            if epoch % 6 != 0:
                loss = loss_ce_1 + loss_tri_1 #+ contra_loss_center + contra_loss_instance
            else:
                loss = loss_ce_1 + loss_tri_1 #+ contra_loss_center

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self._update_ema_variables(self.model_1, self.model_1_ema, self.alpha, epoch * len(data_loader_target) + i)

            prec_1, = accuracy(p_out_t1.data, items[choice_c + 2].data)
            # prec_2, = accuracy(p_out_t2.data, targets.data)

            losses_ce[0].update(loss_ce_1.item())
            # losses_ce[1].update(loss_ce_2.item())
            losses_tri[0].update(loss_tri_1.item())
            loss_s.update(contra_loss_center.item())
            loss_kldiv.update(loss_kl.item())
            losses_tri_unc.update(loss_tri_unc.item())
            contra_loss.update(contra_loss_instance.item())
            precisions[0].update(prec_1[0])
            # precisions[1].update(prec_2[0])

            # print log #
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 1:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce  {:.3f} / loss_kldiv {:.3f}\t'
                      'Loss_tri {:.3f} / Loss_tri_soft {:.3f} \t'
                      'contra_loss_center {:.3f}\t'
                      'contra_loss {:.3f}\t'
                      'Prec {:.2%} / {:.2%}\t'
                      .format(epoch, i, len(data_loader_target),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce[0].avg, loss_kldiv.avg,
                              losses_tri[0].avg,losses_tri_unc.avg, loss_s.avg, contra_loss.avg,
                              precisions[0].avg, precisions[1].avg))



    def range_spbn(self, inputs_1_s, inputs_1_t):
        # arrange batch for domain-specific BN
        device_num = torch.cuda.device_count()
        B, C, H, W = inputs_1_s.size()

        def reshape(inputs):
            return inputs.view(device_num, -1, C, H, W)

        inputs_1_s, inputs_1_t = reshape(inputs_1_s), reshape(inputs_1_t)
        inputs = torch.cat((inputs_1_s, inputs_1_t), 1).view(-1, C, H, W)
        return inputs

    def derange_spbn(self, f_out):
        device_num = torch.cuda.device_count()
        # de-arrange batch
        f_out = f_out.view(device_num, -1, f_out.size(-1))
        f_out_s, f_out_t = f_out.split(f_out.size(1) // 2, dim=1)
        f_out_s, f_out_t = f_out_s.contiguous().view(-1, f_out.size(-1)), f_out_t.contiguous().view(-1, f_out.size(-1))
        return f_out_s, f_out_t


    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        # for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        #    ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

        for (ema_name, ema_param), (model_name, param) in zip(ema_model.named_parameters(), model.named_parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def _parse_data(self, inputs):
        # imgs_1, imgs_2, pids,...,pids2, index = inputs
        inputs_1 = inputs[0].cuda()
        inputs_2 = inputs[1].cuda()
        pids = []
        for i, pid in enumerate(inputs[3:-2]):
            pids.append(pid.cuda())
        index = inputs[-1].cuda()
        return [inputs_1, inputs_2] + pids + [index]


