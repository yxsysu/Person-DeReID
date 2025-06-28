import torch
from torch import nn
import math
from torch.nn import functional as F
import numpy as np


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


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    # dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def cosine_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    x_norm = torch.pow(x, 2).sum(1, keepdim=True).sqrt().expand(m, n)
    y_norm = torch.pow(y, 2).sum(1, keepdim=True).sqrt().expand(n, m).t()
    xy_intersection = torch.mm(x, y.t())
    dist = xy_intersection/(x_norm * y_norm)
    dist = (1. - dist) / 2
    return dist


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # print(dist_mat[is_pos].shape)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletLoss(object):
    """
    Triplet loss using HARDER example mining,
    modified based on original triplet loss using hard example mining
    """

    def __init__(self, margin=None, hard_factor=0.0,
                 aug_weight=0.0):
        self.margin = margin
        self.hard_factor = hard_factor
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

        self.aug_weight = aug_weight

    def __call__(self, global_feat, labels,
                 aug_neg=None, reverse=False,
                 normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
            if aug_neg is not None:
                aug_neg = normalize(aug_neg, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)

        dist_ap *= (1.0 + self.hard_factor)
        dist_an *= (1.0 - self.hard_factor)

        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if reverse:
            y = -y

        if aug_neg is not None:
            dist_mat = euclidean_dist(global_feat, aug_neg)
            aug_dist_an, _ = dist_mat.min(dim=1)
            aug_dist_an *= (1.0 - self.hard_factor)

        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
            if aug_neg is not None:
                loss += self.aug_weight * self.ranking_loss(aug_dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
            if aug_neg is not None:
                loss += self.aug_weight * self.ranking_loss(aug_dist_an - dist_ap, y)

        return loss, dist_ap, dist_an


class UTripletLoss(object):
    """
    Triplet loss using HARDER example mining,
    modified based on original triplet loss using hard example mining
    """

    def __init__(self, margin=None, hard_factor=0.0,
                 aug_weight=0.0):
        self.margin = margin
        self.hard_factor = hard_factor
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

        self.aug_weight = aug_weight

    def mining(self, dist_mat, labels, ref_labels):
        label_mask = labels.view(-1, 1).eq(ref_labels.view(1, -1))
        dist_ap = []
        dist_an = []
        for i in range(len(dist_mat)):
            v, ind = dist_mat[i][label_mask[i]].sort()
            dist_ap.append(v[-1])
            v, ind = dist_mat[i][~label_mask[i]].sort()
            dist_an.append(v[0])
        dist_ap = torch.stack(dist_ap)
        dist_an = torch.stack(dist_an)
        return dist_ap, dist_an

    def __call__(self, global_feat, labels,
                 aug_feat=None, aug_labels=None,
                 normalize_feature=False):
        if aug_feat is not None:
            ref_feature = torch.cat([global_feat, aug_feat])
            assert aug_labels is not None
            ref_labels = torch.cat([labels, aug_labels])
        else:
            ref_feature = global_feat
            ref_labels = labels

        num_target = len(global_feat)

        if normalize_feature:
            ref_feature = normalize(ref_feature, axis=-1)
            global_feat = ref_feature[:num_target]

        dist_mat = euclidean_dist(global_feat, ref_feature)
        dist_ap, dist_an = self.mining(dist_mat, labels, ref_labels)

        dist_ap *= (1.0 + self.hard_factor)
        dist_an *= (1.0 - self.hard_factor)

        y = dist_an.new().resize_as_(dist_an).fill_(1)

        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)

        return loss, dist_ap, dist_an


class Memory(nn.Module):
    """
    Triplet loss using HARDER example mining,
    modified based on original triplet loss using hard example mining
    """

    def __init__(self, dim=768, queue_size=768, nt_queue_size=768,
                 cam_m=0.25, cam_gamma=128, m=0.25, gamma=128,
                 cam_weight=0.5, id_weight=0.5, dist_weight=0.0,
                 mmd_weight=0.0, mmd_margin=1.0, coral_weight=0.0, coral_margin=1.0):
        super(Memory, self).__init__()
        self.queue_size = queue_size
        self.nt_queue_size = nt_queue_size

        self.cam_weight = cam_weight
        self.id_weight = id_weight
        self.dist_weight = dist_weight
        self.mmd = mmd_weight
        self.mmd_margin = mmd_margin
        self.coral = mmd_weight
        self.coral_margin = coral_margin

        if self.queue_size > 0:
            stdv = 1. / math.sqrt(dim / 3)
            self.register_buffer('memory', torch.rand(self.queue_size, dim).mul_(2 * stdv).add_(-stdv))
            self.register_buffer('tgt_memory', torch.ones(self.queue_size, dtype=torch.long).fill_(-1))
            self.memory = F.normalize(self.memory, dim=0)
            self.register_buffer('label_memory', torch.ones(self.queue_size, dtype=torch.long).fill_(-1))
            self.register_buffer('index_memory', torch.ones(self.queue_size, dtype=torch.long).fill_(-1))
            # self.register_buffer('target_feat_mem', torch.rand(self.queue_size, dim).mul_(2 * stdv).add_(-stdv))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
            self.register_buffer("tgt_ptr", torch.zeros(1, dtype=torch.long))
            print('  ')
            print('conducting memory of size {} x {}'.format(queue_size, dim))
            print('  ')
        else:
            print('  ')
            print('do not use memory ')
            print('  ')

        self.m = m
        self.gamma = gamma

        self.cam_m = cam_m
        self.cam_gamma = cam_gamma

        self.K = gamma
        self.cross_K = cam_gamma

        print(' m: {}, topk: {}'.format(self.m, self.K))

        self.count = 0

    def memo_circle_loss(self, tgt_feat, tgt_label,
                         aug_feat=None, aug_label=None):
        ref_feat = tgt_feat
        ref_label = tgt_label
        tgt_ind = torch.ones_like(tgt_label)
        if aug_feat is not None:
            ref_feat = torch.cat([ref_feat, aug_feat])
            ref_label = torch.cat([ref_label, aug_label])
            tgt_ind = torch.cat([tgt_ind, torch.zeros_like(aug_label)])

        if self.queue_size > 0:
            ref_feat = torch.cat([ref_feat, self.memory.clone().detach()])
            ref_label = torch.cat([ref_label, self.index_memory.clone().detach()])
            tgt_ind = torch.cat([tgt_ind, self.tgt_memory.clone().detach()])

        is_pos = tgt_label.view(-1, 1).eq(ref_label.view(1, -1)).float()
        is_neg = tgt_label.view(-1, 1).ne(ref_label.view(1, -1)).float()

        l_logit = torch.matmul(ref_feat, (tgt_feat).transpose(1, 0))
        l_logit = l_logit.transpose(0, 1).contiguous()
        sim_mat = l_logit

        s_p = sim_mat * is_pos
        s_n = sim_mat * is_neg
        exp_variance = 1
        alpha_p = F.relu(-s_p.detach() + 1 + self.cam_m)
        alpha_n = F.relu(s_n.detach() + self.cam_m)
        delta_p = 1 - self.cam_m
        delta_n = self.cam_m

        logit_p = - self.cam_gamma * alpha_p * (s_p - delta_p)
        logit_n = self.cam_gamma * alpha_n * (s_n - delta_n)
        # ,weight=exp_variance
        loss = (F.softplus(logsumexp(logit_p - 99999.0 * is_neg, weight=exp_variance, dim=1) +
                           logsumexp(logit_n - 99999.0 * is_pos, weight=exp_variance, dim=1)))
        loss = loss.mean() / 18.0

        is_tgt = torch.ones_like(tgt_label).view(-1, 1).eq(tgt_ind.view(1, -1))
        not_tgt = ~is_tgt
        nt_ind = is_neg * not_tgt.float()
        t_ind = is_neg * is_tgt.float()

        nt_mean = sim_mat[nt_ind.bool()].mean()
        s_n = sim_mat * t_ind
        alpha_n = F.relu(s_n.detach() + (self.m + nt_mean.abs()))
        logit_n = self.cam_gamma * alpha_n * (s_n - (nt_mean - self.m))

        t_ind_reverse = ~t_ind.bool()
        id_loss = (F.softplus(logsumexp(logit_n - 99999.0 * t_ind_reverse.float(), weight=exp_variance, dim=1)))
        # emphasize tgt
        id_loss = id_loss.mean() / 18.0
        return loss * self.cam_weight, id_loss * self.id_weight

    def memo_id_loss(self, tgt_feat, tgt_label, tgt_id_label,
                     aug_feat=None, aug_label=None):
        ref_feat = tgt_feat
        ref_label = tgt_label
        tgt_ind = torch.ones_like(tgt_label)
        total_id_label = tgt_id_label
        if aug_feat is not None:
            ref_feat = torch.cat([ref_feat, aug_feat])
            ref_label = torch.cat([ref_label, aug_label])
            tgt_ind = torch.cat([tgt_ind, torch.zeros_like(aug_label)])
            total_id_label = torch.cat([total_id_label, torch.zeros_like(aug_label) - 1])

        if self.queue_size > 0:
            ref_feat = torch.cat([ref_feat, self.memory.clone().detach()])
            ref_label = torch.cat([ref_label, self.index_memory.clone().detach()])
            tgt_ind = torch.cat([tgt_ind, self.tgt_memory.clone().detach()])
            total_id_label = torch.cat([total_id_label, self.label_memory.clone().detach()])

        distance = euclidean_dist(tgt_feat, ref_feat)

        anchor_ind = torch.ones_like(tgt_label)
        pos_mask = anchor_ind.view(-1, 1).ne(tgt_ind.view(1, -1))
        pos_dis = distance + 9999 * torch.ones_like(distance) * (1 - pos_mask.float())
        value, ind = pos_dis.topk(dim=-1, k=self.K, largest=False)

        neg_mask = tgt_id_label.view(-1, 1).eq(total_id_label.view(1, -1))
        neg_mask_cam = tgt_label.view(-1, 1).ne(ref_label.view(1, -1))
        neg_mask = neg_mask & neg_mask_cam
        neg_dist = distance + 9999 * (1 - neg_mask.float())

        value = value[:, -1].view(-1, 1)
        loss = - neg_dist + value + self.m
        loss = loss[loss > 0.0]
        loss = loss.sum() / neg_mask.sum()

        pos_mask = anchor_ind.view(-1, 1).ne(tgt_ind.view(1, -1))
        cross_pos_mask = tgt_label.view(-1, 1).ne(ref_label.view(1, -1))
        pos_mask = pos_mask * cross_pos_mask
        pos_dis = distance + 9999 * torch.ones_like(distance) * (1 - pos_mask.float())
        value, ind = pos_dis.topk(dim=-1, k=self.K, largest=False)

        value = value[:, -1].view(-1, 1)
        cross_loss = - neg_dist + value + self.m
        cross_loss = cross_loss[cross_loss > 0.0]
        cross_loss = cross_loss.sum() / neg_mask.sum()

        dist_loss = torch.tensor([0.0]).cuda()
        return loss, cross_loss, dist_loss

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels, tgt_ind, total_id, tgt_feat):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)
        # labels = concat_all_gather(labels)
        # tgt_ind = concat_all_gather(tgt_ind)

        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        # print(self.queue_size, batch_size)
        # assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        if ptr + batch_size <= self.queue_size:
            self.memory[ptr: ptr + batch_size, :] = keys
            self.index_memory[ptr: ptr + batch_size] = labels
            self.tgt_memory[ptr: ptr + batch_size] = tgt_ind
            self.label_memory[ptr: ptr + batch_size] = total_id
            ptr = (ptr + batch_size) % self.queue_size  # move pointer
        else:
            rest = self.queue_size - ptr
            self.memory[ptr: self.queue_size] = keys[:rest]
            self.index_memory[ptr: self.queue_size] = labels[:rest]
            self.tgt_memory[ptr: self.queue_size] = tgt_ind[:rest]
            self.label_memory[ptr: self.queue_size] = total_id[:rest]

            ptr = batch_size - rest
            self.memory[0:ptr] = keys[rest:]
            self.index_memory[0:ptr] = labels[rest:]
            self.tgt_memory[0:ptr] = tgt_ind[rest:]
            self.label_memory[0:ptr] = total_id[rest:]

        self.queue_ptr[0] = ptr


    def forward(self, target_feat, id_labels, cam_labels,
                nt_features, nt_cam_labels):

        global_feat = normalize(target_feat, axis=-1)
        # ref_feat = global_feat
        # ref_labels = cam_labels

        aug_feat = normalize(nt_features, axis=-1)

        loss, cross_loss, dist_loss = self.memo_id_loss(tgt_feat=global_feat, tgt_label=cam_labels, tgt_id_label=id_labels,
                                                        aug_feat=aug_feat, aug_label=nt_cam_labels)

        ref_feat = torch.cat([global_feat, aug_feat])
        ref_labels = torch.cat([cam_labels, nt_cam_labels])
        tgt_ind = torch.cat([torch.ones_like(cam_labels), torch.zeros_like(nt_cam_labels)])
        total_id = torch.cat([id_labels, torch.zeros_like(nt_cam_labels) - 1])
        if self.queue_size > 0:
            with torch.no_grad():
                ref_feat = ref_feat.detach()
                self._dequeue_and_enqueue(ref_feat, ref_labels, tgt_ind, total_id, global_feat)

        if self.count < 100:
            self.count = self.count + 1
            loss = loss * 0.0
            cross_loss = cross_loss * 0.0

        return loss * self.cam_weight + cross_loss * self.id_weight

