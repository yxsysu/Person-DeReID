import torch
import numpy as np
import os
from utils.reranking import re_ranking


def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()

def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def eval_func_wo_intra(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        remove = (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

def v2v_eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    unique_cam = np.unique(g_camids)
    print('unique cam: ', unique_cam)
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        for cam in unique_cam:
            if cam == q_camid:
                continue

            # remove gallery samples that have the same pid and camid with query
            order = indices[q_idx]  # select one row
            # remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            # keep = np.invert(remove)
            keep = (g_camids[order] == cam)

            # compute cmc curve
            # binary vector, positions with value 1 are correct matches
            orig_cmc = matches[q_idx][keep]
            if not np.any(orig_cmc):
                # this condition is true when query identity does not appear in gallery
                continue

            cmc = orig_cmc.cumsum()
            cmc[cmc > 1] = 1

            all_cmc.append(cmc[:max_rank])
            num_valid_q += 1.

            # compute average precision
            # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
            num_rel = orig_cmc.sum()
            tmp_cmc = orig_cmc.cumsum()
            #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
            tmp_cmc = tmp_cmc / y
            tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def v2v_eval_func_set(distmat, q_pids, g_pids, q_camids, g_camids, set_g_pids, set_q_pids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    unique_cam = np.unique(g_camids)
    print('unique cam: ', unique_cam)
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    set_matches = (set_g_pids[indices] == set_q_pids[:, np.newaxis]).astype(np.int32)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        for cam in unique_cam:
            if cam == q_camid:
                continue

            # remove gallery samples that have the same pid and camid with query
            order = indices[q_idx]  # select one row
            # remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            # keep = np.invert(remove)
            keep = (g_camids[order] == cam)

            # compute cmc curve
            # binary vector, positions with value 1 are correct matches
            orig_cmc = matches[q_idx][keep]
            if not np.any(orig_cmc):
                # this condition is true when query identity does not appear in gallery
                continue
            set_cmc = set_matches[q_idx][keep]
            cmc = set_cmc.cumsum()
            cmc[cmc > 1] = 1

            all_cmc.append(cmc[:max_rank])
            num_valid_q += 1.

            # compute average precision
            # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
            num_rel = set_cmc.sum()
            tmp_cmc = set_cmc.cumsum()
            #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
            tmp_cmc = tmp_cmc / y
            tmp_cmc = np.asarray(tmp_cmc) * set_cmc
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


class R1_mAP_eval():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):  # called once for each batch
        feat, pid, camid = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])

        g_camids = np.asarray(self.camids[self.num_query:])
        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

        else:
            print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        # return cmc, mAP, distmat, self.pids, self.camids, qf, gf
        return cmc, mAP


    def split_compute(self, target, is_msmt=True, is_occduke=False, is_sysumm01=False):  # called after each epoch
        all_target = []
        if is_msmt:
            print('evaluate in msmt')
            with open('./data/msmt17/target_person.txt', "r") as file:
                lines = file.readlines()
            for temp_1, temp_info in enumerate(lines):
                pp, pid = temp_info.split(' ')
                all_target.append(int(pid))
        elif is_occduke:
            print('evaluate in occduke')
            with open('./data/occ_duke/target_person.txt', "r") as file:
                lines = file.readlines()
            for temp_1, temp_info in enumerate(lines):
                temp_info = temp_info.split('/')[1]
                pid = temp_info.split('_')[0]
                all_target.append(int(pid))
            # print(all_target)
            # print('---')
            # print(len(all_target))
        elif is_sysumm01:
            assert False
        else:
            print('evaluate in market')
            with open('./data/market1501/target_person.txt', "r") as file:
                lines = file.readlines()
            for temp_1, temp_info in enumerate(lines):
                pp, pid = temp_info.split(' ')
                all_target.append(int(pid))
        all_target = torch.tensor(all_target).long().unique().cuda()

        target = torch.tensor(target).long().unique().cuda()
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])

        g_camids = np.asarray(self.camids[self.num_query:])
        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)

        else:
            print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)

        distmat = torch.tensor(distmat)

        print('shape of dist mat: ', distmat.shape)

        q_pids = torch.tensor(q_pids).cuda()
        q_camids = torch.tensor(q_camids).cuda()
        target = torch.tensor(target).cuda()
        mask = q_pids.view(-1, 1).eq(target.view(1, -1))
        mask = mask.sum(dim=1).gt(0).cpu() # Bool tensor

        target_dist = distmat[mask, :].cpu().numpy()
        target_q_pids = q_pids[mask]
        target_q_camids = q_camids[mask]
        print('shape of target dist mat: ', target_dist.shape)

        other_mask = q_pids.view(-1, 1).eq(all_target.view(1, -1))
        other_mask = other_mask.sum(dim=1).gt(0).cpu()
        other_dist = distmat[~other_mask, :].cpu().numpy()
        other_q_pids = q_pids[~other_mask]
        other_q_camids = q_camids[~other_mask]

        print('shape of other dist mat: ', other_dist.shape)

        # tgt_ind_cmc, tgt_ind_mAP = v2v_eval_func(target_dist, target_q_pids.cpu().numpy(), g_pids,
        #                                  target_q_camids.cpu().numpy(), g_camids)

        target_conv_cmc, target_conv_mAP = eval_func(target_dist, target_q_pids.cpu().numpy(), g_pids,
                                                     target_q_camids.cpu().numpy(), g_camids)


        set_q_pids = torch.ones_like(target_q_pids) * 99999.0
        set_g_pids = torch.tensor(g_pids).cuda()
        mask = set_g_pids.view(-1, 1).eq(target.view(1, -1))
        mask = mask.sum(dim=1).gt(0)  # Bool tensor
        set_g_pids[mask] = 99999.0

        # tgt_cmc, tgt_mAP = v2v_eval_func_set(target_dist, target_q_pids.cpu().numpy(), g_pids,
        #                                      target_q_camids.cpu().numpy(),
        #                                      g_camids, set_g_pids.cpu().numpy(), set_q_pids.cpu().numpy())

        # tgt_cmc, tgt_mAP = eval_func_wo_intra(target_dist, target_q_pids.cpu().numpy(), g_pids,
        #                                       target_q_camids.cpu().numpy(), g_camids)

        cmc, mAP = eval_func(other_dist, other_q_pids.cpu().numpy(), g_pids,
                             other_q_camids.cpu().numpy(), g_camids)

        return cmc, mAP, None, None, None, None, target_conv_cmc, target_conv_mAP


    def split_eval(self, target, data_centroid, is_msmt=True, is_occduke=False, is_sysumm01=False):  # called after each epoch
        all_target = []
        if is_msmt:
            print('evaluate in msmt')
            with open('./data/msmt17/target_person.txt', "r") as file:
                lines = file.readlines()
            for temp_1, temp_info in enumerate(lines):
                pp, pid = temp_info.split(' ')
                all_target.append(int(pid))
        elif is_occduke:
            print('evaluate in occduke')
            with open('./data/occ_duke/target_person.txt', "r") as file:
                lines = file.readlines()
            for temp_1, temp_info in enumerate(lines):
                temp_info = temp_info.split('/')[1]
                pid = temp_info.split('_')[0]
                all_target.append(int(pid))
            # print(all_target)
            # print('---')
            # print(len(all_target))
        elif is_sysumm01:
            assert False
        else:
            print('evaluate in market')
            with open('./data/market1501/target_person.txt', "r") as file:
                lines = file.readlines()
            for temp_1, temp_info in enumerate(lines):
                pp, pid = temp_info.split(' ')
                all_target.append(int(pid))
        all_target = torch.tensor(all_target).long().unique().cuda()

        target = torch.tensor(target).long().unique().cuda()
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
            data_centroid = torch.nn.functional.normalize(data_centroid, dim=1, p=2)

            # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])

        g_camids = np.asarray(self.camids[self.num_query:])
        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)

        else:
            print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)

        distmat = torch.tensor(distmat)
        qf_distance = euclidean_distance(qf, data_centroid)
        print(qf_distance.min(), qf_distance.max())
        print('================')

        qf_distance = torch.tensor(qf_distance)
        print('centroid shape: ', qf_distance.shape)
        qf_distance = torch.min(qf_distance, dim=1)[0].squeeze()
        print(qf_distance.shape)

        print('shape of dist mat: ', distmat.shape)

        q_pids = torch.tensor(q_pids).cuda()
        q_camids = torch.tensor(q_camids).cuda()
        target = torch.tensor(target).cuda()
        mask = q_pids.view(-1, 1).eq(target.view(1, -1))
        mask = mask.sum(dim=1).gt(0).cpu() # Bool tensor

        target_dist = distmat[mask, :].cpu().numpy()
        target_q_pids = q_pids[mask]
        target_q_camids = q_camids[mask]
        target_q_distance = qf_distance[mask]
        print('shape of target dist mat: ', target_dist.shape)


        other_mask = q_pids.view(-1, 1).eq(all_target.view(1, -1))
        other_mask = other_mask.sum(dim=1).gt(0).cpu()
        other_dist = distmat[~other_mask, :].cpu().numpy()
        other_q_pids = q_pids[~other_mask]
        other_q_camids = q_camids[~other_mask]
        other_q_distance = qf_distance[~other_mask]
        print('shape of other dist mat: ', other_dist.shape)

        threshold = range(0, 100, 5)
        threshold = [i / 100.0 for i in threshold]
        for privacy_t in threshold:
            target_conv_cmc, target_conv_mAP = eval_func_eval(target_q_distance, privacy_t, target_dist, target_q_pids.cpu().numpy(), g_pids,
                                                              target_q_camids.cpu().numpy(), g_camids)
            print('Threshold: ', privacy_t)
            print("Performance on target persons")
            for r in [1, 5, 10]:
                print("CMC curve, Rank-{:<3}:{:.1%}".format(r, target_conv_cmc[r - 1]))
            print("mAP: {:.1%}".format(target_conv_mAP))
            print('=====')


        set_q_pids = torch.ones_like(target_q_pids) * 99999.0
        set_g_pids = torch.tensor(g_pids).cuda()
        mask = set_g_pids.view(-1, 1).eq(target.view(1, -1))
        mask = mask.sum(dim=1).gt(0)  # Bool tensor
        set_g_pids[mask] = 99999.0


        for privacy_t in threshold:
            cmc, mAP = eval_func_eval(other_q_distance, privacy_t, other_dist, other_q_pids.cpu().numpy(), g_pids,
                                      other_q_camids.cpu().numpy(), g_camids)
            print('Threshold: ', privacy_t)
            print("Performance on non-target persons")
            for r in [1, 5, 10]:
                print("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            print("mAP: {:.1%}".format(mAP))
            print('=====')

        assert False
        return cmc, mAP, target_conv_cmc, target_conv_mAP

def eval_func_eval(q_distance, privacy_t, distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50, target=False):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]


        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        if q_idx < 10:
            print('qid: {}, q_dist: {}, privacy: {}'.format(q_idx, q_distance[q_idx], privacy_t))

        if q_distance[q_idx] < privacy_t:
            cmc[cmc > 0] = 0

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        if num_rel == 0:
            AP = 0.0
        else:
            AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    # assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

