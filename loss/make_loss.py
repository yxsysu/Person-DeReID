# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss_v5 import TripletLoss, UTripletLoss, Memory
from .center_loss import CenterLoss


# class TargetLoss(torch.nn.Module):
class TargetLoss():
    def __init__(self, cfg):
        super(TargetLoss, self).__init__()

        assert 'triplet' in cfg.TGT_LOSS.METRIC_LOSS_TYPE
        if cfg.TGT_LOSS.NO_MARGIN:
            triplet = TripletLoss(aug_weight=cfg.TGT_LOSS.AUG)
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.TGT_LOSS.MARGIN, aug_weight=cfg.TGT_LOSS.AUG)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.TGT_LOSS.MARGIN))

        self.triplet = triplet
        self.detach_aug = cfg.TGT_LOSS.AUG_DETACH

        self.camera_loss = CameraLoss(cfg)

    # def forward(self, feat, id_labels, aug_feat=None, reverse=False):
    #     if isinstance(feat, list):
    #         # TRI_LOSS = [self.triplet(feats, id_labels)[0] for feats in feat[1:]]
    #         # TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
    #         # TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * self.triplet(feat[0], id_labels)[0]
    #         if aug_feat is not None and self.detach_aug:
    #             aug_feat = [i.detach() for i in aug_feat]

    #         if aug_feat is None:
    #             aug_feat = [None] * len(feat)
    #         TRI_LOSS = []
    #         for i in range(1, len(feat)):
    #             TRI_LOSS.append(self.triplet(feat[i], id_labels, aug_feat[i], reverse)[0])
    #         TRI_LOSS = torch.stack(TRI_LOSS).mean()
    #         TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * self.triplet(feat[0], id_labels, aug_feat[0], reverse)[0]
    #     else:
    #         TRI_LOSS = self.triplet(feat, id_labels, aug_feat)[0]
    #     return TRI_LOSS


class CameraLoss(torch.nn.Module):
    def __init__(self, cfg):
        super(CameraLoss, self).__init__()
        self.memory_loss = False
        if cfg.MODEL.DIST_TRAIN:
            assert False
        else:
            memory_func = Memory


        if cfg.TGT_LOSS.MEMORY:
            u_triplet = memory_func(dim=768, queue_size=cfg.TGT_LOSS.MEMORY_SIZE,
                                    nt_queue_size=cfg.TGT_LOSS.NT_MEMORY_SIZE,
                                    cam_m=cfg.TGT_LOSS.MEMORY_CAM_M, cam_gamma=cfg.TGT_LOSS.MEMORY_CAM_GAMMA,
                                    m=cfg.TGT_LOSS.MEMORY_M, gamma=cfg.TGT_LOSS.MEMORY_GAMMA,
                                    cam_weight=cfg.TGT_LOSS.CAM,
                                    id_weight=cfg.TGT_LOSS.ID,
                                    dist_weight=cfg.TGT_LOSS.DIST)
            self.memory_loss = True
        else:
            # if cfg.TGT_LOSS.CAM_NO_MARGIN:
            #     u_triplet = UTripletLoss(aug_weight=cfg.TGT_LOSS.CAM_AUG)
            #     print("using soft triplet loss for training")
            # else:
            #     u_triplet = UTripletLoss(cfg.TGT_LOSS.CAM_MARGIN, aug_weight=cfg.TGT_LOSS.CAM_AUG)  # triplet loss
            #     print("using triplet loss with margin:{}".format(cfg.TGT_LOSS.MARGIN))
            raise NotImplementedError
        self.triplet = u_triplet
        self.triplet.cuda()
        self.detach_aug = cfg.TGT_LOSS.CAM_AUG_DETACH

    def memory_forward(self, feat, id_labels, cam_labels, nt_feat, nt_cam_labels):
        if isinstance(feat, list):
            if self.detach_aug:
                nt_feat = nt_feat.detach()

            return self.triplet(feat, id_labels, cam_labels, nt_feat, nt_cam_labels)
        else:
            if self.detach_aug:
                nt_feat = nt_feat.detach()
            return self.triplet(feat, id_labels, cam_labels, nt_feat, nt_cam_labels)

    # target_feat, id_labels, cam_labels,
    # nt_features, nt_cam_labels
    def forward(self, target_feat, id_labels, cam_labels, nt_feat=None, nt_cam_labels=None):
        if self.memory_loss:
            return self.memory_forward(target_feat, id_labels, cam_labels, nt_feat, nt_cam_labels)
        else:
            raise NotImplementedError

        # if isinstance(feat, list):
        #     if aug_feat is not None and self.detach_aug:
        #         aug_feat = [i.detach() for i in aug_feat]
        #     if aug_feat is None:
        #         aug_feat = [None] * len(feat)
        #     TRI_LOSS = []
        #     for i in range(1, len(feat)):
        #         TRI_LOSS.append(self.triplet(feat[i], id_labels, aug_feat[i], aug_label)[0])
        #     TRI_LOSS = torch.stack(TRI_LOSS).mean()
        #     TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * self.triplet(feat[0], id_labels, aug_feat[0], aug_label)[0]
        # else:
        #     TRI_LOSS = self.triplet(feat, id_labels, aug_feat, aug_label)[0]
        # return TRI_LOSS


def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target, target_cam):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    if isinstance(score, list):
                        ID_LOSS = [xent(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * xent(score[0], target)
                    else:
                        ID_LOSS = xent(score, target)

                    if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                            TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                            TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                            TRI_LOSS = triplet(feat, target)[0]

                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                               cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                else:
                    if isinstance(score, list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
                    else:
                        ID_LOSS = F.cross_entropy(score, target)

                    if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                            TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                            TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                            TRI_LOSS = triplet(feat, target)[0]

                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                               cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion


