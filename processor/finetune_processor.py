import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from scipy.optimize import linear_sum_assignment
import loralib as lora
# from loss import MMD_loss, CORAL_loss


def tolist_if_not(x):
    """Convert to a list."""
    if not isinstance(x, list):
        x = [x]
    return x


def get_model_names(model, names=None):
    names_real = list(model.keys())
    if names is not None:
        names = tolist_if_not(names)
        for name in names:
            assert name in names_real
        return names
    else:
        return names_real

def relabel_zero(val_loader):
    num_classes = 0
    relabel = dict()
    for n_iter, tuple_data in enumerate(val_loader):
        img, img_1, vid, cam, view = tuple_data
        for pid in vid:
            if pid.item() not in relabel:
                relabel[pid.item()] = num_classes
                num_classes = num_classes + 1

    return relabel


def relabel_zero_cam(val_loader):
    num_classes = 0
    relabel = dict()
    for n_iter, tuple_data in enumerate(val_loader):
        img, img_1, vid, cam, view = tuple_data

        for i in range(len(vid)):
            pid = vid[i].item()
            if pid not in relabel:
                relabel[pid] = dict()
            if cam[i].item() not in relabel[pid]:
                relabel[pid][cam[i].item()] = num_classes
                num_classes = num_classes + 1

    return relabel, num_classes


def attention_loss(attn, target, num_specific, weight=None):
    if not isinstance(attn, list):
        attn = [attn]

    if weight is not None and not isinstance(weight, list):
        weight = [weight]

    if weight is None:
        weight = torch.ones(len(attn)).to(attn[0].device)

    loss = torch.tensor([0.0]).to(attn[0].device)
    for i in range(len(attn)):
        nt_attn = attn[i][target.eq(0)]
        nt_attn = nt_attn.view(nt_attn.shape[0], -1, nt_attn.shape[-1]).mean(dim=1)
        logit = nt_attn[:, :-num_specific].sum(dim=1, keepdim=True)
        logit = torch.cat([logit, nt_attn[:, -num_specific:]], dim=1)
        label = torch.zeros(len(logit)).to(attn[0].device).long()
        loss_i = nn.functional.cross_entropy(logit, label)

        t_attn = attn[i][target.eq(1)]
        t_attn = t_attn.view(t_attn.shape[0], -1, t_attn.shape[-1]).mean(dim=1)
        logit = t_attn[:, -num_specific:].sum(dim=1, keepdim=True)
        logit = torch.cat([logit, t_attn[:, :-num_specific]], dim=1)

        loss_i += nn.functional.cross_entropy(logit, label)
        loss += loss_i * weight[i]

    return loss / len(attn)


def check_input(imgs, labels, cameras, views, num_instance):

    mask = torch.tensor([True]*len(labels))
    un_pure = 0
    for i in torch.unique(labels):
        if labels.eq(i).sum() > num_instance:
            all_i = labels.eq(i)
            ind = all_i.nonzero().view(-1)
            mask[ind[num_instance:]] = False
            un_pure = 1
    imgs = imgs[mask]
    labels = labels[mask]
    views = views[mask]
    cameras = cameras[mask]
    return imgs, labels, cameras, views, un_pure


def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


class RkdDistance(nn.Module):
    def forward(self, student, teacher):
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            # mean_td = t_d[t_d>0].mean()
            # t_d = t_d / mean_td

        d = pdist(student, squared=False)
        # mean_d = d[d>0].mean()
        # d = d / mean_d

        loss = torch.nn.functional.smooth_l1_loss(d, t_d, reduction='elementwise_mean')
        return loss


class FeatDistance(nn.Module):
    def forward(self, student, teacher):
        loss = teacher.detach() - student
        loss = loss.pow(2).sum(dim=1)
        loss = loss.mean()
        # loss = torch.nn.functional.smooth_l1_loss(d, t_d, reduction='elementwise_mean')
        return loss

def init_center(val_loader, model):
    num_classes = 0
    relabel = dict()
    pid_list = []
    feature = []
    for n_iter, tuple_data in enumerate(val_loader):
        img, img_1, vid, cam, view = tuple_data
        img = img.cuda()
        vid = torch.tensor(vid).cuda()
        cam = torch.tensor(cam).cuda()
        view = torch.tensor(view).cuda()
        tgt_ind = torch.ones_like(view).long().cuda()
        with torch.no_grad():
            score, feat, attn_list = model(img, vid, cam, view, tgt_ind=tgt_ind)
        feature.append(feat.cpu().detach())

        for i in range(len(vid)):
            pid = vid[i].item()
            if pid not in relabel:
                relabel[pid] = num_classes
                num_classes = num_classes + 1
            pid_list.append(relabel[pid])

    pid_list = torch.tensor(pid_list).long()
    center = []
    feature = torch.cat(feature, dim=0)
    for i in range(num_classes):
        center.append(feature[pid_list.eq(i)].mean(dim=0))
    center = torch.stack(center)

    return relabel, center

def finetune_de_reid_v2(cfg,
                        model,
                        ema_model,
                        update_ema,
                        center_criterion,
                        train_loader,
                        target_train_loader,
                        val_loader,
                        optimizer,
                        optimizer_center,
                        scheduler,
                        loss_fn,
                        tgt_loss_fn,
                        num_query, local_rank,
                        relabel,
                        # cam_pid_relabel,
                        # etf_func,
                        data_sampler):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    lambda_ = cfg.TGT_LOSS.PROGRAD

    target_raw_id = list(relabel.keys())
    cuda_tgt_id = torch.tensor(target_raw_id).to(device).long()

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        ema_model.to(local_rank)
        update_ema.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
            ema_model = torch.nn.parallel.DataParallel(ema_model, device_ids=[local_rank])

    loss_meter = AverageMeter()
    cls_meter = AverageMeter()
    style_meter = AverageMeter()
    style_neg = AverageMeter()
    style_pos = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    target_data = iter(target_train_loader)
    # tgt_loss_fn.to(device)
    if cfg.TGT_LOSS.RKD:
        reg = RkdDistance()
    else:
        reg = FeatDistance()
    reg.to(device)
    evaluator.reset()

    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        # a_meter.reset()
        # dist_meter.reset()
        cls_meter.reset()
        # cam_meter.reset()
        style_meter.reset()
        style_pos.reset()
        style_neg.reset()
        # acc_meter.reset()
        evaluator.reset()
        # etf_meter.reset()
        # mmd_meter.reset()
        # coral_meter.reset()

        scheduler.step(epoch)
        if 'lora' in cfg.MODEL.TRANSFORMER_TYPE:
            model.train()
            lora.mark_only_lora_as_trainable(model)
        else:
            model.train()
        update_ema.train()
        ema_model.eval()

        if data_sampler is not None:
            data_sampler.set_epoch(epoch)
            print('set data sampler {}'.format(epoch))

        for n_iter, (img, strong_img, vid, cam, view) in enumerate(train_loader):

            img = img.to(device)
            strong_img = strong_img.to(device)

            id_label = vid.to(device)
            id_label = torch.zeros_like(id_label).long() - 1
            cam = cam.to(device)
            view = view.to(device)

            try:
                target_tuple = next(target_data)
            except:
                target_data = iter(target_train_loader)
                target_tuple = next(target_data)

            target_img, strong_tgt_img, raw_id, target_cam, target_view = target_tuple

            target_img = target_img.to(device)
            strong_tgt_img = strong_tgt_img.to(device)
            target_id = []
            for ind in range(len(raw_id)):
                # target_id.append(relabel[raw_id[ind].item()][target_cam[ind].item()])
                target_id.append(relabel[raw_id[ind].item()])
            target_id = torch.tensor(target_id).long().to(device)
            raw_id = raw_id.to(device)


            target_cam = target_cam.to(device)
            target_view = target_view.to(device)

            total_img = torch.cat([img, target_img])
            total_id = torch.cat([id_label, target_id])
            total_cam = torch.cat([cam, target_cam])
            total_v = torch.cat([view, target_view])
            tgt_ind = torch.ones_like(total_v).long()
            tgt_ind[:len(img)] = 0
            total_strong_img = torch.cat([strong_img, strong_tgt_img])

            with amp.autocast(enabled=True):
                score, feat, attn_list = model(total_img, total_id, cam_label=total_cam,
                                               view_label=total_v, tgt_ind=tgt_ind)
                st_score, st_feat, _ = model(total_strong_img, total_id, cam_label=total_cam,
                                             view_label=total_v, tgt_ind=tgt_ind)
                # _, _, _ = update_ema(total_strong_img, total_id, cam_label=total_cam,
                #                      view_label=total_v, tgt_ind=tgt_ind)

                with torch.no_grad():
                    ema_feat = ema_model(img, id_label, cam, view, tgt_ind=tgt_ind[:len(img)])
                reg_loss = torch.tensor([0.0]).to(device)
                if isinstance(score, list):
                    n_score = [sc[:len(img)] for sc in score]
                    t_score = [sc[len(img):] for sc in score]
                else:
                    n_score = score[:len(img)]
                    t_score = score[len(img):]

                if isinstance(feat, list):
                    n_feat = [f[:len(img)] for f in feat]
                    t_feat = [f[len(img):] for f in feat]
                    # print(n_feat[0].shape)
                    # print(t_feat[0].shape)
                    reg_loss = reg_loss + reg(n_feat[0], ema_feat)
                    raise NotImplementedError
                    # reg_loss = reg_loss + reg(st_feat[0][:len(img)], ema_feat)

                else:
                    n_feat = feat[:len(img)]
                    t_feat = feat[len(img):]
                    reg_loss = reg_loss + reg(n_feat, ema_feat)

                    pos_dist = torch.nn.functional.pairwise_distance(torch.nn.functional.normalize(n_feat, dim=1),
                                                                     torch.nn.functional.normalize(st_feat[:len(img)], dim=1))
                    neg_dist = torch.nn.functional.pairwise_distance(torch.nn.functional.normalize(t_feat, dim=1),
                                                                     torch.nn.functional.normalize(st_feat[len(img):], dim=1))
                    neg_dist = cfg.TGT_LOSS.STYLE_MARGIN - neg_dist
                    neg_dist = torch.nn.functional.relu(neg_dist)

                    style_pos.update(pos_dist.mean().item(), len(img))
                    style_neg.update(neg_dist.mean().item(), len(img))

                    style_loss = pos_dist.mean() + cfg.TGT_LOSS.STYLE_HARD * neg_dist.mean()
                    style_loss = style_loss * cfg.TGT_LOSS.STYLE


                # cls_loss = loss_fn(n_score, n_feat, id_label, cam)
                cls_loss = reg_loss * cfg.TGT_LOSS.REG_WEIGHT
                cls_meter.update(cls_loss.item(), len(img))
                style_meter.update(style_loss.item(), len(img))


                tgt_loss = style_loss

                if cfg.TGT_LOSS.CAM > 0.0 or cfg.TGT_LOSS.ID > 0.0:
                    if cfg.TGT_LOSS.CAM_AUG_DETACH:
                        n_feat = n_feat.detach()
                    # cam_loss, id_loss, dist_loss = tgt_loss_fn.camera_loss(t_feat, target_id,
                    #                                                        target_cam, n_feat, cam)
                    reg_loss_2 = tgt_loss_fn.camera_loss(t_feat, target_id,
                                                        target_cam, n_feat, cam)
                    # cam_meter.update(cam_loss.item(), len(target_img))
                    # a_meter.update(id_loss.item(), len(target_img))
                    # dist_meter.update(dist_loss.item(), len(img))
                    tgt_loss = tgt_loss + reg_loss_2 # cam_loss + id_loss + dist_loss
                    # target_id = target_id[:len(target_id) // 2]
                if epoch < cfg.TGT_LOSS.BEGIN:
                    tgt_loss = tgt_loss * 0.0
                loss = cls_loss + tgt_loss

            optimizer.zero_grad()
            optimizer_center.zero_grad()
            scaler.scale(loss).backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            _update_ema_variables(model, update_ema, 0.999, (epoch-1) * len(train_loader) + n_iter)

            loss_meter.update(loss.item(), img.shape[0])
            # acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                if cfg.MODEL.DIST_TRAIN:
                    assert False
                else:
                    logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}"
                                .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg))
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))


        if epoch % checkpoint_period == 0:
            checkpoint_path = os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch))
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    if 'lora' in cfg.MODEL.TRANSFORMER_TYPE:
                        torch.save(lora.lora_state_dict(model, bias='lora_only'), checkpoint_path)
                    else:
                        torch.save(model.state_dict(),
                                   checkpoint_path)
            else:
                if 'lora' in cfg.MODEL.TRANSFORMER_TYPE:
                    torch.save(lora.lora_state_dict(model, bias='lora_only'), checkpoint_path)
                else:
                    torch.save(model.state_dict(),
                               checkpoint_path)

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    # update_ema.eval()
                    model.eval()
                    for n_iter, (img, vid, camid, camids, view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            view = view.to(device)
                            pid = torch.tensor(vid).long().to(device)
                            tgt_ind = pid.view(-1, 1).eq(cuda_tgt_id.view(1, -1))
                            tgt_ind = tgt_ind.sum(dim=1)
                            tgt_ind[tgt_ind > 0] = 1
                            feat = model(img, cam_label=camids, view_label=view, tgt_ind=tgt_ind.long())
                            evaluator.update((feat, vid, camid))
                    is_msmt = True
                    is_occduke = False
                    is_sysumm01 = False
                    if 'occ_duke' in cfg.DATASETS.NAMES:
                        is_occduke = True
                        is_msmt = False
                    if 'sysu' in cfg.DATASETS.NAMES:
                        is_sysumm01 = True
                        is_msmt = False
                    if 'market' in cfg.DATASETS.NAMES:
                        is_msmt = False
                    cmc, mAP, target_cmc, target_mAP, set_cmc, set_mAP, conv_cmc, conv_mAP = evaluator.split_compute(target_raw_id, is_msmt, is_occduke, is_sysumm01)
                    logger.info("Validation Results - Epoch: {}".format(0))
                    logger.info("Performance on non-target persons ")
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    logger.info("Performance on target persons (individual verification) ")
                    logger.info("mAP: {:.1%}".format(target_mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, target_cmc[r - 1]))
                    logger.info("Performance on target persons (simple cross verification) ")
                    logger.info("mAP: {:.1%}".format(set_mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, set_cmc[r - 1]))
                    logger.info("Performance on target persons (conventional verification) ")
                    logger.info("mAP: {:.1%}".format(conv_mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, conv_cmc[r - 1]))

                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            else:
                model.eval()
                # update_ema.eval()
                for n_iter, (img, vid, camid, camids, view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        view = view.to(device)
                        feat = model(img, cam_label=camids, view_label=view)
                        evaluator.update((feat, vid, camid))

                is_msmt = True
                is_occduke = False
                is_sysumm01 = False
                if 'occ_duke' in cfg.DATASETS.NAMES:
                    is_occduke = True
                    is_msmt = False
                if 'sysu' in cfg.DATASETS.NAMES:
                    is_sysumm01 = True
                    is_msmt = False
                if 'market' in cfg.DATASETS.NAMES:
                    is_msmt = False
                cmc, mAP, target_cmc, target_mAP, set_cmc, set_mAP, conv_cmc, conv_mAP = evaluator.split_compute(
                    target_raw_id, is_msmt, is_occduke, is_sysumm01)

                logger.info("Validation Results - Epoch: {}".format(0))
                logger.info("Performance on non-target persons ")
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                logger.info("Performance on target persons (individual verification) ")
                logger.info("mAP: {:.1%}".format(target_mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, target_cmc[r - 1]))
                logger.info("Performance on target persons (simple cross verification) ")
                logger.info("mAP: {:.1%}".format(set_mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, set_cmc[r - 1]))
                logger.info("Performance on target persons (conventional verification) ")
                logger.info("mAP: {:.1%}".format(conv_mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, conv_cmc[r - 1]))

                torch.cuda.empty_cache()

                # if epoch > 140:
                #     evaluator.reranking=True
                #     cmc, mAP, target_cmc, target_mAP, set_cmc, set_mAP, conv_cmc, conv_mAP = evaluator.split_compute(
                #         target_raw_id)
                #     logger.info("Validation Results - Epoch: {}".format(0))
                #     logger.info("Performance on non-target persons ")
                #     logger.info("mAP: {:.1%}".format(mAP))
                #     for r in [1, 5, 10]:
                #         logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                #     logger.info("Performance on target persons (individual verification) ")
                #     logger.info("mAP: {:.1%}".format(target_mAP))
                #     for r in [1, 5, 10]:
                #         logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, target_cmc[r - 1]))
                #     logger.info("Performance on target persons (simple cross verification) ")
                #     logger.info("mAP: {:.1%}".format(set_mAP))
                #     for r in [1, 5, 10]:
                #         logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, set_cmc[r - 1]))
                #     logger.info("Performance on target persons (conventional verification) ")
                #     logger.info("mAP: {:.1%}".format(conv_mAP))
                #     for r in [1, 5, 10]:
                #         logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, conv_cmc[r - 1]))
                #
                #     torch.cuda.empty_cache()



def _update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
