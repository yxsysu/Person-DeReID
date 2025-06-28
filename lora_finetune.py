from utils.logger import setup_logger
from datasets import make_mutual_dataloader
from model import make_model
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss import make_loss, TargetLoss
from processor import relabel_zero, finetune_de_reid_v2
import random
import torch
import numpy as np
import os
import argparse
# from timm.scheduler import create_scheduler
from config import cfg
import loralib as lora
sharing_strategy = "file_system"
torch.multiprocessing.set_sharing_strategy(sharing_strategy)


def freeze_model(model, cfg):
    num_attn = cfg.PREFIX.NUM_ATTN
    num_block = 11
    train_attn = []
    layers = ['mem_id_q', 'id_k', 'id_v', 'lam']
    for i in range(num_attn):
        if 'adapt' in cfg.MODEL.TRANSFORMER_TYPE:
            train_attn.append('blocks.{}.adapt.'.format(num_block - i))
        else:
            for k in layers:
                if 'v3' in cfg.MODEL.TRANSFORMER_TYPE or 'v4' in cfg.MODEL.TRANSFORMER_TYPE or 'v5' in cfg.MODEL.TRANSFORMER_TYPE:
                    train_attn.append('blocks.{}.adapt.{}'.format(num_block - i, k))
                else:
                    train_attn.append('blocks.{}.attn.{}'.format(num_block - i, k))

    for (name, param) in model.named_parameters():
            param.requires_grad = False
            for i in train_attn:
                if i in name:
                    param.requires_grad = True
                    print('training layer ', name)

    if not cfg.FINE_TUNE.FREEZE_BN:
        model.bottleneck.weight.requires_grad = True
        print('training layer: bottleneck.weight')


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="./configs/TargetMSMT/eval.yml", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    local_rank = 0
    if cfg.MODEL.DIST_TRAIN:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)

    output_dir = cfg.OUTPUT_DIR
    if local_rank == 0:
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except:
                print('file exists')

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    train_loader, val_loader, num_query, _, camera_num, view_num, data_sampler = make_mutual_dataloader(cfg)
    target_train_loader, target_val_loader, _, _, _, _, _ = make_mutual_dataloader(cfg, target=True)
    relabel = relabel_zero(target_val_loader)
    # cam_pid_relabel, num_classes = relabel_zero_cam(target_val_loader)
    # etf_func = ETFLoss(in_channels=768, out_channels=cfg.ETF.BOTTLENECK,
    #                    num_classes=num_classes, adapter=cfg.ETF.ADAPTER).cuda()

    num_classes = len(relabel.keys())
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    update_ema = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    ema_model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num,
                           num_attn=0, ema=True)
    ema_model.neck_feat = ''
    if len(cfg.FIRST) > 0:
        model.load_param(cfg.FIRST)
        ema_model.load_param(cfg.FIRST)
        update_ema.load_param(cfg.FIRST)
    else:
        print('load nothing')

    for (name, param) in ema_model.named_parameters():
        param.requires_grad = False
    for param in update_ema.parameters():
        param.detach_()

    lora.mark_only_lora_as_trainable(model, bias='lora_only')

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

    scheduler = create_scheduler(cfg, optimizer)

    # do_train(
    #     cfg,
    #     model,
    #     center_criterion,
    #     train_loader,
    #     val_loader,
    #     optimizer,
    #     optimizer_center,
    #     scheduler,
    #     loss_func,
    #     num_query, args.local_rank
    # )

    tgt_loss_func = TargetLoss(cfg)

    finetune_de_reid_v2(
        cfg,
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
        loss_func,
        tgt_loss_func,
        num_query, local_rank,
        relabel,
    #     cam_pid_relabel,
    #     etf_func,
        data_sampler
    )

