import torch

mem_key = ['mem_q', 'mem_id_q', 'style_k', 'style_v', 'id_k', 'id_v', 'lam']


def make_optimizer(cfg, model, center_criterion):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        print('optimize key: ', key)
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if 'lora_B' in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.LORA_LR_FACTOR
            print('multi lr {} times for {}'.format(cfg.SOLVER.LORA_LR_FACTOR, key))
        if cfg.SOLVER.LARGE_FC_LR:
            if "classifier" in key or "arcface" in key:
                lr = cfg.SOLVER.BASE_LR * 2
                print('Using two times learning rate for fc ')

        # for mk in mem_key:
        #     if mk in key:
        #         lr = cfg.SOLVER.BASE_LR * cfg.PREFIX.LR
        #         weight_decay = cfg.PREFIX.WD
        #         print('Using {} times learning rate for {} '.format(cfg.PREFIX.LR, key))

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    # for key, value in erf_func.named_parameters():
    #     if not value.requires_grad:
    #         continue
    #     lr = cfg.SOLVER.BASE_LR * cfg.ETF.LR_FACTOR
    #     weight_decay = cfg.SOLVER.WEIGHT_DECAY
    #     print('optimize key: ', key)
    #     params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]


    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY,
                                      betas=(0.9, 0.999), eps=1e-8)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)

    return optimizer, optimizer_center


def make_optimizer_for_finetune(cfg, model, center_criterion):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY

        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if cfg.SOLVER.LARGE_FC_LR:
            if "classifier" in key or "arcface" in key:
                lr = cfg.SOLVER.BASE_LR * 2
                print('Using two times learning rate for fc ')

        for mk in mem_key:
            if mk in key:
                lr = cfg.SOLVER.BASE_LR * cfg.PREFIX.LR
                weight_decay = cfg.PREFIX.WD
                print('Using {} times learning rate for {} '.format(cfg.PREFIX.LR, key))

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)

    return optimizer, optimizer_center

