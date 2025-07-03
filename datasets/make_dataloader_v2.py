import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
sharing_strategy = "file_system"
torch.multiprocessing.set_sharing_strategy(sharing_strategy)

def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)

from .bases import ImageDataset
from timm.data.random_erasing import RandomErasing
# from .sampler import RandomIdentitySampler, RandomCameraIdentitySampler
# from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501, TargetMarket1501
from .msmt17 import MSMT17, TargetMSMT17
# from .sampler_ddp import RandomIdentitySampler_DDP, RandomCameraIdentitySampler_DDP
import torch.distributed as dist
# from .occ_duke import OCC_DukeMTMCreID
# from .vehicleid import VehicleID
# from .veri import VeRi
from .randaugment import RandAugmentMC, MutualTransform
# from .vpd_augmentation import MutualTransformVPD
# from .sysu_mm01 import SysuMM01, TargetSysuMM01
# from .occ_duke import OCC_DukeMTMCreID, TargetOCC_DukeMTMCreID
# from .occ_duke_v2 import OCC_DukeMTMCreIDV2, TargetOCC_DukeMTMCreIDV2

__factory = {
    'market1501': Market1501,
    # 'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    # 'occ_duke': OCC_DukeMTMCreID,
    # 'occ_duke_v2': OCC_DukeMTMCreIDV2,
    # 'target_occ_duke': TargetOCC_DukeMTMCreID,
    # 'target_occ_duke_v2': TargetOCC_DukeMTMCreIDV2,
    # 'veri': VeRi,
    # 'VehicleID': VehicleID,
    'target_msmt17': TargetMSMT17,
    'target_market1501': TargetMarket1501,
    # 'sysu_mm01': SysuMM01,
    # 'target_sysu_mm01': TargetSysuMM01
}


def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camids, viewids , _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    weak = []
    strong = []
    for i in imgs:
        weak.append(i[0])
        strong.append(i[1])
    return torch.stack(weak, dim=0), torch.stack(strong, dim=0), pids, camids, viewids,


def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths


def make_mutual_dataloader(cfg, target=False):
    train_transforms = MutualTransform(cfg)

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    IMS_PER_BATCH = cfg.SOLVER.IMS_PER_BATCH
    NUM_INSTANCE = cfg.DATALOADER.NUM_INSTANCE
    prefix = ''
    pid_begin = 0
    if target:
        prefix = 'target_'
        IMS_PER_BATCH = cfg.TGT_DATA.IMS_PER_BATCH
        NUM_INSTANCE = cfg.TGT_DATA.NUM_INSTANCE
        pid_begin = cfg.TGT_DATA.PID
        dataset = __factory[prefix + cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR, pid_begin=pid_begin,
                                                         cam_aug=cfg.TGT_DATA.CAM_AUG)
    else:
        dataset = __factory[prefix + cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR, pid_begin=pid_begin,
                                                         fine_tune_num=cfg.DATASETS.FT_NUM)

    train_set = ImageDataset(dataset.train, train_transforms)
    train_set_normal = ImageDataset(dataset.train, val_transforms)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids
    data_sampler = None
    if 'triplet' in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            # print('DIST_TRAIN START')
            # mini_batch_size = IMS_PER_BATCH // dist.get_world_size()
            # if target and cfg.TGT_DATA.CAMERA_SAMPLE:
            #     data_sampler = RandomCameraIdentitySampler_DDP(dataset.train, IMS_PER_BATCH, NUM_INSTANCE)
            # else:
            #     data_sampler = RandomIdentitySampler_DDP(dataset.train, IMS_PER_BATCH, NUM_INSTANCE)

            # if not target:
            #     data_sampler = torch.utils.data.distributed.DistributedSampler(dataset.train)
            #     print('using random sampling for non target')
            # batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            # train_loader = torch.utils.data.DataLoader(
            #     train_set,
            #     num_workers=num_workers,
            #     batch_sampler=batch_sampler,
            #     collate_fn=train_collate_fn,
            #     pin_memory=cfg.DATALOADER.PIN_MEMORY,
            #     worker_init_fn=set_worker_sharing_strategy,
            #     # drop_last=True
            # )
            assert False
        else:
            if target:
                train_loader = DataLoader(
                    train_set, batch_size=IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
                    collate_fn=train_collate_fn, worker_init_fn=set_worker_sharing_strategy, # drop_last=True
                )
            else:
                train_loader = DataLoader(
                    train_set, batch_size=IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
                    collate_fn=train_collate_fn, worker_init_fn=set_worker_sharing_strategy, # drop_last=True
                )

    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn, worker_init_fn=set_worker_sharing_strategy, # drop_last=True
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    cf = val_collate_fn
    if target:
        val_set = train_set
        cf = train_collate_fn
        print(' the val set is the train set with val transform in target person dataset.')

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers, drop_last=False,
        collate_fn=cf, pin_memory=cfg.DATALOADER.PIN_MEMORY, worker_init_fn=set_worker_sharing_strategy
    )

    # train_loader_normal = DataLoader(
    #     train_set_normal, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
    #     collate_fn=val_collate_fn
    # )
    if target:
        print('data sampler: ', data_sampler)
    return train_loader, val_loader, len(dataset.query), num_classes, cam_num, view_num, data_sampler
