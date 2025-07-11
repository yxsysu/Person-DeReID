
import glob
import re

import os.path as osp

from .bases import BaseImageDataset

import torch
import numpy as np


DATASET_ROOT = '/data/yixing/msmt17_v1'
if not osp.exists(DATASET_ROOT):
    DATASET_ROOT = '/mnt/Datasets/reid_dataset/msmt17_v1/'
    print('loading msmt from juice')
# DATASET_ROOT = '/mnt/Datasets/reid_dataset/msmt17_v1/'
if not osp.exists(DATASET_ROOT):
    DATASET_ROOT = '/sdb/yixing/dataset/msmt17_v1/'
    print('loading msmt from local server')
if not osp.exists(DATASET_ROOT):
    assert False


class MSMT17(BaseImageDataset):
    """
    MSMT17

    Reference:
    Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR 2018.

    URL: http://www.pkuvmc.com/publications/msmt17.html

    Dataset statistics:
    # identities: 4101
    # images: 32621 (train) + 11659 (query) + 82161 (gallery)
    # cameras: 15
    """
    dataset_dir = 'MSMT17'

    def __init__(self, root='', verbose=True, pid_begin=0, fine_tune_num=10000, **kwargs):
        super(MSMT17, self).__init__()
        self.pid_begin = 0
        self.dataset_dir = DATASET_ROOT
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.test_dir = osp.join(self.dataset_dir, 'test')
        self.list_train_path = osp.join(self.dataset_dir, 'list_train.txt')
        self.list_val_path = osp.join(self.dataset_dir, 'list_val.txt')
        # self.list_query_path = osp.join(self.dataset_dir, 'list_query.txt')
        # self.list_gallery_path = osp.join(self.dataset_dir, 'list_gallery.txt')
        self.list_query_path = './data/msmt17/new_query.txt'
        self.list_gallery_path = './data/msmt17/new_gallery.txt'

        self._check_before_run()
        bar = 900
        train = self._process_dir(self.train_dir, self.list_train_path)
        val = self._process_dir(self.train_dir, self.list_val_path)
        train += val
        # train = [i for i in train if i[1] < bar]  # for pretrained
        print('bar: ', bar, '  bar+fine_tune_num: ', bar+fine_tune_num)
        train = [i for i in train if i[1] >= bar and i[1] < bar + fine_tune_num] # for fine tune
        # print('do not load val in msmt')
        query = self._process_dir(self.test_dir, self.list_query_path)
        gallery = self._process_dir(self.test_dir, self.list_gallery_path)

        if verbose:
            print("=> MSMT17 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)
    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))

    def _process_dir(self, dir_path, list_path):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()
        dataset = []
        pid_container = set()
        cam_container = set()
        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            pid = int(pid)  # no need to relabel
            camid = int(img_path.split('_')[2])
            img_path = osp.join(dir_path, img_path)
            dataset.append((img_path,  self.pid_begin +pid, camid-1, 1))
            pid_container.add(pid)
            cam_container.add(camid)
        print(cam_container, 'cam_container')
        # check if pid starts from 0 and increments with 1
        if 'train' in dir_path:
            for idx, pid in enumerate(pid_container):
                assert idx == pid, "See code comment for explanation"
        return dataset


class TargetMSMT17(BaseImageDataset):
    """
    MSMT17

    Reference:
    Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR 2018.

    URL: http://www.pkuvmc.com/publications/msmt17.html

    Dataset statistics:
    # identities: 4101
    # images: 32621 (train) + 11659 (query) + 82161 (gallery)
    # cameras: 15
    """
    dataset_dir = 'MSMT17'

    def __init__(self, root='', verbose=True, pid_begin=100, cam_aug=False, **kwargs):
        super(TargetMSMT17, self).__init__()
        self.pid_begin = pid_begin
        self.dataset_dir = DATASET_ROOT
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.test_dir = osp.join(self.dataset_dir, 'test')
        self.list_train_path = './data/msmt17/target_person.txt'

        self.list_query_path = './data/msmt17/target_query.txt'
        self.list_gallery_path = './data/msmt17/target_gallery.txt'

        self._check_before_run()
        all_target = []
        with open('./data/msmt17/target_person.txt', "r") as file:
            lines = file.readlines()
        for temp_1, temp_info in enumerate(lines):
            pp, pid = temp_info.split(' ')
            all_target.append(int(pid))
        all_target = torch.tensor(all_target).long().unique()
        all_target = all_target.numpy()
        print('self pid begin: ', pid_begin)
        self.all_target = all_target[0:self.pid_begin]

        if cam_aug:
            train = self._process_dir_train(self.test_dir, self.list_train_path)
            print('using cam aug')
        else:
            train = self._process_dir(self.test_dir, self.list_train_path)
        query = self._process_dir(self.test_dir, self.list_query_path)
        gallery = self._process_dir(self.test_dir, self.list_gallery_path)

        if verbose:
            print("=> TargetPerson in MSMT17 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))

    def _process_dir(self, dir_path, list_path):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()
        dataset = []
        pid_container = set()
        cam_container = set()
        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            pid = int(pid)  # no need to relabel
            if pid not in self.all_target:
                continue
            camid = int(img_path.split('_')[2])
            img_path = osp.join(dir_path, img_path)
            dataset.append((img_path, pid, camid - 1, 1))
            pid_container.add(pid)
            cam_container.add(camid)
        print(cam_container, 'cam_container')
        # check if pid starts from 0 and increments with 1

        return dataset

    def _process_dir_train(self, dir_path, list_path):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()
        dataset = []
        pid_container = set()
        cam_container = set()
        time_dir = {'morning': 0, 'afternoon': 1, 'noon': 2}

        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            pid = int(pid)  # no need to relabel
            # camid = int(img_path.split('_')[2])
            if pid not in self.all_target:
                continue
            camid = int(img_path.split('_')[2]) - 1
            # print(image_name, pid, camid)
            time = img_path.split('_')[3][4:]
            camid = camid * 3 + time_dir[time]
            if pid == -1:
                continue  # junk images are just ignored

            img_path = osp.join(dir_path, img_path)
            dataset.append((img_path, pid, camid, 1))
            pid_container.add(pid)
            cam_container.add(camid)
        print(cam_container, 'cam_container')
        # check if pid starts from 0 and increments with 1

        return dataset
