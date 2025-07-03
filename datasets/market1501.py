# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re
import torch
import os.path as osp

from .bases import BaseImageDataset
from collections import defaultdict
import pickle


def get_camid(img_path):
    img_path = img_path.split('/')[1]
    camid = int(img_path.split('_')[1][1]) - 1
    return camid


DATASET_ROOT = '/mnt/Datasets/reid_dataset/market1501'
if not osp.exists(DATASET_ROOT):
    DATASET_ROOT = '/sda/yixing/Market-1501'
    print('loading market1501 from local')
assert osp.exists(DATASET_ROOT)

class Market1501(BaseImageDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background) 750
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'market1501'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(Market1501, self).__init__()
        self.dataset_dir = DATASET_ROOT
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = self.dataset_dir
        self.gallery_dir = self.dataset_dir

        self._check_before_run()

        self.list_query_path = './data/market1501/new_query.txt'
        self.list_gallery_path = './data/market1501/new_gallery.txt'
        bar = 600

        train = self._process_dir(self.train_dir, relabel=True)
        # train = [i for i in train if i[1] < bar] # for pretrianed
        train = [i for i in train if i[1] >= bar]  # for fine tune

        query = self._process_dir_path(self.query_dir, self.list_query_path, relabel=False)
        gallery = self._process_dir_path(self.gallery_dir, self.list_gallery_path, relabel=False)

        if verbose:
            print("=> Market1501 loaded")
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
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in sorted(img_paths):
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]

            dataset.append((img_path, pid, camid, 1))
        return dataset

    def _process_dir_path(self, dir_path, list_path, relabel=False):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()
        dataset = []
        pid_container = set()
        cam_container = set()

        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            pid = int(pid)  # no need to relabel
            camid = get_camid(img_path)
            img_path = osp.join(dir_path, img_path)

            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 0 <= camid <= 5

            dataset.append((img_path, pid, camid, 1))
            pid_container.add(pid)
            cam_container.add(camid)
        print(cam_container, 'cam_container')
        # check if pid starts from 0 and increments with 1
        if 'train' in dir_path:
            for idx, pid in enumerate(pid_container):
                assert idx == pid, "See code comment for explanation"
        return dataset


class TargetMarket1501(BaseImageDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background) 750
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'market1501'

    def __init__(self, root='', verbose=True, pid_begin=100, cam_aug=False, **kwargs):
        super(TargetMarket1501, self).__init__()
        self.dataset_dir = DATASET_ROOT
        self.train_dir = self.dataset_dir
        self.query_dir = self.dataset_dir
        self.gallery_dir = self.dataset_dir

        self._check_before_run()
        all_target = []
        with open('./data/market1501/target_person.txt', "r") as file:
            lines = file.readlines()
        for temp_1, temp_info in enumerate(lines):
            pp, pid = temp_info.split(' ')
            all_target.append(int(pid))
        all_target = torch.tensor(all_target).long().unique()
        all_target = all_target.numpy()
        self.pid_begin = pid_begin
        self.all_target = all_target[0:self.pid_begin]

        self.list_train_path = './data/market1501/target_person.txt'
        self.list_query_path = './data/market1501/target_query.txt'
        self.list_gallery_path = './data/market1501/target_gallery.txt'

        train = self._process_dir_path(self.train_dir, self.list_train_path)
        query = self._process_dir_path(self.query_dir, self.list_query_path)
        gallery = self._process_dir_path(self.gallery_dir, self.list_gallery_path)

        if verbose:
            print("=> Target Market1501 loaded")
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
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in sorted(img_paths):
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]

            dataset.append((img_path, pid, camid, 1))
        raise NotImplementedError
        return dataset

    def _process_dir_path(self, dir_path, list_path):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()
        dataset = []
        pid_container = set()
        cam_container = set()

        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            pid = int(pid)  # no need to relabel
            camid = get_camid(img_path)
            img_path = osp.join(dir_path, img_path)

            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 0 <= camid <= 5
            if pid not in self.all_target:
                continue

            dataset.append((img_path, pid, camid, 1))
            pid_container.add(pid)
            cam_container.add(camid)
        print(cam_container, 'cam_container')
        # check if pid starts from 0 and increments with 1
        # if 'train' in dir_path:
        #     for idx, pid in enumerate(pid_container):
        #         assert idx == pid, "See code comment for explanation"
        return dataset
