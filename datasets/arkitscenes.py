import os
from glob import glob
from typing import Callable, Optional

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DistributedSampler, DataLoader

from utils.transforms import ColorConversionFunc
from utils.common import to_tensor

META_DATA_CSV_FILE = 'metadata.csv'
WIDE = 'wide'
HIGHRES_DEPTH = "highres_depth"
LOWRES_DEPTH = "lowres_depth"
LOW_RES = (192, 256)
HIGH_RES = (1440, 1920)
MILLIMETER_TO_METER = 1000

class ARKitScenesDataset(Dataset):
    def __init__(self, dataset_path:str, split='test', gray=False):
        assert split in ['train','test','val']
        super(ARKitScenesDataset, self).__init__()
        self.root = dataset_path
        self.split = split
        self.gray = gray
        split_dir_name = {'train':'Training', 'test':'Validation', 'val':'Validation'}[split]

        self.split_path = os.path.join(dataset_path, split_dir_name)
        
        self.samples = []  # videos_id, sample_id, sky_direction
        self.meta_data = pd.read_csv(os.path.join(self.root, META_DATA_CSV_FILE))
        self.meta_data = self.meta_data[self.meta_data['fold'] == split_dir_name]
        for video_id, sky_direction in zip(self.meta_data['video_id'], self.meta_data['sky_direction']):
            video_folder = os.path.join(self.split_path, str(video_id))
            color_files = glob(os.path.join(video_folder, WIDE, '*.png'))
            self.samples.extend([[str(video_id), str(os.path.basename(file)), sky_direction]
                                 for file in color_files])        

    @staticmethod
    def rotate_image(img, direction):
        if direction == 'Up':
            pass
        elif direction == 'Left':
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif direction == 'Right':
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif direction == 'Down':
            img = cv2.rotate(img, cv2.ROTATE_180)
        else:
            raise Exception(f'No such direction (={direction}) rotation')
        return img
    
    def load_sample_from_key(self, key):
        '''key: [video_id, img_id, sky_direciton]'''
        sample = {}
        video_id, img_id, sky_direction = key
        img_id = str(img_id)
        rgb_file = os.path.join(self.split_path, video_id, WIDE, img_id)
        high_res_depth_file = os.path.join(self.split_path, video_id, HIGHRES_DEPTH, img_id)
        low_res_depth_file = os.path.join(self.split_path, video_id, LOWRES_DEPTH, img_id)
        sample['key'] = f"{video_id}-{img_id}-{sky_direction}"
        sample['L_Image'] = ARKitScenesDataset.load_image(rgb_file, HIGH_RES, False, sky_direction, self.gray)
        sample['L_Depth'] = ARKitScenesDataset.load_image(high_res_depth_file, HIGH_RES, True, sky_direction, self.gray)
        sample['L_Ref_Depth'] = ARKitScenesDataset.load_image(low_res_depth_file, LOW_RES, True, sky_direction, self.gray)
        return sample


    @staticmethod
    def load_image(path, shape, is_depth, sky_direction, gray:bool=False):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img.shape[:2] != shape:
            img = cv2.resize(img, shape[::-1], interpolation=cv2.INTER_NEAREST if is_depth else cv2.INTER_LINEAR)
        # img = ARKitScenesDataset.rotate_image(img, sky_direction)
        if is_depth:
            img = img.astype(np.float32) / MILLIMETER_TO_METER  # (H,W)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.                 # (H,W,3)
            if gray:
                img = ColorConversionFunc.RGB2GRAY(img)
        return img

    def __getitem__(self, index: int):
        key = self.samples[index]
        sample = self.load_sample_from_key(key)
        sample['Pattern'] = 'noproj'
        sample['pattern_name'] = 'noproj'
        sample['L_Mask'] = sample['L_Depth'] < 10.
        return to_tensor(sample)

    def __len__(self) -> int:
        return len(self.samples)


def create_arkitscenes_dataloader(
        batch_size, num_workers, rank, world_size, 
        split:str, data_root:str, shuffle = True, gray = True, ddp:bool = True
    ):
    '''
    data_root: train/test/val所在的文件夹.  
    gray: 在加载时时否把图片转换为灰度图.  
    '''
    dataset = ARKitScenesDataset(data_root, split, gray)
    # print(len(dataset))
    if ddp:
        sampler = DistributedSampler(dataset, world_size, rank, shuffle, drop_last=(split == 'train'))
    else:
        sampler = None
    dataloader = DataLoader(
        dataset, batch_size, shuffle if sampler is None else None, sampler, pin_memory=False, 
        drop_last=True if split == 'train' else False,
        num_workers=num_workers, # collate_fn=merge_all
    )  # 提供了sampler的时候不应设置shuffle.
    return dataloader