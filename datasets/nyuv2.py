import os
import numpy as np
from torch.utils.data import Dataset, DistributedSampler, DataLoader
import h5py

from utils.common import to_tensor
from utils.transforms import ColorConversionFunc

class NYUv2Dataset(Dataset):
    def __init__(self, dataset_path:str, split='test', gray=True):
        super().__init__()
        assert split in ['train', 'val', 'test']
        self.split = split
        self.gray = gray
        self.dataset_path = dataset_path
        self.split_path = os.path.join(dataset_path, split)
        self.file_paths = self.load_file_paths(self.split_path)

    def load_file_paths(self, split_path):
        scene_dirs = os.listdir(split_path)
        file_paths = []
        for scene_name in scene_dirs:
            this_scene_dir = os.path.join(split_path, scene_name)
            file_names = sorted(os.listdir(this_scene_dir))
            file_paths += [
                os.path.join(this_scene_dir, fname) for fname in file_names
            ]
        return file_paths
    
    def extrace_key_from_path(self, img_full_path:str):
        relpath = os.path.relpath(img_full_path, self.split_path) # "ai_xxx_xxx/images/scene_cam_xx_final_preview/frames.XXXX.tonemap.jpg"
        path_splits = relpath.split(os.sep)
        scene_key = path_splits[0]
        frame_id = os.path.splitext(path_splits[-1])[0]
        return "-".join([scene_key, frame_id])
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, index):
        h5_path = self.file_paths[index]
        f = h5py.File(h5_path, 'r')
        rgb = np.array(f['rgb']).transpose(1,2,0).astype(np.float32) / 255.  # 变回channel_last.
        dep = np.array(f['depth'])
        f.close()
        if self.gray:
            rgb = ColorConversionFunc.RGB2GRAY(rgb)

        ret = {'L_Image': rgb, 'L_Depth': dep, 'key': self.extrace_key_from_path(h5_path),
               'Pattern': 'noproj', 'pattern_name': 'noproj', 'L_Mask': dep < 20.}
        return to_tensor(ret)
    

def create_nyuv2_dataloader(
        batch_size, num_workers, rank, world_size, 
        split:str, data_root:str, shuffle = True, gray = True, ddp:bool = True
    ):
    '''
    data_root: train/test/val所在的文件夹.  
    gray: 在加载时时否把图片转换为灰度图.  
    '''
    dataset = NYUv2Dataset(data_root, split, gray)
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