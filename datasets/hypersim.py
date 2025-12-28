import os
import numpy as np
from torch.utils.data import Dataset, DistributedSampler, DataLoader
import h5py
from glob import glob
try:
    import imageio.v2 as imageio
except:
    import imageio


from utils.common import to_tensor
from utils.transforms import ColorConversionFunc


def hypersim_distance_to_depth(npyDistance):
    intWidth, intHeight, fltFocal = 1024, 768, 886.81

    npyImageplaneX = np.linspace((-0.5 * intWidth) + 0.5, (0.5 * intWidth) - 0.5, intWidth).reshape(
        1, intWidth).repeat(intHeight, 0).astype(np.float32)[:, :, None]
    npyImageplaneY = np.linspace((-0.5 * intHeight) + 0.5, (0.5 * intHeight) - 0.5,
                                 intHeight).reshape(intHeight, 1).repeat(intWidth, 1).astype(np.float32)[:, :, None]
    npyImageplaneZ = np.full([intHeight, intWidth, 1], fltFocal, np.float32)
    npyImageplane = np.concatenate(
        [npyImageplaneX, npyImageplaneY, npyImageplaneZ], 2)

    npyDepth = npyDistance / np.linalg.norm(npyImageplane, 2, 2) * fltFocal
    return npyDepth


class HypersimDataset(Dataset):
    def __init__(self, dataset_path:str, split='test', gray=True):
        super().__init__()
        assert split in ['train', 'val', 'test']
        self.split = split
        self.gray = gray
        self.dataset_path = dataset_path
        self.split_path = os.path.join(dataset_path,split)
        self.img_paths, self.dep_paths = self.load_file_keys(self.split_path)


    def load_file_keys(self, split_path):
        scene_dirs = os.listdir(split_path)
        img_paths = []  # jpg
        dep_paths = []  # distance in meter.
        for scene_name in scene_dirs:
            img_dirs = os.path.join(split_path, scene_name, 'images', '*final_preview*')
            img_dirs = sorted(glob(img_dirs), key=lambda x: os.path.basename(x).split("_")[2])
            depth_dirs = os.path.join(split_path, scene_name, 'images', '*geometry_hdf5*')
            depth_dirs = sorted(glob(depth_dirs), key=lambda x: os.path.basename(x).split("_")[2])
            
            n_sub_scenes = len(img_dirs)
            for i in range(n_sub_scenes):
                final_preview_dir = img_dirs[i]
                geometry_hdf5_dir = depth_dirs[i]
                tonemaps = sorted(glob(os.path.join(final_preview_dir, '*tonemap.jpg')),
                                  key=lambda x: os.path.basename(x).split(".")[1])
                distances= sorted(glob(os.path.join(geometry_hdf5_dir, "*depth_meters.hdf5")),
                                  key=lambda x: os.path.basename(x).split(".")[1])
                img_paths += tonemaps
                dep_paths += distances
        return img_paths, dep_paths
    
    def extrace_key_from_path(self, img_full_path:str):
        relpath = os.path.relpath(img_full_path, self.split_path) # "ai_xxx_xxx/images/scene_cam_xx_final_preview/frames.XXXX.tonemap.jpg"
        path_splits = relpath.split(os.sep)
        scene_key = path_splits[0]
        scene_cam_id = path_splits[2].split("_")[2]
        frame_id = path_splits[-1].split(".")[1]
        return "-".join([scene_key, scene_cam_id, frame_id])
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        dep_path = self.dep_paths[index]
        # load image
        img = imageio.imread(img_path).astype(np.float32) / 255.  # (h,w,3), (0, 1)
        if self.gray:
            img = ColorConversionFunc.RGB2GRAY(img)
        # load depth
        f = h5py.File(dep_path, 'r')
        dep = np.array(f['dataset']).astype(np.float32)
        f.close()
        dep = hypersim_distance_to_depth(dep)
        ret = {'L_Image': img, 'L_Depth': dep, 'key': self.extrace_key_from_path(img_path),
               'Pattern': 'noproj', 'pattern_name': 'noproj', 'L_Mask': dep < 20.}
        return to_tensor(ret)
    
def create_hypersim_dataloader(
        batch_size, num_workers, rank, world_size, 
        split:str, data_root:str, shuffle = True, gray = True, ddp:bool = True
    ):
    '''
    data_root: train/test/val所在的文件夹.  
    gray: 在加载时时否把图片转换为灰度图.  
    '''
    dataset = HypersimDataset(data_root, split, gray)
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


if __name__ == '__main__':
    # generate split.txt that dav2's official training script needed.
    dataset_path = "datasets/hypersim"
    split = 'test'
    output_path = "zoo/DAv2/metric_depth/dataset/splits/hypersim/test.txt"

    hypersim_dataset = HypersimDataset(dataset_path, split, False)
    img_paths = [os.path.abspath(p) for p in hypersim_dataset.img_paths]
    dep_paths = [os.path.abspath(p) for p in hypersim_dataset.dep_paths]

    with open(output_path, 'w') as f:
        for imgp, depp in zip(img_paths, dep_paths):
            f.write(f"{imgp} {depp}\n")