import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
try:
    import imageio.v2 as imageio
except:
    import imageio

from .file_fetcher import BaseFileFetcher, LocalFileFetcher, OssFileFetcher
from .utils import to_tensor, merge_all

class DeepslDataset(Dataset):
    def __init__(self, file_fetcher:BaseFileFetcher, gray:bool=True,
                 parameters=True, patternname=None, normal=True, materialtype=True):
        super().__init__()
        self.gray = gray
        self.file_fetcher = file_fetcher
        self.parameters = parameters
        self.patternname= patternname
        self.normal = normal
        self.materialtype = materialtype

    def __len__(self):
        return len(self.file_fetcher.flatten_keys())
    
    def __getitem__(self, index):
        key = self.file_fetcher.flatten_keys()[index]
        data = self.file_fetcher.fetch(key, self.parameters, self.patternname, self.normal, self.materialtype)
        data = to_tensor(data)
        if self.gray:
            self.convert_imgs_to_gray(data)
        data_newk ={}  # remove .xxx from data's keys
        for k, v in data.items():
            newk = k.split(".")[0]
            data_newk[newk] = v
        data_newk['key'] = key
        return data

    def convert_imgs_to_gray(self, data:dict):
        weights = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32)
        for k in data:
            if 'Image' in k:
                v = data[k]
                v = torch.sum(v*weights, dim=-1, keepdim=True).expand_as(v)
                data[k] = v
        return data

    def get_patterns_names(self):
        return self.file_fetcher.fetch_patterns_names()

    def get_pattern(self, pattern_name):
        return self.file_fetcher.fetch_pattern(pattern_name)
    
class SimplifiedStereoDataset(DeepslDataset):
    '''
    no normal, no materialtype, return {'L_Image':,'R_Image':,'L_Depth':, 'R_Depth':, 'L_intri'...}  
    only support 'images' mode, do not support 'decomposition' mode.  

    If the patternname parameter is not specified, images from each viewpoint under multiple patterns 
    will be traversed separately, 
    resulting in num_patterns * num_views sets of stereo data.
    '''
    def __init__(self, file_fetcher, gray=True, parameters=True, patternname=None, normal=False, materialtype=False):
        super().__init__(file_fetcher, gray, parameters, patternname, normal, materialtype)
        if self.patternname is None or self.patternname == 'proj':
            self.num_patterns = len(self.file_fetcher.fetch_patterns_names())
            self.list_patterns = self.file_fetcher.fetch_patterns_names()
        elif self.patternname == 'all':
            self.num_patterns = len(self.file_fetcher.fetch_patterns_names()) + 1
            self.list_patterns = self.file_fetcher.fetch_patterns_names() + ['noproj']
        else:
            self.num_patterns = 1
            self.list_patterns = [self.patternname]

    def __len__(self):
        l = super().__len__()
        return l * self.num_patterns

    def __getitem__(self, index):
        # if self.patternname is not None and not self.patternname in ['all', 'proj']:
        #     pattern_to_fetch = self.patternname
        #     flatten_key_to_fetch = self.file_fetcher.flatten_keys()[index]
        #     data = super().__getitem__(index)
        # else:
        pattern_to_fetch = self.list_patterns[index % self.num_patterns]
        flatten_key_to_fetch = self.file_fetcher.flatten_keys()[index // self.num_patterns]
        data = self.file_fetcher.fetch(
            flatten_key_to_fetch, self.parameters, pattern_to_fetch, self.normal, self.materialtype)
        data = to_tensor(data)
        if self.gray:
            data = self.convert_imgs_to_gray(data)
        for k in list(data.keys()):
            newk = k.split(".")[0] # if self.patternname is None else k 去掉文件后缀...
            prefix = newk[:2]
            if prefix != 'L_' and prefix != 'R_' and prefix != 'P_':
                newk = "_".join(newk.split("_")[1:])
            v = data.pop(k)
            data[newk] = v
        data['key'] = flatten_key_to_fetch
        data['pattern'] = pattern_to_fetch
        return data
        # # rename keys.
        # ret = {}
        # for k, v in data.items():
        #     newk = k.split(".")[0] # if self.patternname is None else k 去掉文件后缀...
        #     prefix = newk[:2]
        #     if prefix != 'L_' and prefix != 'R_' and prefix != 'P_':
        #         newk = "_".join(newk.split("_")[1:])
        #     ret[newk] = v
        # ret['key'] = flatten_key_to_fetch
        # ret['pattern'] = pattern_to_fetch
        # del data
        # return ret
    

class SimplifiedStereoDatasetWithPattern(SimplifiedStereoDataset):
    def __init__(self, file_fetcher, gray=True, parameters=True, patternname=None, normal=False, materialtype=False):
        assert patternname != 'all' and patternname != 'noproj', "This dataloader won't load noproj images."
        super().__init__(file_fetcher, gray, parameters, patternname, normal, materialtype)
        # load patterns.
        self.patterns_images = {
            k: torch.from_numpy(self.file_fetcher.fetch_pattern(k)) for k in self.list_patterns
        }

    def __getitem__(self, index):
        sample:dict = super().__getitem__(index)
        pattern_name = sample.pop("pattern")
        sample['Pattern'] = self.patterns_images[pattern_name]
        sample['pattern_name'] = pattern_name
        return sample


def _create_dataloader_with_type(dataset_to_create, batch_size, num_workers, rank, world_size,
                      split:str, data_root:str, cleaned, decomp:bool = False, ftype = 'local', 
                      shuffle = True,
                      gray = True,
                      parameters=True, patternname=None, normal=True, materialtype=True,
                      ddp:bool = True
                      ):
    '''
    data_root: The root directory containing the data folder, where the 'train', 'test', and 'val' subfolders are located.
    ftype: Data source type, either 'local' or 'oss'.
    parameters: Whether to load intrinsic and extrinsic camera parameters.
    patternname: Specifies which pattern's images to load.
    - 'all': Load images rendered from all patterns including 'noproj'.
    - 'proj' (or None): Load only images rendered from all projected patterns, excluding 'noproj'.
    - Otherwise: Load images rendered from a specific pattern indicated by the given name.
    cleaned: Whether to use the cleaned version of the dataset.
    decomp: Whether to use decomposition mode; currently fixed to False.
    gray: Whether to convert images to grayscale upon loading.
    normal, materialtype: Whether to load normal maps and material type annotations, respectively.
    '''
    if ftype == 'local':
        filefetcher = LocalFileFetcher(split, data_root, decomp, cleaned)
    elif ftype == 'oss':
        filefetcher = OssFileFetcher(split, data_root, decomp, cleaned)
    else:
        raise ValueError(f"Unknown file fetcher type: {ftype}")
    dataset = dataset_to_create(filefetcher, gray, parameters, patternname, normal, materialtype)
    # print(len(dataset))
    if ddp:
        sampler = DistributedSampler(dataset, world_size, rank, shuffle, drop_last=True if split == 'train' else False)
    else:
        sampler = None
    dataloader = DataLoader(
        dataset, batch_size, shuffle if sampler is None else None, sampler, pin_memory=False, 
        drop_last=True if split == 'train' else False,
        num_workers=num_workers, # collate_fn=merge_all
    )  # 提供了sampler的时候不应设置shuffle.
    return dataloader

def create_raw_dataloader(batch_size, num_workers, rank, world_size,
                      split:str, data_root:str, cleaned:bool, decomp:bool = False, ftype = 'local', 
                      shuffle = True,
                      gray = True,
                      parameters=True, patternname=None, normal=True, materialtype=True,
                      ddp:bool = True
                      ):
    return _create_dataloader_with_type(
                DeepslDataset, batch_size, num_workers, rank, world_size,
                split, data_root, cleaned, decomp, ftype, shuffle, gray,
                parameters, patternname, normal, materialtype, ddp
            )

def create_simplified_stereo_dataloader(
        batch_size, num_workers, rank, world_size,
        split:str, data_root:str, cleaned:bool, decomp:bool=False,ftype = 'local',
        shuffle = True, gray=True, parameters = True, patternname = None, ddp:bool=True, material_type:bool=False):
    return _create_dataloader_with_type(
        SimplifiedStereoDataset, batch_size, num_workers, rank, world_size,
        split, data_root, cleaned, decomp, ftype, shuffle, gray, parameters,
        patternname, False, material_type, ddp
    )

def create_simplified_stereo_dataloader_with_pattern(
        batch_size, num_workers, rank, world_size,
        split:str, data_root:str, cleaned:bool, decomp:bool=False,ftype = 'local',
        shuffle = True, gray=True, parameters = True, patternname = None, ddp:bool=True, material_type:bool=False):
    return _create_dataloader_with_type(
        SimplifiedStereoDatasetWithPattern, 
        batch_size, num_workers, rank, world_size,
        split, data_root, cleaned, decomp, ftype, shuffle, gray, parameters,
        patternname, False, material_type, ddp
    )

if __name__ == "__main__":
    from tqdm import tqdm
    import psutil
    import time
    import objgraph

    import megfile
    open_func = megfile.smart_open

    def show_dict(d):
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                to_show = str(v.shape) + f", min={v.min():.2f}, max={v.max():.2f}"
            else:
                to_show = v
            print(k,":",to_show)
    def check_dict(d1:dict, d2:dict):
        def check_leaf(v1, v2):
            if isinstance(v1, torch.Tensor):
                assert torch.all(v1 == v2)
            elif isinstance(v1, list):
                assert len(v1) == len(v2)
                for i in range(len(v1)):
                    assert v1[i] == v2[i]
            assert False, "unknown leaf type"
        if isinstance(d1,  dict):
            for v1, v2 in zip(d1.values(), d2.values()):
                check_dict(v1, v2)
        else:
            check_leaf(d1, d2)
    def test_shape(d):
        for k, v in d.items():
            if not isinstance(v, torch.Tensor):
                continue
            if not hasattr(test_shape, 'registered'):
                setattr(test_shape, 'registered', {})
            if not k in test_shape.registered:
                test_shape.registered[k] = v.shape[1:]  # 避开batch.
            else:
                if v.shape[1:] != test_shape.registered[k]:
                    raise ValueError("Unmatched shape!")
    def save_images(sample:dict, save_dir:str, id:int):
        import matplotlib.pyplot as plt
        for k, v in sample.items():
            if 'image' in k.lower() or 'pattern' == k.lower():
                # 保存为图片
                for idx, im in enumerate(torch.unbind(v, dim=0)):
                    p = os.path.join(save_dir, f"{id}_{idx}_{k}.png")
                    im = (np.clip(im.numpy(), 0, 1)*255).astype(np.uint8)
                    with open_func(p, 'wb') as f:
                        imageio.imwrite(f, im, format='png')
            elif 'depth' in k.lower():
                for idx, depth in enumerate(torch.unbind(v, dim=0)):
                    p = os.path.join(save_dir, f"{id}_{idx}_{k}.png")
                    depth = torch.clip(depth, 0, 5)
                    depth = depth.squeeze().numpy()
                    # print(f"{idx}_{k}: min={depth.min()}, max={depth.max()}")
                    plt.imshow(depth, vmin=depth.min(), vmax=depth.max(), cmap='jet')
                    plt.colorbar()
                    with open_func(p, 'wb') as f:
                        plt.savefig(f)
                    plt.close()

    torch.manual_seed(42)
    dataloader_local = create_simplified_stereo_dataloader_with_pattern(
        8,0,0,1,'train','s3://ljh-deepsl-data/',
        True, False, 'local', False, True, patternname=None, ddp=True
    )
    # dataloader_oss = create_dataloader(16, 4, 0, 1, 'train', 's3://ljh-deepsl-data/',
    #                                False, 'oss', False, True)
    i = 0
    pbar = tqdm(dataloader_local)
    with open("obj000.txt", 'w') as f:
        objgraph.show_growth(limit=1000, file=f)
    for data_local in pbar:
        selfmem_info = psutil.Process().memory_info()
        rss = selfmem_info.rss / 1024.**2
        vms = selfmem_info.vms / 1024.**2
        data_size = selfmem_info.data / 1024.**2
        pbar.set_description(f"_rss:{rss:.2f}MB_vms:{vms:.2f}MB_data:{data_size:.2f}")
        i += 1
        # time.sleep(0.5)
        if i % 100 == 0:
            with open(f"obj{i:03d}.txt", 'w') as f:
                objgraph.show_growth(limit=1000, file=f)
        if i>= 500:
            break