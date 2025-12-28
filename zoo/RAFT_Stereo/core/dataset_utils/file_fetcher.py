#######
# (split, scene_id, view_id, mode) -> datas  
#######
import os
from glob import glob
import numpy as np
import io
try:
    import imageio.v2 as imageio
except:
    import imageio
import pickle
import Imath
import OpenEXR

def load_exr(path, type = "RGB", bit16 = False):
    '''
    type: "RGB", "NORMAL", "Z"
    '''
    exr_file = OpenEXR.InputFile(path)

    # 获取图像的宽度和高度
    dw = exr_file.header()['dataWindow']
    # print(exr_file.header()['channels'])
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT if not bit16 else Imath.PixelType.HALF)
    if type == 'RGB':
        ch = ["R", "G", "B"]
    elif type == 'NORMAL':
        ch = ["X", "Y", "Z"]
    else:
        ch = ["V"]
    channels = exr_file.channels(ch, FLOAT)
    dtype = np.float16 if bit16 else np.float32
    # 将数据转换为NumPy数组
    if type == 'Z':
        d = np.frombuffer(channels[0], dtype=dtype).reshape(height, width).astype(np.float32)
        return d.copy()
    r = np.frombuffer(channels[0], dtype=dtype).reshape(height, width).astype(np.float32)
    g = np.frombuffer(channels[1], dtype=dtype).reshape(height, width).astype(np.float32)
    b = np.frombuffer(channels[2], dtype=dtype).reshape(height, width).astype(np.float32)

    # 如果需要，你可以将RGB值合并为一个图像
    image = np.stack([r, g, b], axis=-1)
    exr_file.close()
    return image.copy()   # 原buffer是只读的, 需要copy一下变成可写的.

class BaseFileFetcher:
    def __init__(self, split, data_root, decomp, cleaned = False):
        assert split in ['train', 'test', 'val', 'Ai2thor']
        self.split = split
        self.data_root = data_root
        self.decomp = decomp
        self.mode = 'decompositions' if self.decomp else 'images'
        self.split_root = None   # 往下就是scene_id / view id.
        self.cleaned = cleaned
        self._flatten_keys = []
        self._view_contents = []
        self.reserve_flags = {}
        
        self.__first_fetch_flatten_keys = True

        # about patterns.
        self.pattern_dir = os.path.join(self.data_root, 'patterns')
        pattern_split_file = os.path.join(self.pattern_dir, "split.json")
        import json
        if pattern_split_file.startswith("s3:"):
            import megfile
            with megfile.smart_open(pattern_split_file, 'r') as f:
                pattern_names = json.load(f)[self.split]  # with filetype suffix.
        else:
            with open(pattern_split_file, 'r') as f:
                pattern_names = json.load(f)[self.split]
        self.pattern_paths = {os.path.splitext(p)[0]:os.path.join(self.pattern_dir, p) for p in pattern_names if not 'white' in p}
        # {pattern names: pattern paths}
        self.white_pattern_path = os.path.join(self.pattern_dir, 'white.png')
        
        self.open_func = megfile.smart_open

    def fetch(self, key, parameters:bool, patternname:str = None, normal=True, materialtype=True):
        raise NotImplementedError
    
    def fetch_params_from_dict(self, params, vid:int):
        '''
        返回相机和投影仪内参, 左右相机外参.  
        '''
        ret = {}
        ret['L_intri'] = params['intrinsic']['L'].astype(np.float32)
        ret['R_intri'] = params['intrinsic']['R'].astype(np.float32)
        ret['P_intri'] = params['intrinsic']['Proj'].astype(np.float32)
        ret['L_extri'] = params['extrinsic'][vid]['L'].astype(np.float32)
        ret['R_extri'] = params['extrinsic'][vid]['R'].astype(np.float32)
        # projector在中间且L,R,P没有相对旋转.
        ret['P_extri'] = (ret['L_extri'] + ret['R_extri']) / 2
        return ret

    def fetch_patterns_names(self):
        return list(self.pattern_paths.keys())

    def fetch_patterns_paths(self):
        return list(self.pattern_paths.values())
    
    def fetch_white_paths(self):
        return self.white_pattern_path

    def fetch_pattern(self, patternname):
        if patternname == 'white':
            p = self.white_pattern_path
        else:
            p = self.pattern_paths[patternname]
        with self.open_func(p, 'rb') as f:
            content = f.read()
        with io.BytesIO(content) as stream:
            pat = imageio.imread(stream).astype(np.float32) / 255.
        if pat.ndim == 2:
            pat = pat[:,:,None].repeat(3, -1)
        elif pat.shape[-1] == 1:
            pat = pat.repeat(3,-1)
        return pat

    
    def _init_flatten_keys(self):
        raise NotImplementedError

    def _init_single_view_contents(self):
        raise NotImplementedError
    
    def _init_reserve_flags(self):
        raise NotImplementedError
    
    def flatten_keys(self):
        if self.__first_fetch_flatten_keys:
            self.__first_fetch_flatten_keys = False
            if self.cleaned:
                def fun(fk:str):
                    sid, vid = fk.split("/")
                    vid = f"{int(vid):03d}" if len(vid) != 3 else vid
                    return self.reserve_flags.get(f"{sid}/{vid}", True)
                self._flatten_keys = [f for f in self._flatten_keys if fun(f)]
        return self._flatten_keys

    def view_contents(self):
        return self._view_contents

class LocalFileFetcher(BaseFileFetcher):
    def __init__(self, split, data_root, decomp, cleaned = False):
        super().__init__(split, data_root, decomp, cleaned)
        self.split_root = os.path.join(data_root,split,self.mode)
        self._init_flatten_keys()
        self._init_single_view_contents()
        if self.cleaned:
            self._init_reserve_flags()

    def _init_flatten_keys(self):
        self._flatten_keys = []
        scene_ids = sorted(os.listdir(self.split_root))
        for sid in scene_ids:
            scene_id_dir = os.path.join(self.split_root, sid)
            filenames = sorted(os.listdir(scene_id_dir))
            filenames.remove("parameters.npz")
            filenames.remove("config.json")
            max_vid = int(filenames[-1].split("_")[0])
            for vid in range(max_vid + 1):
                self._flatten_keys.append(f"{sid}/{vid:03d}")

    def _init_single_view_contents(self):
        k = self._flatten_keys[0]
        sid, vid = k.split('/')
        scene_dir = os.path.join(self.split_root, sid)
        files = sorted(glob(os.path.join(scene_dir, f"{vid}*")))
        self._view_contents = ['_'.join(os.path.basename(f).split("_")[1:]) for f in files]

    def _init_reserve_flags(self):
        mode_root = os.path.dirname(self.split_root)
        reserve_flag_file = os.path.join(mode_root, 'reserve_flags.pkl')
        if not os.path.exists(reserve_flag_file):
            print(f"{reserve_flag_file} does not exist! no cleaned data!")
            return
        with open(reserve_flag_file, 'rb') as f:
            self.reserve_flags = pickle.load(f)
    
    def fetch(self, key:str, parameters:bool, patternname:str = None, normal=True, materialtype=True):
        sid, vid = key.split("/")
        ret = {}
        scene_dir = os.path.join(self.split_root, sid)
        for c in self._view_contents:
            if not self.decomp and 'Image' in c and not patternname == 'all':
                if (patternname is None or patternname == 'proj') and 'noproj' in c:
                    continue
                # patternname是具体的pattern名称.
                c_patname = c.split("_")[0]
                if patternname != c_patname:
                    continue
            # if patternname != None and "Image" in c and not patternname in c:
            #     continue
            if not normal and "Normal" in c:
                continue
            if not materialtype and "MaterialType" in c:
                continue
            fpath = os.path.join(scene_dir, f"{vid}_{c}")
            if fpath.endswith(".exr"):
                if 'Depth' in c or 'MaterialType' in c:
                    ret[c] = load_exr(fpath, 'Z', bit16=True)
                elif 'Normal' in c:
                    ret[c] = load_exr(fpath, 'NORMAL', bit16=True)
                else:
                    ret[c] = load_exr(fpath, 'RGB', bit16=True)
            else:
                ret[c] = imageio.imread(fpath).astype(np.float32) / 255
        if parameters:
            ret.update(self.fetch_params(key))
        return ret
        
    # @lru_cache(32)
    def fetch_params(self, key):
        sid, vid = key.split("/")
        vid = int(vid)
        fpath = os.path.join(self.split_root, sid, 'parameters.npz')
        with open(fpath, 'rb') as f:
            params = np.load(f, allow_pickle=True)['arr_0'].tolist()
        return self.fetch_params_from_dict(params, vid)
        
    
class OssFileFetcher(BaseFileFetcher):
    def __init__(self, split, data_root, decomp, cleaned = False):
        '''
        read from oss (s3)
        '''
        import megfile
        import pickle
        super().__init__(split, data_root, decomp, cleaned)
        if split == 'train':
            self.split_root = os.path.join(data_root, 'data', self.mode)
        else:
            self.split_root = os.path.join(data_root, self.split, 'data', self.mode)
        with megfile.smart_open(os.path.join(os.path.dirname(self.split_root), f"sid_{self.mode}.pickle"), 'rb') as f:
            self.sids = pickle.load(f)
        self.megfile = megfile
        self._init_flatten_keys()
        self._init_single_view_contents()
        if self.cleaned:
            self._init_reserve_flags()

    def _init_flatten_keys(self):
        self._flatten_keys = []
        scene_ids = sorted(self.sids)
        for sid in scene_ids:
            scene_id_dir = os.path.join(self.split_root, sid)
            filenames = sorted(self.megfile.smart_listdir(scene_id_dir))
            filenames.remove("parameters.npz")
            filenames.remove("config.json")
            max_vid = int(filenames[-1].split("_")[0])
            for vid in range(max_vid + 1):
                self._flatten_keys.append(f"{sid}/{vid:03d}")

    def _init_single_view_contents(self):
        k = self._flatten_keys[0]
        sid, vid = k.split("/")
        scene_dir = os.path.join(self.split_root, sid)
        files = sorted(self.megfile.smart_glob(os.path.join(scene_dir, f"{vid}*")))
        self._view_contents = ['_'.join(os.path.basename(f).split("_")[1:]) for f in files]

    def _init_reserve_flags(self):
        mode_root = os.path.dirname(self.split_root)
        reserve_flag_file = os.path.join(mode_root, 'reserve_flags.pkl')
        if not self.megfile.smart_exists(reserve_flag_file):
            print(f"{reserve_flag_file} does not exist! no cleaned data!")
            return
        with self.megfile.smart_open(reserve_flag_file, 'rb') as f:
            self.reserve_flags = pickle.load(f)
    
    def fetch(self, key, parameters:bool, patternname:str = None, normal=True, materialtype=True):
        sid, vid = key.split("/")
        ret = {}
        scene_dir = os.path.join(self.split_root, sid)
        for c in self._view_contents:
            if not self.decomp and 'Image' in c and not patternname == 'all':
                if (patternname is None or patternname == 'proj') and 'noproj' in c:
                    continue
                c_patname = c.split("_")[0]
                if patternname != c_patname:
                    continue
            # if patternname != None and "Image" in c and not patternname in c:
            #     continue
            if not normal and "Normal" in c:
                continue
            if not materialtype and "MaterialType" in c:
                continue
            fpath = os.path.join(scene_dir, f"{vid}_{c}")
            with self.megfile.smart_open(fpath, 'rb') as f:
                contents = f.read()
            with io.BytesIO(contents) as stream:
                if fpath.endswith(".exr"):
                    if 'Depth' in c or 'MaterialType' in c:
                        ret[c] = load_exr(stream, 'Z', bit16=True)
                    elif 'Normal' in c:
                        ret[c] = load_exr(stream, 'NORMAL', bit16=True)
                    else:
                        ret[c] = load_exr(stream, 'RGB', bit16=True)
                else:
                    ret[c] = imageio.imread(stream).astype(np.float32) / 255
        if parameters:
            ret.update(self.fetch_params(key))
        return ret

    # @lru_cache(32)
    def fetch_params(self, key):
        sid, vid = key.split("/")
        vid = int(vid)
        fpath = os.path.join(self.split_root, sid, 'parameters.npz')
        with self.megfile.smart_open(fpath, 'rb') as f:
            params = np.load(f, allow_pickle=True)['arr_0'].tolist()
        dict_param = self.fetch_params_from_dict(params, vid)
        del params
        return dict_param