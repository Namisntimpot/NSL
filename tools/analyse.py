import os
import pickle
import numpy as np
import argparse
from tqdm import tqdm
from easydict import EasyDict
import json

from utils.common import FileIOHelper
from zoo.data_info import MAX_DEPTH_INFO

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exproot", type=str, default='s3://ljh-deepsl-data/exp/eval')
    parser.add_argument("--expname", type=str)
    parser.add_argument("--point-cloud", action='store_true')
    parser.add_argument("--data-root", type=str, default='s3://ljh-deepsl-data/')
    parser.add_argument("--n-worst", type=int, default=10)
    parser.add_argument("--n-best", type=int, default=1)
    parser.add_argument("--outlier-thresh", type=int, default=100)
    parser.add_argument("--per-pat-metrics", action='store_true')
    # parser.add_argument("--remove-outlier", action='store_true')
    args = parser.parse_args()
    return args

def get_exp_args(expdir):
    import json
    iohelper = FileIOHelper()
    with iohelper.open(os.path.join(expdir, 'args.json'), 'r') as f:
        expargs = json.load(f)
    return EasyDict(expargs)

def get_score_file(expdir):
    def is_pkl_wanted(pklname):
        if not os.path.basename(pklname)[0].isdigit() and not os.path.basename(pklname).startswith('per_'):
            return True
        return not '_per_' in pklname

    iohelper = FileIOHelper()
    pkls = iohelper.glob(os.path.join(expdir, "*.pkl"))
    pkls = [pkl for pkl in pkls if is_pkl_wanted(os.path.basename(pkl))]
    with iohelper.open(pkls[0], 'rb') as f:
        score_file = pickle.load(f)
    for i in range(1, len(pkls)):
        with iohelper.open(pkls[i], 'rb') as f:
            score_file.update(pickle.load(f))
    return score_file

def mean_score_file(score_dict:dict, too_big_thresh:int = 100):
    '''
    score_dict: {
        'key': {'metric_name': metric} or None.
    }
    '''
    total = {}
    num = {}
    outlier = {}
    for k, v in score_dict.items():
        if v is None:
            continue
        if 'mae' in v and v['mae'] > too_big_thresh:   # 大的离谱的异常点.
            for mname in v:
                outlier[mname] = outlier.get("mname", 0) + 1
            continue
        for mname, val in v.items():
            if np.isnan(val):  
                outlier[mname] = outlier.get("mname", 0) + 1
                continue
            total[mname] = total.get(mname, 0) + val
            num[mname] = num.get(mname, 0) + 1
    total = {mname: v / num[mname] for mname, v in total.items()}
    print("num_outliers:", outlier)
    return total, outlier

def fix_per_mat_metrics(expdir):
    iohelper = FileIOHelper()
    if len(iohelper.glob(os.path.join(expdir, "1_*.pkl"))) == 0:  # single gpu.
        return
    per_mat_metric_pkl = os.path.join(expdir, 'per_mat_per_img_metrics.pkl')
    per_mat_metric_json= os.path.join(expdir, 'per_mat_avg_metrics.json')
    if len(iohelper.glob(per_mat_metric_pkl)) != 0:
        with iohelper.open(per_mat_metric_pkl, 'rb') as f:
            d = pickle.load(f)
        per_mat_metrics = {}
        for mat_name, key_to_metric_dict in d.items():
            total = {}
            cnt = {}
            for key, metric_name_to_val in key_to_metric_dict.items():
                for metric_name, val in metric_name_to_val.items():
                    if val is None or np.isnan(val) or np.isinf(val):
                        continue
                    total[metric_name] = total.get(metric_name, 0) + val
                    cnt[metric_name] = cnt.get(metric_name, 0) + 1
            per_mat_metrics[mat_name] = {k: total[k] / cnt[k] for k in cnt}
        with iohelper.open(per_mat_metric_json, 'w') as f:
            json.dump(per_mat_metrics, f, indent=4)


def per_pattern_score(score_dict:dict):
    per_pat_sc = {} # per_pat_sc[pattern_name][key][metric_name] = float
    avg_per_pat_sc = {}  # avg_per_pat_sc[pattern_name][metric_name] = float
    total_per_pat_sc = {}# total_per_pat_sc[pattern_name][metric_name] = total_number.
    for key, metrics in score_dict.items():
        sid, vid, pattern, lr = key.split("/")
        if metrics is None:
            continue
        for metric_name, val in metrics.items():
            if pattern not in per_pat_sc:
                per_pat_sc[pattern] = {}
            if key not in per_pat_sc[pattern]:
                per_pat_sc[pattern][key] = {}
            per_pat_sc[pattern][key][metric_name] = val
    
    for pattern, key_2_metrics in per_pat_sc.items():
        if pattern not in avg_per_pat_sc:
            avg_per_pat_sc[pattern] = {}
            total_per_pat_sc[pattern] = {}
        for key, metrics in key_2_metrics.items():
            for metric_name, val in metrics.items():
                if val is None or np.isnan(val) or np.isinf(val):
                    continue
                avg_per_pat_sc[pattern][metric_name] = avg_per_pat_sc[pattern].get(metric_name, 0) + val
                total_per_pat_sc[pattern][metric_name]=total_per_pat_sc[pattern].get(metric_name,0)+ 1
    for pattern, metrics in avg_per_pat_sc.items():
        for metric_name in metrics:
            metrics[metric_name] /= total_per_pat_sc[pattern][metric_name]
    return per_pat_sc, avg_per_pat_sc


def export_pointclouds(expargs, data_root:str, samples_dir, ply_dir, n_worst=10, n_best=10):
    from utils.visualize import vis_comparable_pointcloud
    from deepsl_data.dataloader.file_fetcher import LocalFileFetcher
    file_fetcher = LocalFileFetcher(
        'test', data_root, False, True
    )
    max_depth = MAX_DEPTH_INFO[expargs.get("dataset", 'deepsl')]
    if expargs.get("dataset", "deepsl") != 'deepsl':
        raise NotImplementedError

    iohelper = FileIOHelper()
    pklfiles = iohelper.listdir(samples_dir)
    pklfiles = [fname for fname in pklfiles if fname.endswith(".pkl") and 'L_' in fname]
    pklfiles = sorted(pklfiles, key= lambda x: float(x.split("-")[5].split("_")[1]), reverse=True)  # 降序

    def do_export_pointclouds(pklname):
        sid, vid, pattern = pklname.split("-")[:3]
        pkl_path = os.path.join(samples_dir, pklname)
        ply_path = os.path.join(ply_dir, os.path.splitext(pklname)[0] + ".ply")
        with iohelper.open(pkl_path, 'rb') as f:
            pd_depth = pickle.load(f).squeeze()  # H*W
        key = f"{int(sid):05d}/{int(vid)}"
        gt_data = file_fetcher.fetch(key, True, pattern, False, False)
        gt_depth = None
        K = None
        rgb = None
        for k, v in gt_data.items():
            if 'L_' in k:
                if 'intri' in k:
                    K = v
                elif 'Depth' in k:
                    gt_depth = v.squeeze()
                elif 'Image' in k:
                    rgb = v.squeeze() * 255.
        vis_comparable_pointcloud(
            K, gt_depth, pd_depth, ply_path, rgb, max_depth
        )
    _cnt = 0
    pbar = tqdm(None, total=n_worst, desc="worst point clouds")
    exported = set()
    for i in range(len(pklfiles)):
        pklname = pklfiles[i]
        key = "_".join(pklname.split("-")[:2])
        if key in exported:
            continue
        do_export_pointclouds(pklname)
        exported.add(key)
        _cnt += 1
        pbar.update(1)
        if _cnt >= n_worst:
            break

    _cnt = 0
    pbar = tqdm(None, total=n_best, desc="best point clouds")
    exported = set()
    for i in range(len(pklfiles)):
        pklname = pklfiles[-1 - i]
        key = "_".join(pklname.split("-")[:2])
        if key in exported:
            continue
        do_export_pointclouds(pklname)
        exported.add(key)
        _cnt += 1
        pbar.update(1)
        if _cnt >= n_best:
            break

def print_dict(d:dict, level=0):
    prefix = "  " * level
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"{prefix}{k}: ")
            print_dict(v, level+1)
        else:
            print(f"{prefix}{k}: {v}")

def main():
    args = parse_args()
    expdir = os.path.join(args.exproot, args.expname)
    setattr(args, 'expdir', expdir)
    iohelper = FileIOHelper()
    exp_args = get_exp_args(expdir)
    score_file = get_score_file(expdir)
    mean_scores, num_outliers = mean_score_file(score_file, args.outlier_thresh)
    if len(num_outliers) != 0:  # anomaly exists.
        mean_scores_str = "_".join([f"{k}-{v:.2f}" for k, v in mean_scores.items()])
        with iohelper.open(os.path.join(expdir, f"{mean_scores_str}.txt"), 'w') as f:
            pass
    print_dict(mean_scores)

    fix_per_mat_metrics(expdir)

    if args.per_pat_metrics:
        per_pat_sc, avg_per_pat_sc = per_pattern_score(score_file)
        with iohelper.open(os.path.join(expdir, "per_pat_per_image_metrics.pkl"), 'wb') as f:
            pickle.dump(per_pat_sc, f)
        with iohelper.open(os.path.join(expdir, "per_pat_avg_metrics.json"), "w") as f:
            json.dump(avg_per_pat_sc, f, indent=4)
        print_dict(avg_per_pat_sc)        


    if args.point_cloud:
        sample_dir = os.path.join(expdir, 'samples')
        ply_dir = os.path.join(expdir, 'pointclouds')
        export_pointclouds(exp_args, args.data_root, sample_dir, ply_dir, args.n_worst, args.n_best)

if __name__ == '__main__':
    main()