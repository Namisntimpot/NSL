import os
import numpy as np
import torch
import torch.distributed as dist
from easydict import EasyDict
from pathlib import Path
import pickle
import torch.multiprocessing.spawn
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from zoo import make_inference_model
from zoo.data_info import MAX_DEPTH_INFO
from deepsl_data.dataloader.dataloader import create_simplified_stereo_dataloader, create_simplified_stereo_dataloader_with_pattern
from zoo.inference_models import BaseInferenceModel
from utils.common import FileIOHelper, get_free_port, general_gather, manual_seed, check_make_dirs, copy_file
from utils.args import ParserWrapper
from utils.visualize import vis_batch

def parse_args():
    parser = ParserWrapper()
    parser.add(ParserWrapper.ARGS_GROUP_DATALOADER)
    parser.add(ParserWrapper.ARGS_GROUP_INF)
    parser.add(ParserWrapper.ARGS_GROUP_EXP)
    parser.add_argument("--sample-rate", type=float, help="Probability of saving a depth map. Default: 0.", default=0.0)
    parser.add_argument("--start-cnt", type=int, default=-1)  # 只是应对stereoanything的神奇bug...
    parser.add_argument("--end-cnt", type=int, default=-1)
    parser.add_argument('--local-rank', default=0, type=int)

    args = parser.parse_args()
    return args

def update_per_image_metrics(per_image_metrics:dict, metrics:dict, keys:list, patterns:list):
    for idx, k in enumerate(keys):
        for item_name, mtrs in metrics.items():
            identifier = f"{k}/{patterns[idx]}/{item_name}"
            if mtrs is None:
                per_image_metrics[identifier] = None
            else:
                per_image_metrics[identifier] = {mtr: val[idx].item() for mtr, val in mtrs.items()}
    return per_image_metrics

def update_per_mat_metrics(per_mat_per_img_metrics:dict, per_mat_metrics:dict, keys:list, patterns:list):
    '''per_mat_per_img_metrics[materialname][identifier][metric_name]=float.
       per_mat_metrics[item_name][materialname][metric_name]=list[float]'''
    material_name_list = None
    for item_name, material_name_2_metrics in per_mat_metrics.items():
        if material_name_2_metrics is not None:
            material_name_list = list(material_name_2_metrics.keys())
            break
    for material_name in material_name_list:
        tmp = {item_name: material_name_2_metrics[material_name] if material_name_2_metrics is not None else None \
               for item_name, material_name_2_metrics in per_mat_metrics.items()}
        per_mat_per_img_metrics[material_name] = update_per_image_metrics(
            per_mat_per_img_metrics.get(material_name, {}), tmp, keys, patterns
        )
    return per_mat_per_img_metrics


def update_avg_metrics(avg_metrics:dict, total:dict, metrics:dict):
    for mtrs in metrics.values():
        if mtrs is None:
            continue
        for mname, batched_val in mtrs.items():
            n = batched_val.shape[0]
            if torch.any(batched_val.isnan()):
                continue
            avg_metrics[mname] = avg_metrics.get(mname, 0) + torch.sum(batched_val).item()
            if total is not None:
                total[mname] = total.get(mname, 0) + n
    return avg_metrics

def update_per_mat_avg_metrics(per_mat_avg_metrics:dict, per_mat_metrics:dict):
    '''per_mat_avg_metrics[material_name][metric_name] = sum(float)
       per_mat_metrics[item_name][material_name][metric_name] = list[float]'''
    material_name_list = None
    for item_name, material_name_2_metrics in per_mat_metrics.items():
        if material_name_2_metrics is not None:
            material_name_list = list(material_name_2_metrics.keys())
            break
    for material_name in material_name_list:
        tmp = {item_name: material_name_2_metrics[material_name] if material_name_2_metrics is not None else None \
               for item_name, material_name_2_metrics in per_mat_metrics.items()}
        per_mat_avg_metrics[material_name] = update_avg_metrics(
            per_mat_avg_metrics.get(material_name, {}), None, tmp
        )
    return per_mat_avg_metrics


def merge_gathered_avg_metrics(gather_avg:dict, gather_total:dict, whole_total:float):
    return {mname: sum([val.item() * gather_total[idx] for idx, val in enumerate(vals)]) / whole_total\
                  for mname, vals in gather_avg.items()}


def average_per_image_metrics(per_image_metrics:dict):
    '''per_image_metrics: [key][metric_name] = float'''
    cnt = {}
    total = {}
    for key, metric_name_to_val in per_image_metrics.items():
        if metric_name_to_val is None:
            continue
        for metric_name, val in metric_name_to_val.items():
            if val is None or np.isnan(val) or np.isinf(val):
                continue
            cnt[metric_name] = cnt.get(metric_name, 0) + 1
            total[metric_name]=total.get(metric_name,0)+val
    return {k: total[k] / cnt[k] for k in cnt}


def do_evaluate(rank, args:EasyDict, model:BaseInferenceModel, dataloader:DataLoader):
    # import pynvml
    # pynvml.nvmlInit()
    # handle = pynvml.nvmlDeviceGetHandleByIndex(rank)
    ds_max_depth = MAX_DEPTH_INFO[args.dataset]
    
    per_image_metrics = {}
    avg_metrics = {k: 0 for k in args.metrics}
    total = {k: 0 for k in args.metrics}

    # per_material_metrics
    if args.per_mat_metrics:
        from deepsl_data.dataloader.material_mask import get_material_catagories
        mat_cats = get_material_catagories(level=1)
        per_mat_per_img_metrics = {k:{} for k in mat_cats}
        per_mat_avg_metrics = {k:{} for k in mat_cats}

    io_helper = FileIOHelper()
    if args.get('sample_rate', 0) > 0:
        sample_dir = os.path.join(args.expdir, "samples")

    start_cnt = args.get('start_cnt', 0)
    end_cnt = args.get('end_cnt', len(dataloader))
    start_cnt = 0 if start_cnt == -1 else start_cnt
    end_cnt = len(dataloader) if end_cnt == -1 else end_cnt
    cnt = 0
    if args.get("sample_rate", 0) > 0:
        sample_interval = np.ceil(len(dataloader) * args.get("sample_rate"))
        sample_interval = np.floor(len(dataloader) / sample_interval)

    if rank == 0:
        pbar = tqdm(total=len(dataloader))
    for data in dataloader:
        if cnt < start_cnt:
            cnt += 1
            if rank == 0:
                pbar.update(1)
            continue

        pred_depth, metrics, data = model.inference(data, per_mat_metrics=args.per_mat_metrics)
        if args.per_mat_metrics:
            metrics, per_mat_metrics = metrics

        keys = data['key']  # key: sid/vid
        patterns = data['pattern'] if not args.load_pattern else data['pattern_name']
        update_per_image_metrics(per_image_metrics, metrics, keys, patterns)
        update_avg_metrics(avg_metrics, total, metrics)
        if args.per_mat_metrics:
            update_per_mat_metrics(per_mat_per_img_metrics, per_mat_metrics, keys, patterns)
            update_per_mat_avg_metrics(per_mat_avg_metrics, per_mat_metrics)

        cnt += 1
        # save sample.
        if args.get("sample_rate", 0) > 0 and cnt % sample_interval == 0:
        # if True:  # for DEBUG
            fname_list = []
            for idx, k in enumerate(keys):  # iterate batch.
                lr_name = []
                for item_name, depth in pred_depth.items():
                    if depth is None:
                        continue
                    identifier = f"{k}/{patterns[idx]}/{item_name}"
                    m = per_image_metrics[identifier]
                    depth_to_save = depth[idx].cpu().numpy()
                    fpath = f"{identifier.replace('/', '-')}-{'-'.join([f'{k}_{v:.2f}' for k, v in m.items()])}.pkl"
                    fpath_full = os.path.join(sample_dir, fpath)
                    with io_helper.open(fpath_full, 'wb') as f:
                        pickle.dump(depth_to_save, f)
                    lr_name.append(fpath[:-4])
                    # fpath = fpath.replace(".pkl", ".png")
                    # plt.imshow(depth_to_save, cmap='jet', vmin=depth_to_save.min(), vmax=depth_to_save.max())
                    # plt.colorbar()
                    # with io_helper.open(fpath, 'wb') as f:
                    #     plt.savefig(f)
                    # plt.close()
                fname_list.append("-".join(lr_name) + ".png")
            vis_batch(data, pred_depth['L_Image'], pred_depth.get('R_Image', None), 0, args.expdir, ds_max_depth, rank, fname_list=fname_list, share_depth_range=True)
        
        if rank == 0:
            pbar.update(1)
            # 检查一下显存泄露问题.
            # reserved = torch.cuda.memory_reserved(device=torch.device(f"cuda:{rank}"))
            # allocated = torch.cuda.memory_allocated(device=torch.device(f"cuda:{rank}"))
            # memoryinfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            # total_mem = memoryinfo.total
            # allocated = memoryinfo.used
            # free = memoryinfo.free
            # pbar.set_description(f"{cnt}: total:{total_mem/1e6:.2f}, cuda memory allocated: {allocated / 1e6:.2f}, free: {free / 1e6:.2f}")
        if cnt >= end_cnt:
            break
        # clean...
        del pred_depth, metrics
        torch.cuda.empty_cache()

    avg_metrics = {mname: avg_metrics[mname] / total[mname] for mname in avg_metrics}
    if args.per_mat_metrics:
        per_mat_avg_metrics = {
            material_name: {
                metric_name: material_name_2_metric[metric_name] / total[metric_name] \
                for metric_name in material_name_2_metric
            } for material_name, material_name_2_metric in per_mat_avg_metrics.items()
        }    
        return total[list(total.keys())[0]], avg_metrics, per_image_metrics, per_mat_avg_metrics, per_mat_per_img_metrics
    return total[list(total.keys())[0]], avg_metrics, per_image_metrics


def evaluate(rank, world_size, port, args:EasyDict, model:BaseInferenceModel, dist_inited:bool = False):
    logger = logging.getLogger(__name__)
    io_helper = FileIOHelper()
    # setup ddp
    if not dist_inited:  # 已经ddp初始化了
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = f"{port}"
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # setup model for ddp
    model.setup_ddp(rank, world_size)
    # print(f"rank {rank}: type(model)={type(model.model.module)}, model.device = {model.model.module.device}")
    # create dataloader
    if args.dataset == 'deepsl':
        logger.info("create deepsl dataset")
        if not args.load_pattern:
            dataloader = create_simplified_stereo_dataloader(
                args.batch_size, args.num_workers, rank, world_size,
                args.split, args.data_root, (not args.no_clean), args.decomp, args.ftype, 
                args.shuffle, (not args.no_gray), (not args.no_param), args.pattern, 
                material_type=args.load_material_type
            )
        else:
            dataloader = create_simplified_stereo_dataloader_with_pattern(
                args.batch_size, args.num_workers, rank, world_size, args.split,
                args.data_root, (not args.no_clean), args.decomp, args.ftype,
                args.shuffle, (not args.no_gray), (not args.no_param), args.pattern, 
                material_type=args.load_material_type
            )
    elif args.dataset == 'hypersim':
        logger.info("create hypersim dataset")
        from datasets.hypersim import create_hypersim_dataloader
        dataloader = create_hypersim_dataloader(
            args.batch_size, args.num_workers, rank, world_size, args.split, 
            args.data_root, args.shuffle, (not args.no_gray)
        )
    elif args.dataset == 'nyuv2':
        logger.info("create nyuv2 dataset")
        from datasets.nyuv2 import create_nyuv2_dataloader
        dataloader = create_nyuv2_dataloader(
            args.batch_size, args.num_workers, rank, world_size, args.split, 
            args.data_root, args.shuffle, (not args.no_gray)
        )
    elif args.dataset == 'arkitscenes':
        logger.info("create arkitscenes dataset")
        from datasets.arkitscenes import create_arkitscenes_dataloader
        dataloader = create_arkitscenes_dataloader(
            args.batch_size, args.num_workers, rank, world_size, args.split, 
            args.data_root, args.shuffle, (not args.no_gray)
        )
    elif args.dataset == 'dreds':
        logger.info("create DREDS dataset")
        from datasets.dreds import create_dreds_dataloader
        dataloader = create_dreds_dataloader(
            args.batch_size, args.num_workers, rank, world_size, args.split, 
            args.data_root, args.shuffle, normal=False, material_type=False
        )
    else:
        raise NotImplementedError
    
    if args.per_mat_metrics and args.dataset != "deepsl":
        logger.warning("per-material-metrics only supports deepsl dataset.")
        args.per_mat_metrics = False

    test_interval = args.start_cnt != -1 or args.end_cnt != -1
    start_cnt = 0 if args.start_cnt != -1 else args.start_cnt
    end_cnt = len(dataloader) if args.end_cnt != -1 else args.end_cnt


    total, avg, per_image_metrics, *extra = do_evaluate(
        rank, args, model, dataloader
    )

    # debug: 改用从文件加载出来后的per_images字典
    # gather_total = general_gather(rank, world_size, total)
    # gather_avg = general_gather(rank, world_size, avg)

    if args.per_mat_metrics:
        per_mat_avg_metrics, per_mat_per_image_metrics = extra
    #     gather_per_mat_avg = general_gather(rank, world_size, per_mat_avg_metrics)
    #     pass

    # if rank == 0:
    #     gather_total = [t.item() for t in gather_total]
    #     whole_total = sum(gather_total)
    #     whole_avg = merge_gathered_avg_metrics(gather_avg, gather_total, whole_total)
    #     if args.per_mat_metrics:
    #         whole_per_mat_avg = {
    #             material_name: merge_gathered_avg_metrics(metric_name_2_vals, gather_total, whole_total) \
    #             for material_name, metric_name_2_vals in gather_per_mat_avg.items()
    #         }

    if not test_interval:
        rank_metrics_file_name = f"{rank}_per_img_metrics.pkl"
        rank_per_mat_metrics_file_name = f"{rank}_per_mat_per_img_metrics.pkl"
    else:
        rank_metrics_file_name = f"{rank}_per_img_metrics_{start_cnt}_{end_cnt}.pkl"
        rank_per_mat_metrics_file_name = f"{rank}_per_mat_per_img_metrics_{start_cnt}_{end_cnt}.pkl"
    rank_metrics_file_path = os.path.join(args.expdir, rank_metrics_file_name)
    with io_helper.open(rank_metrics_file_path, 'wb') as f:
        pickle.dump(per_image_metrics, f)
    if args.per_mat_metrics:
        rank_per_mat_metrics_file_path = os.path.join(args.expdir, rank_per_mat_metrics_file_name)
        with io_helper.open(rank_per_mat_metrics_file_path, 'wb') as f:
            pickle.dump(per_mat_per_image_metrics, f)
    
    dist.barrier()

    if rank == 0:
        for i in range(1, world_size):
            if not test_interval:
                rank_metrics_file_path = os.path.join(args.expdir, f"{i}_per_img_metrics.pkl")
                rank_per_mat_metrics_file_path = os.path.join(args.expdir, f"{i}_per_mat_per_img_metrics.pkl")
            else:
                rank_metrics_file_path = os.path.join(args.expdir, f"{i}_per_img_metrics_{start_cnt}_{end_cnt}.pkl")
                rank_per_mat_metrics_file_path = os.path.join(args.expdir, f"{i}_per_mat_per_img_metrics_{start_cnt}_{end_cnt}.pkl")
            with io_helper.open(rank_metrics_file_path, 'rb') as f:
                per_image_metrics_other = pickle.load(f)
                per_image_metrics.update(per_image_metrics_other)
            if args.per_mat_metrics:
                with io_helper.open(rank_per_mat_metrics_file_path, 'rb') as f:
                    per_mat_per_image_metrics_other = pickle.load(f)
                    for material_name, identifier_2_metric_dict in per_mat_per_image_metrics.items():
                        identifier_2_metric_dict.update(per_mat_per_image_metrics_other[material_name])

        # compute average metrics.
        whole_avg = average_per_image_metrics(per_image_metrics)
        if args.per_mat_metrics:
            whole_per_mat_avg = {
                mat_name: average_per_image_metrics(matname_to_per_img_metrics) \
                for mat_name, matname_to_per_img_metrics in per_mat_per_image_metrics.items()
            }

        rank0_fname = "-".join([f"{mname}_{val:.2f}" for mname, val in whole_avg.items()]) + ".pkl"
        if test_interval:
            rank0_fname = f"{start_cnt}_{end_cnt}-" + rank0_fname
        with io_helper.open(os.path.join(args.expdir, rank0_fname), 'wb') as f:
            pickle.dump(per_image_metrics, f)

        if args.per_mat_metrics:
            rank0_per_mat_fname = "per_mat_per_img_metrics.pkl"
            if test_interval:
                rank0_per_mat_fname = f"{start_cnt}_{end_cnt}-" + rank0_per_mat_fname
            with io_helper.open(os.path.join(args.expdir, rank0_per_mat_fname), 'wb') as f:
                pickle.dump(per_mat_per_image_metrics, f)

        whole_avg_json_fname = "avg_metrics.json"
        with io_helper.open(os.path.join(args.expdir, whole_avg_json_fname), 'w') as f:
            json.dump(whole_avg, f, indent=4)
        if args.per_mat_metrics:
            whole_per_mat_avg_json_fname = "per_mat_avg_metrics.json"
            with io_helper.open(os.path.join(args.expdir, whole_per_mat_avg_json_fname), 'w') as f:
                json.dump(whole_per_mat_avg, f, indent=4)

        logger.info("Evaluate done.")
        for mname, avg in whole_avg.items():
            logger.info(f"{mname}: {avg}")

if __name__ == "__main__":
    import json
    import multiprocessing
    multiprocessing.set_start_method('spawn')

    ddp_inited = False
    if 'RANK' in os.environ:
        from utils.dist import setup_distributed_full
        rank, world_size = setup_distributed_full()
        ddp_inited = True

    interaction = (not ddp_inited) or (rank == 0)
    
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    if not ddp_inited:
        assert args.num_workers == 0, "Something I don't know is wrong when num_workers > 0, please set num_workers=0"

    manual_seed(args.seed)
    dirs_to_make = [args.expdir]
    if args.sample_rate > 0:
        dirs_to_make.append(os.path.join(args.expdir, "samples"))
    if interaction:
        check_make_dirs(*dirs_to_make)
        if args.cfgs_path is not None and os.path.exists(args.cfgs_path):
            copy_file(args.cfgs_path, os.path.join(args.expdir, os.path.basename(args.cfgs_path)))

    port = get_free_port()
    n_gpus = torch.cuda.device_count()
    if interaction:
        logger.info(args)
        logger.info(f"num-gpus used: {n_gpus}")
        logger.info(f"free port: {port}")
        io_helper = FileIOHelper()
        with io_helper.open(os.path.join(args.expdir, 'args.json'), 'w') as f:
            json.dump(args, f, indent=4)

    model = make_inference_model(
        args.name, args.dsname, args.inp_size, args.pretrained, args.cfgs_path,
        args.ddp, not args.no_match_lr, args.metrics
    )
    if interaction:
        logger.info(f"Model {args.name} created.")

    if interaction and (args.start_cnt != -1 or args.end_cnt != -1):
        logger.info(f"Only test: {args.start_cnt} - {args.end_cnt}")
    
    if not ddp_inited:
        torch.multiprocessing.spawn(
            evaluate, nprocs=n_gpus, args=(n_gpus, port, args, model)
        )
    else:
        evaluate(rank, world_size, port, args, model, True)