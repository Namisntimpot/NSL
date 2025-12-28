import os
import argparse
from easydict import EasyDict
from pathlib import Path
from datetime import datetime

class ParserWrapper:
    ARGS_GROUP_DATALOADER = ("dataloader", 0)
    ARGS_GROUP_INF = ('inf_model', 1)
    ARGS_GROUP_TRAIN = ('train', 1)
    ARGS_GROUP_EXP = ('exp', 100)
    ACT_GROUPS = {}

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.add_argument = self.parser.add_argument

    def add(self, group:tuple):
        ParserWrapper.ACT_GROUPS[group[0]] = group[1]
        func = getattr(ParserWrapper, f'parser_add_{group[0]}')
        self.parser = func(self.parser)

    def parse_args(self):
        args = self.parser.parse_args()
        args = EasyDict(vars(args))
        postprocess_func_list = [(getattr(ParserWrapper, f'postprocess_{k}'), v) for k, v in ParserWrapper.ACT_GROUPS.items()]
        postprocess_func_list = sorted(postprocess_func_list, key=lambda x: x[1])
        for func, _ in postprocess_func_list:
            args = func(args)
        return args

    @staticmethod
    def parser_add_dataloader(parser:argparse.ArgumentParser):
        '''
        add args about dataloader.
        '''
        parser.add_argument("--dataset", type=str, help='dataset to evaluate on', default='deepsl')
        parser.add_argument("--batch-size", type=int, help="batch size")
        parser.add_argument("--num-workers", type=int, help="Number of workers to load data. recommend num-cores/num-gpus")
        parser.add_argument("--split", type=str, help="train/val/test")
        parser.add_argument("--data-root", type=str, help="path to the deepsl-data dataset.", default='data/')
        parser.add_argument("--no-clean", action='store_true', help='do not use cleaned data.')
        parser.add_argument("--decomp", action='store_true', help="decomposition mode.")
        parser.add_argument("--ftype", type=str, help='where the data are stored. local/oss. default: local', default='local', choices=['local', 'oss'])
        parser.add_argument("--no-gray", action='store_true', help='do not convert to GRAY in advance')
        parser.add_argument("--no-param", action='store_true', help='do not load camera parameters')
        parser.add_argument("--pattern", type=str, help="only load images rendered with the specified pattern.", default=None)
        parser.add_argument("--load-pattern", action='store_true')

        return parser

    @staticmethod
    def postprocess_dataloader(args:EasyDict):
        args['shuffle'] = args['split'] == 'train'
        return args

    @staticmethod
    def parser_add_inf_model(parser:argparse.ArgumentParser):
        parser.add_argument("--name", type=str, help="Name of the model to evaluate.")
        parser.add_argument("--dsname", type=str, help="Name of the dataset on which the model was trained.", default='none')
        parser.add_argument("--inp-size-w", type=int, help="Input resolution: w. default:1280", default=1280)
        parser.add_argument("--inp-size-h", type=int, help="Input resolution: h. default:720",  default=720)
        parser.add_argument("--pretrained", type=str, help="Path to the pretrained parameter file.")
        parser.add_argument("--cfgs-path", type=str, help="Path to the cfgs file. May be none.", default=None)
        parser.add_argument("--ddp", action='store_true', help='whether to use ddp mode')
        parser.add_argument("--no-match-lr", action='store_true', help="Matching happened between image and pattern rather than between left and right images")
        # parser.add_argument("--metrics", type=str, help="Comma seperated metrics names.", default="epe,d1,bad_1,bad_2,bad_3")
        parser.add_argument("--metrics", type=str, help="Comma seperated metrics names. default:'rmse,delta_05,absrel'", default="rmse,delta_05,absrel")
        parser.add_argument("--per-mat-metrics", action='store_true')
        return parser

    @staticmethod
    def postprocess_inf_model(args:EasyDict):
        args['inp_size'] = (args['inp_size_w'], args['inp_size_h'])
        args['metrics'] = args['metrics'].split(',')
        args['load_material_type'] = args['per_mat_metrics']
        if args['per_mat_metrics']:
            assert args['dataset'] == 'deepsl', "only deepsl dataset support per_mat_metrics"
        return args

    @staticmethod
    def parser_add_exp(parser:argparse.ArgumentParser):
        parser.add_argument("--exproot", type=str, default=".", help="where the 'exp' dir will be")
        parser.add_argument("--expname", type=str, help="Custom exp name.", default=None)
        parser.add_argument("--seed", type=int, default=42)
        return parser

    @staticmethod
    def postprocess_exp(args:EasyDict):
        if 'inf_model' in ParserWrapper.ACT_GROUPS:
            behavior_level = 'eval'
            # data_level = args.get('split', 'unkown')
            ckpt_fname = Path(args.pretrained).stem
            exp_tag_list = [args.name, ckpt_fname, f"gray_{not args.no_gray}", f"pat_{args.pattern}"]
            if args.expname:
                exp_tag_list = [args.expname] + exp_tag_list
            exp_level = "-".join(exp_tag_list)
        elif 'train' in ParserWrapper.ACT_GROUPS:
            behavior_level = 'train'
            # TODO: training exp's name.
            now = datetime.now()
            formatted_timestamp = now.strftime("%m%d_%H_%M")
            if args.expname:
                exp_level = "-".join([args.expname, formatted_timestamp])
            else:
                exp_level = "-".join([formatted_timestamp])
        args['expdir'] = os.path.join(args.exproot, "exp", behavior_level, exp_level)

        if hasattr(args, 'ckptdir') and args.ckptdir is None:
            args.ckptdir = os.path.join(args.expdir, 'ckpt')
        if hasattr(args, 'logdir') and args.logdir is not None:
            args.logdir = os.path.join(args.logdir, os.path.basename(args.expdir))
        else:
            args.logdir = os.path.join(args.expdir, 'tensorboard')
        # if not os.path.exists(args['expdir']):
        #     os.makedirs(args['expdir'])
        
        return args
    
    @staticmethod
    def parser_add_train(parser:argparse.ArgumentParser):
        parser.add_argument("--cfgs-path", type=str, help="config file path.")
        parser.add_argument("--resume", type=str, default=None, help="for continuing training.")
        parser.add_argument("--logdir", type=str, default=None, help='local log dir for tensorboard.')
        parser.add_argument("--ckptdir", type=str, default=None, help="specify ckpt dir. default: None(created automatically)")
        parser.add_argument("--no-ddp", action='store_true', help='do not use ddp training')
        return parser

    @staticmethod
    def postprocess_train(args:EasyDict):
        exproot = args.exproot
        if exproot.startswith("s3:") and args.logdir is None:
            raise ValueError("Tensorboard only support local log dir, please specify an extra local logdir "
                            "when exproot indicates a remote path.")

        from utils.common import load_config
        cfgs_path = args.cfgs_path
        cfg = load_config(cfgs_path)
        
        if args.resume is not None:
            from utils.common import FileIOHelper
            iohelper = FileIOHelper()
            if iohelper.isfile(args.resume):
                cfg['Model']['ckpt_path'] = args.resume
            
        args['cfgs'] = cfg

        # override expname if args.expname is not specified.
        if args.expname is None:
            args.expname = cfg.get("Expname", None)

        return args