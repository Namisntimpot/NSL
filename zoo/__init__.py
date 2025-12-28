__all_inference_models__ = {}

def register_inference_model(name):
    def decorator(cls):
        __all_inference_models__[name] = cls
        return cls
    return decorator

def make_inference_model(
        name, dsname:str, inp_size:tuple, pretrained_model:str, 
        cfgs_path:set, ddp:bool, match_lr:bool, metrics_names:list
    ):
    module = __all_inference_models__[name]
    return module(dsname, inp_size, pretrained_model, cfgs_path, ddp, match_lr, metrics_names)

from . import inference_models