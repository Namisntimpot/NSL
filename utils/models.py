import torch
import torch.nn as nn

def remove_module_by_path(model:nn.Module, path:str):
    nodes = path.split(".")
    par = model
    child = model._modules[nodes[0]]
    for i in range(1, len(nodes)):
        par = child
        child = par._modules[nodes[i]]
    del par._modules[nodes[-1]]
    return model

def remove_module_after_layer(model:nn.Module, last_reserve_layer:str):
    reserve_flag = True
    layers_to_remove = set()
    for n, m in model.named_modules():
        if n == last_reserve_layer:
            reserve_flag = False
            continue
        if not reserve_flag:
            layers_to_remove.add(n)
    def __remove(m:nn.Module, path:str):
        should_remove = False
        if path in layers_to_remove:
            should_remove = True
        to_rm = []
        for n, c in m.named_children():
            rm = __remove(c, path + f".{n}")
            if rm:
                to_rm.append(n)
        for k in to_rm:
            del m._modules[k]
        return should_remove
    
    to_rm = []
    for n, c in model.named_children:
        if __remove(c, n):
            to_rm.append(n)
    for k in to_rm:
        del model._modules[k]
    return model

def parse_optimizer(model:nn.Module, **optim_cfgs):
    import torch.optim as optim
    ty = optim_cfgs['type']
    lr_cfgs:dict = optim_cfgs['lr']
    kwargs = {k:v for k, v in optim_cfgs.items() if k != 'type' and k!='lr'}
    base_lr = lr_cfgs['base_lr']
    kwargs['lr'] = base_lr
    params_group = {}   # submodule_name: [list of params]
    occupied = set()
    lr_cfgs_paths = sorted(list(lr_cfgs.keys()), key=lambda x: len(x.split(".")), reverse=True)
    for submodule_name in lr_cfgs_paths:
        lr = lr_cfgs[submodule_name]
        if submodule_name == 'base_lr':
            continue
        for param_name, param in model.named_parameters():
            if param_name.startswith(submodule_name) and not param_name in occupied:
                occupied.add(param_name)
                params_group[submodule_name] = params_group.get(submodule_name, []) + [param]
                if lr == 0:
                    param.requires_grad = False
    other_params = [p for n, p in model.named_parameters() if not n in occupied]
    group_args = [{'params': other_params}]
    for submodule_name, params in params_group.items():
        lr = lr_cfgs[submodule_name]
        if lr != 0:
            group_args.append(
                {
                    'params': params,
                    'lr': lr
                }
            )
    return getattr(optim, ty)(group_args, **kwargs)


def parse_scheduler(optimizer, num_steps_per_epoch, **sche_cfgs):
    from collections.abc import Iterable
    import torch.optim.lr_scheduler as lr_scheduler

    def parse_steps(step):
        num = step[0]
        kind:str = step[1]
        return int(num) if kind.startswith("s") else int(num * num_steps_per_epoch)
    def parse_single_scheduler(cfg):
        ty, kwargs = list(cfg.items())[0]
        kwargs = {k: parse_steps(v) if isinstance(v, Iterable) and len(v) == 2 else v 
                      for k, v in kwargs.items()}
        return getattr(lr_scheduler, ty)(optimizer, **kwargs)

    if 'milestones' in sche_cfgs:  # sequential scheduler
        schedulers = []
        for single_sche_cfg in sche_cfgs['scheduler']:
            schedulers.append(
                parse_single_scheduler(single_sche_cfg)
            )
        milestones = sche_cfgs['milestones']
        milestones = [parse_steps(v) for v in milestones]
        return lr_scheduler.SequentialLR(optimizer, schedulers, milestones)        
    else:
        if 'scheduler' in sche_cfgs:
            sche_cfgs = sche_cfgs['scheduler']
            assert len(sche_cfgs) == 1, "only 1 lr_scheduler will be used if SequentialLR is not specified"
            return parse_single_scheduler(sche_cfgs[0])
        else:
            assert len(sche_cfgs) == 1, "only 1 lr_scheduler will be used if SequentialLR is not specified"
            return parse_single_scheduler(sche_cfgs)