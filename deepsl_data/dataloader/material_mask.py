import numpy as np
import torch

MATERIAL_TYPE_NUMBER = {
    'Diffuse.General': 0, 'Diffuse.Classic':1, 'Diffuse.Plastic':2, 'Diffuse.Clay': 3, 'Diffuse.Fabric': 4, 'Diffuse.Wood':5,
    'Diffuse.Leather': 6, 'Diffuse.Paper': 7, 'Diffuse.Rubber': 8, 
    'Specular.Metal': 9, 'Specular.Porcelain': 10, 'Specular.Plasticsp': 11, 'Specular.Paintsp': 12, 'Specular.Mirror': 13, 'Specular.Screen': 14,
    'Transmission.Transparent':15, 'Transmission.Crystal': 16,
    'Transmission.Film': 17, 'Transmission.Subsurface': 18, 'Transmission.Light': 19, 'Transmission.Glass': 20,
}

__material_type_number_range = {}

def __fetch_type_number_range(level):
    '''
    Can ensure that the numbering of each category does not overlap with each other
    '''
    if level in __material_type_number_range:
        return __material_type_number_range[level]
    type_num_range = {}
    for k, v in MATERIAL_TYPE_NUMBER.items():
        cat = '.'.join(k.split(".")[:level])
        if cat not in type_num_range:
            type_num_range[cat] = [v, v]
        else:
            cur_range = type_num_range[cat]
            if v < cur_range[0]:
                cur_range[0] = v
            if v > cur_range[1]:
                cur_range[1] = v
    __material_type_number_range[level] = type_num_range
    return type_num_range        

def material_masks(material_type, level = 1):
    '''
    tomask: tensors to mask.  
    material_type: pixel-level material types  
    level: 1 or 2 for now. The level of detail in material classification
    '''
    assert level in [1, 2]
    type_range = __fetch_type_number_range(level)
    cat_masks = {}
    for cat, rg in type_range.items():
        cat_masks[cat] = (material_type >= rg[0]) & (material_type <= rg[1])
    return cat_masks

def get_material_catagories(level):
    assert level in [1, 2]
    cats = set()
    for k in MATERIAL_TYPE_NUMBER:
        cat = ".".join(k.split(".")[:level])
        cats.add(cat)
    return cats