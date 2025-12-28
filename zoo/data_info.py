MEAN_STD_INFO = {
    'none':{
        'mean': [0,0,0],
        'std': [1,1,1,],
    },
    'vga':{
        'mean': [0,0,0],
        'std': [1,1,1],
    },  # for fps testing
    'sceneflow':{
        'mean': [ 0.485, 0.456, 0.406 ],
        'std':  [ 0.229, 0.224, 0.225 ],
    },
    'deepsl-gray':{
        'mean': None,
        'std': None,
    },
    'deepsl-color':{
        'mean': None,
        'std': None,
    },
    'dav2-fix':{
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    },
    'deepsl':{
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    },
    'nyuv2': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    },
    'arkitscenes': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    },
    'hypersim': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    },
    'dreds': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    },
}

MAX_DEPTH_INFO = {
    'deepsl': 10,
    'hypersim': 20,
    'nyuv2': 20,
    'arkitscenes':10,
    'dreds': 10,
}

RESO_INFO = {
    'default': (1280, 720),
    'dav2-fix': (1280, 720),
    'deepsl': (1280, 720),
    'nyuv2': (640, 480),
    'arkitscenes': (1920, 1440),
    'hypersim': (1024, 768),
    'dreds': (640, 360),
    'vga': (640, 480),
}