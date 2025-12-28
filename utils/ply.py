import os
import numpy as np
import struct
from .common import FileIOHelper

def export_ply(f, points, colors = None):
    '''
    points: (..., 3), np.ndarray (metric)  
    colors: (..., 3), np.ndarray, color value should be within (0, 255)  
    f: str or io stream
    '''
    has_color = colors is not None
    points = np.array(points) if not isinstance(points, np.ndarray) else points
    points = points.reshape(-1, 3)
    if has_color:
        colors = np.array(colors) if not isinstance(colors, np.ndarray) else colors
        colors = colors.reshape(-1, 3).astype(np.uint8)
        assert colors.shape == points.shape
    n_points = points.shape[0]
    
    should_close = False
    if isinstance(f, str):
        should_close = True
        iohelper = FileIOHelper()
        f = iohelper.open(f, 'wb')

    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n_points}",
        "property float x",
        "property float y",
        "property float z",
        "end_header"
    ]
    if has_color:
        header.insert(-1, "property uchar red")
        header.insert(-1, "property uchar green")
        header.insert(-1, "property uchar blue")

    header = '\n'.join(header) + '\n'
    f.write(header.encode("ascii"))
    for i in range(n_points):
        pos = points[i]
        if has_color:
            c = colors[i]
            f.write(struct.pack('<3f3B', pos[0], pos[1], pos[2], c[0], c[1], c[2]))
        else:
            f.write(struct.pack('<3f', pos[0], pos[1], pos[2]))

    if should_close:
        f.close()