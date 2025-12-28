import numpy as np

class ColorConversionFunc:
    @staticmethod
    def get_conversion_func(from_format:str, to_format:str):
        if from_format.upper() == to_format.upper():
            return lambda x: x
        return getattr(ColorConversionFunc, f"{from_format.upper()}2{to_format.upper()}")

    @staticmethod
    def HSV2RGB(color:np.ndarray):
        color[color == 1.0] = 0
        h, s, v = color[..., 0], color[..., 1], color[..., 2]

        # 将 H 转换为 [0, 6) 区间
        h = h * 6.0
        i = np.int32(h)  # 获取整数部分
        f = h - i   # 获取小数部分

        # 计算 p, q, t
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))

        r = np.select([i==0, i==1, i==2, i==3, i==4, i==5], 
                      [v, q, p, p, t, v])
        g = np.select([i==0, i==1, i==2, i==3, i==4, i==5], 
                      [t, v, v, q, p, p])
        b = np.select([i==0, i==1, i==2, i==3, i==4, i==5], 
                      [p, p, t, v, v, q])
        return np.stack([r, g, b], axis=-1)
    
    @staticmethod
    def RGB2HSV(color:np.ndarray):
        # 确保输入在 [0, 1] 范围内
        r, g, b = np.clip(color[..., 0], 0, 1), np.clip(color[..., 1], 0, 1), np.clip(color[..., 2], 0, 1)
        
        # 计算最大值和最小值
        c_max = max(r, g, b)  # 明度 V
        c_min = min(r, g, b)
        delta = c_max - c_min  # 色彩区间差值
        h = np.select([delta == 0, c_max == r, c_max == g, c_max == b],
                      [0, ((g - b) / delta) % 6, (b - r) / delta + 2, (r - g) / delta + 4])
        h = h / 6    # 将色相归一化到 [0, 1] 范围
        h = np.select([h < 0, h>=0], [h+1, h])
        # 计算饱和度 S
        s = np.select([c_max == 0, c_max != 0], [0, delta / c_max])
        s = 0 if c_max == 0 else delta / c_max
        # 计算明度 V
        v = c_max
        return np.stack([h, s, v], axis=-1)
    
    @staticmethod
    def SRGB2LINEAR(color:np.ndarray):
        return np.where(color <= 0.0404482362771082, color / 12.92, np.power((color + 0.055)/1.055, 2.4))

    @staticmethod
    def LINEAR2SRGB(img:np.ndarray):
        return np.where(img <= 0.0031308, 
                        img * 12.92, 
                        1.055 * np.power(img, 1/2.4) - 0.055)
    
    @staticmethod
    def RGB2GRAY(rgb:np.ndarray):
        '''rgb: (...,h,w,3), within range (0, 1)!'''
        weights = np.array([0.299, 0.587, 0.114], dtype=rgb.dtype)
        gray = np.sum(rgb * weights, axis=-1, keepdims=True)
        return np.repeat(gray, 3, axis=-1)

if __name__ == "__main__":
    rgb = np.array([0.0, 0.3, 0.1])
    hsv = ColorConversionFunc.RGB2HSV(rgb)
    print(hsv)
    print(ColorConversionFunc.HSV2RGB(hsv))