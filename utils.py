from collections import defaultdict
import numpy as np
from PIL import ImageFilter, Image


def blurr_mask(mask, thr=32, gauss=5):
    blurred_mask = mask.filter(ImageFilter.GaussianBlur(gauss))
    blurred_mask = blurred_mask.point( lambda p: 255 if p > thr else 0 )
    return blurred_mask


def is_good_color(color):
    mean = sum(color) / len(color)
    if sum(color) == 0 or sum(color) >= 250 * len(color):
        return False
    if max(color) - min(color) <= 40:
        return False
    return True

def infer_mask(image):
    mask = np.zeros(image.size, dtype=np.uint8)
    colors = defaultdict(int)
    for i in range(image.size[0]):
        for j in range(image.size[1]):
            color = image.getpixel((i, j))
            if is_good_color(color):
                colors[color] += 1
    t = [[value, key] for key, value in colors.items()]
    t.sort(reverse=True)
    topcolor = max(colors, key=colors.get)
    topcolor = np.array(topcolor)
    
    for i in range(image.size[0]):
        for j in range(image.size[1]):
            pixel = np.array(image.getpixel((i, j)))
            
            if np.square(pixel.astype(float) - topcolor.astype(float)).sum() <= 25 ** 2:
                mask[i, j] = 255
    mask = Image.fromarray(mask.T)
    return mask