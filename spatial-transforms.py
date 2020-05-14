import cv2
import sys
import numpy as np
from display_utils import *

img = cv2.imread(sys.argv[1] if len(sys.argv) > 1 else 'test.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def scale(f):
    return 255 / np.max(f) * f

def negative():
    m = np.max(img)
    filtered = np.apply_along_axis(lambda x: m - x, 0, img)
    return np.uint8(scale(filtered))

def log_transform(emph, offset):
    m = 1 + np.max(img)
    filtered = np.log1p(img / m)
    return np.uint8(scale(filtered))

def exp_transform(emph, offset):
    m = np.max(img)
    filtered = np.exp(img / m)
    return np.uint8(scale(filtered))

def nth_power(n):
    m = np.max(img)
    filtered = np.power(img / m, n)
    return np.uint8(scale(filtered))

@np.vectorize
def value(pix, r1, s1, r2, s2):
    if 0 <= pix <= r1:
        return s1 / r1 * pix
    elif r1 < pix <= r2:
        return (s2 - s1) / (r2 - r1) * (pix - r1) + s1
    else:
        return (255 - s2) / (255 - r2) * (pix - r2) + s2

def contrast(*ps):
    res = value(img, *ps)
    return (render_function(lambda x: value(x, *ps), (255, 255)),
            res.astype(np.uint8))

views = ['original', 'negative', 'log', 'exp', '2ndpow', '3rdpow', '2ndroot', '3rdroot', 'contrast stretching']
white = empty((255, 255))
img2 = side_by_side(
        nxm_matrix_view(range(2 * len(views)), views, 1, 2),
        white, img,
        white, negative(),
        white, log_transform(1, 0),
        white, exp_transform(1, 0),
        white, nth_power(2),
        white, nth_power(3),
        white, nth_power(1/2),
        white, nth_power(1/3),
        *contrast(20, 40, 180, 200),
        )
cv2.imshow('Out', img2)

wait_for_key('q', lambda: cv2.waitKey(0), cv2.destroyAllWindows)
