import cv2
import sys
import numpy as np
from display_utils import *

img = cv2.imread(sys.argv[1] if len(sys.argv) > 1 else 'test.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def negative():
    m = np.max(img)
    filtered = np.apply_along_axis(lambda x: m - x, 0, img)
    return np.uint8(filtered)

def log_transform(emph, offset):
    m = np.max(img)
    filtered = np.log1p(img / m)
    return np.uint8(filtered * m * emph / np.max(filtered) + offset)

def exp_transform(emph, offset):
    m = np.max(img)
    filtered = np.exp(img / m)
    return np.uint8(filtered * m * emph / np.max(filtered) + offset)

def nth_power(n):
    m = np.max(img)
    filtered = np.power(img / m, n)
    return np.uint8(filtered * m / np.max(filtered))

views = ['original', 'negative', 'log', 'exp', '2ndpow', '3rdpow', '2ndroot', '3rdroot']
img2 = side_by_side(
        nxn_matrix_view(range(len(views)), views, 1),
        img,
        negative(),
        log_transform(1, 0),
        exp_transform(1, 0),
        nth_power(2),
        nth_power(3),
        nth_power(1/2),
        nth_power(1/3),
        )
cv2.imshow('Out', img2)

wait_for_key('q', lambda: cv2.waitKey(0), cv2.destroyAllWindows)
