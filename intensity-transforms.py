import cv2
import sys
import numpy as np
from display_utils import *

img = cv2.imread(sys.argv[1] if len(sys.argv) > 1 else 'test.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

A = 50
B = 100
SliceIntensity = 130
SliceOffset = 60

@np.vectorize
def value(pix, *rs, cache={}):
    rs_and_slopes = []
    if tuple(rs) in cache:
        rs_and_slopes = cache[tuple(rs)]
    else:
        last_point = [0,0]
        for _rs in chunks(rs, 2):
            rs_ = list(_rs)
            slope = (rs_[1] - last_point[1]) / (rs_[0] - last_point[0])
            rs_and_slopes.append((last_point[0], rs_[0], last_point[1], slope))
            last_point = rs_
        cache[tuple(rs)] = rs_and_slopes

    for rs_and_slope in rs_and_slopes:
        if pix > rs_and_slope[0] and pix <= rs_and_slope[1]:
            return (pix - rs_and_slope[0]) * rs_and_slope[3] + rs_and_slope[2]

    return 0


def gray(*ps):
    res = value(img, *ps)
    return (render_function(lambda x: value(x, *ps), (255, 255)),
            res.astype(np.uint8))

def bitslice():
    return [
            cv2.bitwise_and(img, np.full(img.shape, 2 ** k, np.uint8)) * 255
        for k in range(8)
    ]

views = ['gray slice', 'bit slice']
white = empty((255, 255))
img2 = side_by_side(
        layout_with_names({
            'original': Horizontal(13, 0),
            'gray slice 0': Horizontal(1, 2),
            'gray slice 1': Horizontal(3, 4),
            'bitslice': nxn_matrix_view(range(5, 13), ['' for x in range(8)], 4)
        }, Vertical),
        img,
        *gray(A, A, A, SliceIntensity, B, SliceIntensity, B, B, 255, 255),
        *gray(0, SliceOffset, A, SliceOffset, A, SliceIntensity + SliceOffset, B, SliceIntensity + SliceOffset, B, SliceOffset, 255, SliceOffset),
        *bitslice(),
        white
)
cv2.imshow('Out', img2)

wait_for_key('q', lambda: cv2.waitKey(0), cv2.destroyAllWindows)
