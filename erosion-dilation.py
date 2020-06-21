import cv2
import sys
from display_utils import *
from time import time


img = cv2.imread(sys.argv[1] if len(sys.argv) > 1 else 'test.png')

def wrap(v, x):
    if x < 0:
        return v + x
    return x

def get_neighbors(img, st, origin, w, h):
    img = np.pad(img, max(*origin), mode='wrap')
    selected = img[w:2*origin[0]+w,h:h+2*origin[1]]
    res = np.multiply(selected, st)
    return res[res != 0]

def sd_apply(img, st, origin, w, h, fn, res):
    start = time()
    for wi in range(w):
        for hi in range(h):
            nh = get_neighbors(img, st, origin, wi, hi)
            if len(nh):
                res[wi, hi] = fn(nh)
    end = time()
    print(end - start)

def erode(img, st, origin = None):
    if origin is None:
        origin = (st.shape[0] // 2, st.shape[1] // 2)
    img = np.copy(img)
    res = np.zeros(img.shape, dtype=np.uint8)
    w, h, c = img.shape[:3]
    for ci in range(c):
        current_channel = img[:,:,ci]
        current_res = res[:,:,ci]
        sd_apply(current_channel, st, origin, w, h, np.min, current_res)
    return res

def dilate(img, st, origin = None):
    if origin is None:
        origin = (st.shape[0] // 2, st.shape[1] // 2)
    img = np.copy(img)
    res = np.zeros(img.shape, dtype=np.uint8)
    w, h, c = img.shape[:3]
    for ci in range(c):
        current_channel = img[:,:,ci]
        current_res = res[:,:,ci]
        sd_apply(current_channel, st, origin, w, h, np.max, current_res)
    return res

structuring_element = np.ones((4, 4), np.uint8)
ero = erode(img, structuring_element)
dil = dilate(img, structuring_element)
op = dilate(ero, structuring_element)
cl = erode(dil, structuring_element)

img2 = side_by_side(
        Vertical(
            Tagged('Original', 0),
            Horizontal(
                Tagged('Eroded', 1),
                Tagged('Dilated', 2),
            ),
            Horizontal(
                Tagged('Opening', 3),
                Tagged('Closing', 4),
            ),
        ),
        img, ero, dil, op, cl
)
cv2.imshow('Out', img2)

wait_for_key('q', lambda: cv2.waitKey(0), cv2.destroyAllWindows)
