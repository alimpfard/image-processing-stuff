import cv2
import sys
from display_utils import *
from scipy.signal import convolve2d

def laplacian_kernel(pos, diagonals, /, kernels={
        3: [[0,1,0],[1,-4,1],[0,1,0]],
        2: [[1,1,1],[1,-8,1],[1,1,1]],
        1: [[0,-1,0],[-1,4,-1],[0,-1,0]],
        0: [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]
    }):
        return np.array(kernels[2 * int(pos) + int(diagonals)])

def laplacian_enhancement_kernel(offset, diagonals):
    if diagonals:
        return np.array([[-1,-1,-1],[-1,offset+8,-1],[-1,-1,-1]])
    else:
        return np.array([[0,-1,0],[-1,offset + 4,-1],[0,-1,0]])


img = cv2.imread(sys.argv[1] if len(sys.argv) > 1 else 'test.png')

img = (img.sum(axis=2) / 3).astype(np.uint8)

def positive_laplacian():
    mask = laplacian_kernel(True, True)
    res = convolve2d(img, mask, mode='same')
    return res

def negative_laplacian():
    mask = laplacian_kernel(False, True)
    res = convolve2d(img, mask, mode='same')
    return res

def sharpen_laplacian(lp, pos):
    if pos:
        return (img - lp).clip(0, 255).astype(np.uint8)
    else:
        return (img + lp).clip(0, 255).astype(np.uint8)

def scale(thing):
    thing = np.copy(thing.astype(np.float))
    thing -= np.min(thing)
    thing /= np.max(thing)
    thing *= 255
    return thing.astype(np.uint8)


positive_lp = positive_laplacian()
negative_lp = negative_laplacian()

img2 = side_by_side(
        Vertical(
            Tagged('Original', 0),
            Tagged('Positive Laplacian', Horizontal(1, 2)),
            Tagged('Negative Laplacian', Horizontal(3, 4))
        ),
        img, scale(positive_lp), sharpen_laplacian(positive_lp, True), scale(negative_lp), sharpen_laplacian(negative_lp, False)
)
cv2.imshow('Out', img2)

wait_for_key('q', lambda: cv2.waitKey(0), cv2.destroyAllWindows)
