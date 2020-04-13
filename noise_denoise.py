import cv2
import sys
import numpy as np
import heapq
from display_utils import side_by_side, Tagged, Spacer, Vertical, Horizontal

def mfilter(data, filter_size):
    indexer = filter_size // 2
    window = [
        (i, j)
        for i in range(-indexer, filter_size-indexer)
        for j in range(-indexer, filter_size-indexer)
    ]
    index = len(window) // 2
    for i in range(len(data)):
        for j in range(len(data[0])):
            data[i][j] = heapq.nlargest(index+1,
                (0 if (
                    min(i+a, j+b) < 0
                    or len(data) <= i+a
                    or len(data[0]) <= j+b
                ) else data[i+a][j+b]
                for a, b in window)
            )[index]
    return data

image = cv2.imread(sys.argv[1] if len(sys.argv) > 1 else 'test.png')

how_much_spice = float(sys.argv[2] if len(sys.argv) > 2 else 0.06)
how_much_blur = int(sys.argv[3] if len(sys.argv) > 3 else 3)
rgb_spice = True # False for true "Pepper and Salt"

noisy = np.copy(image)

if rgb_spice:
    noise = np.random.rand(*image.shape) * 255
    noise = noise < (255 * how_much_spice)
    noisy[noise] = 255
else:
    black = noise < (255 * how_much_spice)
    white = noise > (255 * (1 - how_much_spice))
    noisy[black] = [0,0,0]
    noisy[white] = [255,255,255]


## cv2 denoise
cv_denoise = cv2.medianBlur(noisy, how_much_blur)

## manual denoise
denoise = np.copy(noisy)
denoise[:,:,0] = mfilter(denoise[:,:,0], 3)
denoise[:,:,1] = mfilter(denoise[:,:,1], 3)
denoise[:,:,2] = mfilter(denoise[:,:,2], 3)

cv2.imshow('result', side_by_side(
        Tagged('Median Filter Denoise - AliMPFard', Vertical(
                Spacer(40),
                Horizontal(
                    Tagged('Original', 0, fill='rgb(0,0,255)'),
                    Spacer(image.shape[1] // 15),
                    Tagged('Noisy', 1, fill='rgb(0,0,255)')
                ),
                Spacer(image.shape[0] // 15),
                Horizontal(
                    Tagged('CV2', 2, fill='rgb(0,0,255)'),
                    Spacer(image.shape[1] // 15),
                    Tagged('Mine', 3, fill='rgb(0,0,255)')
                )
        ), fill='#a0736c'),
    image,
    noisy,
    cv_denoise,
    denoise)
)

while cv2.waitKey(0) != ord('q'):
    pass

cv2.destroyAllWindows()
