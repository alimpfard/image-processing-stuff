import cv2
import sys
from display_utils import *
from touma_utils import *

img = cv2.imread(sys.argv[1] if len(sys.argv) > 1 else 'test.png')

plist = (4, 8, 16, 32, 64, 128)
m = max(plist)
resl=[]
noisy = AddGaussianNoise(img, 0, 64, m)
subtracted = img - noisy[0]
results = [[], []]

for i,val in enumerate(plist):
    tmp = AverageImage(noisy[:val], scale=True)
    tmp2 = img - tmp
    tmp2 = (tmp2 - np.min(tmp2) * 255 / np.max(tmp2)).astype(np.uint8)
    results[0].append([f'Average ({val})', 'Subtraction'])
    results[1].append(tmp)
    results[1].append(tmp2)

layout = Vertical(
        0,
        nxm_matrix_view(range(1,3), ['Noisy', 'Subtraction'], 3, 1).transpose(),
        *[nxm_matrix_view(range(3+i*2,5+i*2), results[0][i], 2, 1).transpose() for i in range(len(results[0]))]
    )

print(layout)

cv2.imshow('Out', side_by_side(layout,
    img, noisy[0], subtracted, *results[1]
))

wait_for_key('q', lambda: cv2.waitKey(0), cv2.destroyAllWindows)
