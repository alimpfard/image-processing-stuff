import cv2
import sys
from display_utils import *

img = cv2.imread(sys.argv[1] if len(sys.argv) > 1 else 'test.png')

img2 = side_by_side(
        Vertical(
            Tagged('Original', 0),
            Tagged('Histogram', Histogram(0, normalised=True))
        ),
        img
)
cv2.imshow('Out', img2)

wait_for_key('q', lambda: cv2.waitKey(0), cv2.destroyAllWindows)
