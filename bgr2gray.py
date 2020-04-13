import cv2
import sys
import numpy as np
from display_utils import side_by_side, Tagged, Spacer, Vertical, Horizontal

image = cv2.imread(sys.argv[1] if len(sys.argv) > 1 else 'test.png')

## opencv way
cv_gray = cv2.cvtColor(image,  cv2.COLOR_BGR2GRAY)

## manual way
gray = cv2.convertScaleAbs(
        np.sum(
            ## Luma - good old way
            # image * np.array([[[ 0.21, 0.72, 0.07 ]]]),
            ## luminosity
            image * np.array([[[ 0.11, 0.59, 0.3 ]]]),
            axis = 2))

cv2.imshow('result', side_by_side(
        Tagged('BGR2Gray by AliMPFard',
            Vertical(
                Spacer(40),
                Tagged('original', 0),
                Spacer(20),
                Horizontal(
                    Tagged('OpenCV', [1] * 3),
                    Spacer(20),
                    Tagged('Mine', [2] * 3)
                )
            ), fill='#a06c73'),
        image, cv_gray, gray
    )
)

while cv2.waitKey(0) & 0xff != ord('q'):
    pass

cv2.destroyAllWindows()
