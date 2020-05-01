import cv2
import numpy as np
from display_utils import *

img = cv2.imread('fft-test.png')
img = np.sum(img, 2) / 3

# fft - np
fft = np.fft.fft2(img)

# fft - shift
fshift = np.fft.fftshift(fft)

# Touma: magnitude is displayed
mag_spectrum = 20 * np.log(np.abs(fshift))

cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Output', 1000, 1000)

color = '#0000ff'
cv2.imshow('Output', side_by_side(
    Tagged('FFT Demo - AliMPFard', 
        Vertical(
            Spacer(40),
            Vertical(
                Horizontal(
                    Tagged('Original', [0] * 3, fill=color),
                    Spacer(),
                    Tagged('FFT - unshifted', [1] * 3, fill=color),
                ),
                Spacer(),
                Horizontal(
                    Tagged('FFT - shifted', [2] * 3, fill=color),
                    Spacer(),
                    Tagged('FFT - shifted, magnitude', [3] * 3, fill=color)
                )
            )
        )
    ),
    img, np.abs(fft), np.abs(fshift), mag_spectrum
))

wait_for_key('q', lambda: cv2.waitKey(0), cv2.destroyAllWindows)