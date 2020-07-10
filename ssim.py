import sys
from display_utils import *
from time import time

Backend.PIL()

img0 = read(sys.argv[1] if len(sys.argv) > 1 else 'test.png')
img1 = read(sys.argv[2] if len(sys.argv) > 2 else 'test.png')

def count(arr):
    return np.prod(arr.shape)

K1 = 0.01 # arbitrary constant << 1
K2 = 0.01 # arbitrary constant << 1
C1 = (K1 * 255) ** 2
C2 = (K2 * 255) ** 2
C3 = C2 / 2 # chosen as per the paper.

# Calculates the luminance value of the image
# this is just the mean of all the pixels.
def lum_of(image):
    return np.sum(image) / count(image)

# Calculates the contrast of the image
# This is the standard deviation of all the pixels.
def contr_of(image, lum):
    return np.sqrt(np.sum((image - lum) ** 2) / (count(image)-1))

# The "structure" of an image is just its normalised value
# the values signify how intensely a pixel deviates from the norm.
def struct_of(image, lum, contr):
    return (image - lum) / contr

# Comparison of luminance components
# as given in the paper, this is the `l(.)` function
def luminance_compare(lum0, lum1):
    return (2*lum0*lum1 + C1) / (lum0 * lum0 + lum1 * lum1 + C1)

# Comparison of contrast components
# as given in the paper, this is the `c(.)` function
def contrast_compare(ctr0, ctr1):
    return (2*ctr0*ctr1 + C1) / (ctr0 * ctr0 + ctr1 * ctr1 + C2)

# Covariance
# This functions calculates a discrete estimate for the
# covariance coefficient between two images.
def covar(x, y, mx, my):
    return np.sum(np.multiply(x - mx, y - my)) / (count(x) - 1)

# Comparison of structure components
# This is the `s(.)` function.
def structure_compare(str0, str1, m0, m1, sdev0, sdev1):
    return (covar(str0, str1, sdev0, sdev1) + C3) / (sdev0 * sdev1 + C3)

# SSIM
# alpha, beta, and gamma symbolise how important each component is
# the default 1,1,1 causes all components to contribute equally to the
# overall index.
def ssim(img0, img1, alpha = 1, beta = 1, gamma = 1):
    l0 = lum_of(img0)
    l1 = lum_of(img1)
    c0 = contr_of(img0, l0)
    c1 = contr_of(img1, l1)
    s0 = struct_of(img0, l0, c0)
    s1 = struct_of(img1, l1, c1)
    a = luminance_compare(l0, l1) ** alpha
    b = contrast_compare(c0, c1) ** beta
    c = structure_compare(s0, s1, l0, l1, c0, c1) ** gamma
    return a * b * c

img2 = side_by_side(
        Vertical(
            Tagged(f'SSIM = {ssim(img0, img1)}',
                Horizontal(
                    Vertical(Tagged('Img0', 0)),
                    Vertical(Tagged('Img1', 1)),
                )
            )
        ),
        img0, img1
)
show(img2, persist=True)

