import sys
from display_utils import *
from time import time

Backend.PIL()
img0 = read(sys.argv[1] if len(sys.argv) > 1 else 'test.png')
img1 = read(sys.argv[2] if len(sys.argv) > 2 else 'test.png')

def count(arr):
    return np.prod(arr.shape)

K1 = 0.01 # arbitrary constant << 1
K2 = 0.03 # arbitrary constant << 1
C1 = (K1 * 255) ** 2
C2 = (K2 * 255) ** 2
C3 = C2 / 2 # chosen as per the paper.

# Calculates the luminance value of the image
# this is just the mean of all the pixels.
def lum_of(image, W):
    return np.sum(np.multiply(W, image))

# Calculates the contrast of the image
# This is the standard deviation of all the pixels.
def contr_of(image, W, lum):
    return np.sqrt(np.sum(np.multiply(W, (image - lum) ** 2)))
# Covariance
# This functions calculates a discrete estimate for the
# covariance coefficient between two images.
def covar(x, y, mx, my, W):
    return np.sum(np.multiply(W, np.multiply(x - mx, y - my)))

# SSIM
# alpha, beta, and gamma symbolise how important each component is
# the default 1,1,1 causes all components to contribute equally to the
# overall index.
def ssim(img0, img1, W, *args):
    mx = lum_of(img0, W)
    my = lum_of(img1, W)
    sd0 = contr_of(img0, W, mx)
    sd1 = contr_of(img1, W, mx)
    cov = covar(img0, img1, mx, my, W)
    return np.multiply(2*np.multiply(mx, my) + C1, 2*cov + C2) / ((mx**2 + my**2 + C1) * (sd0 ** 2 + sd1 ** 2 + C2))

def gaussian_kernel(size=5, std=1):
    t = 2 * std ** 2
    m = (size - 1) / 2
    x, y = np.abs(np.mgrid[0:size, 0:size] - m).astype(np.float64)
    tmp = np.exp(-(x ** 2 + y ** 2) / t) / t
    tmp = tmp / tmp[0][0]
    return tmp / np.sum(tmp, dtype=np.float64)

def mssim(img0, img1, alpha = 1, beta = 1, gamma = 1):
    if img0.shape != img1.shape:
        raise Exception("Fuck off, same sizes only")

    size = 11
    kern = gaussian_kernel(size, 1.5)
    combined_ssim = []
    def mssim_single_channel(img0, img1):
        for i in range(0, img0.shape[0] - size + 1):
            for j in range(0, img0.shape[1] - size + 1):
                window0 = img0[i:i+size,j:j+size]
                window1 = img1[i:i+size,j:j+size]
                wssim = ssim(window0, window1, kern, alpha, beta, gamma)
                combined_ssim.append(wssim)

    if len(img0.shape) == 3:
        for z in range(img0.shape[2]):
            mssim_single_channel(img0[:,:,z], img1[:,:,z])
    elif len(img0.shape) == 2:
        mssim_single_channel(img0, img1)
    else:
        raise Exception("Check your images pal, those are weird")

    return np.average(np.array(combined_ssim))


app = mssim(img0, img1)

img2 = side_by_side(
        Vertical(
            Tagged(f'SSIM = {app}',
                Horizontal(
                    Vertical(Tagged('Img0', 0)),
                    Vertical(Tagged('Img1', 1)),
                )
            )
        ),
        img0, img1
)
show(img2, persist=True)

