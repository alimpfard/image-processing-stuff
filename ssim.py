import sys
from time import time

from display_utils import *

# Grab an implementation of PSNR to compare with.
from psnr import PSNR

# Initialise a new PIL based display tool from display_utils
Backend.PIL()

# Check command line arguments to see if any image is specified or not
# Use default images if not.
img0 = read(sys.argv[1] if len(sys.argv) > 1 else 'Test Set/test1.png')
img1 = read(sys.argv[2] if len(sys.argv) > 2 else 'Test Set/test6.jpeg')

# Define needed constants as described in paper
K1 = 0.01 # arbitrary constant << 1
K2 = 0.03 # arbitrary constant << 1
C1 = (K1 * 255) ** 2
C2 = (K2 * 255) ** 2
C3 = C2 / 2 # chosen as per the paper.

GENERAL_FORM = True # use the general form of the formulae

# Calculates the luminance value of the window
# This is just the weighted average of all the pixels inside of window.
def lum_of(image, W):
    return np.sum(np.multiply(W, image))

# Calculates the contrast of the window
# This is the standard deviation of all the pixels.
def contr_of(image, W, lum):
    return np.sqrt(np.sum(np.multiply(W, (image - lum) ** 2)))

# Covariance
# This function calculates a discrete estimate for the
# covariance coefficient between two windows.
def covar(x, y, mx, my, W):
    return np.sum(np.multiply(W, np.multiply(x - mx, y - my)))

# Luminance comparison function
def lum_com(l1,l2):
    return ((2 * (l1 * l2) + C1) / (l1 ** 2 + l2 ** 2 + C1))

# Contrast comparison function
def con_com(s1,s2):
    return ((2 * (s1 * s2) + C2) / (s1 ** 2 + s2 ** 2 + C2))

# Structure comparison function
def str_com(s1,s2,st):
    return ((st + C3) / (s1 * s2 + C3))

# SSIM
# alpha, beta, and gamma symbolise how important each component is
# The default 1,1,1 causes all components to contribute equally to the
# overall index.
def ssim(window0, window1, W, alpha, beta, gamma, *args):
    # Calculate luminance factor of the two windows
    mx = lum_of(window0, W)
    my = lum_of(window1, W)
    # Calculate standard deviation of the two windows as an estimation of contrast
    sd0 = contr_of(window0, W, mx)
    sd1 = contr_of(window1, W, mx)
    # Calculate correlation coefficient of the two windows
    cov = covar(window0, window1, mx, my, W)
    if GENERAL_FORM:
        return (lum_com(mx,my) ** alpha) * (con_com(sd0,sd1) ** beta) * (str_com(sd0,sd1,cov) ** gamma)
    else:
        return np.multiply(2*np.multiply(mx, my) + C1, 2 * cov + C2) / ((mx ** 2 + my ** 2 + C1) * (sd0 ** 2 + sd1 ** 2 + C2))

# Gaussian Kernel Generator
# We use this function to genrate gaussian weight matrix.
# This matrix is used as a countermeasure to blocking artifacts.
def gaussian_kernel(size=5, std=1):
    t = 2 * std ** 2
    m = (size - 1) / 2
    x, y = np.abs(np.mgrid[0:size, 0:size] - m).astype(np.float64)
    tmp = np.exp(-(x ** 2 + y ** 2) / t) / t
    tmp = tmp / tmp[0][0]
    return tmp / np.sum(tmp, dtype=np.float64)

# Mean SSIM
# This function calculates SSIM for each corresponding subwindow of the two images
# and then returns the average of these SSIMs as a metric of overall image similarity.
def mssim(img0, img1, alpha = 1, beta = 1, gamma = 1):
    if img0.shape != img1.shape:
        raise Exception("Bad input, two images should be of the same size!")

    # Set window size to 11 (the same as paper)
    size = 11
    # Generate a weight matrix
    kern = gaussian_kernel(size, 1.5)
    # Initialise an empty list of SSIM values
    combined_ssim = []

    # A function that takes two single channel images to first compute SSIMs for
    # every corresponding subwindow and add the values to aforementioned list.
    def mssim_single_channel(img0, img1):
        # A simple rolling window model
        for i in range(0, img0.shape[0] - size + 1):
            for j in range(0, img0.shape[1] - size + 1):
                # Extract the current window from images
                window0 = img0[i:i+size,j:j+size]
                window1 = img1[i:i+size,j:j+size]
                # Compute SSIM and add it to the list
                wssim = ssim(window0, window1, kern, alpha, beta, gamma)
                combined_ssim.append(wssim)

    if len(img0.shape) == 3:
        # Split the channels and process them seperately
        # if the images are polychrome
        for z in range(img0.shape[2]):
            mssim_single_channel(img0[:,:,z], img1[:,:,z])
    elif len(img0.shape) == 2:
        # Process the images directly if they are monochrome
        mssim_single_channel(img0, img1)
    else:
        # Bad image case! Either a linear signal or a weird array
        raise Exception("Check your images pal, those are weird")

    # Return the average of SSIMs
    return np.average(np.array(combined_ssim))

# Run the function to get MSSIM of the two images
app = mssim(img0, img1)

# Calculate the PSNR of the two images - this is to compare with SSIM
psnr = PSNR(img0, img1)

# Put them togheter to show the result
img2 = side_by_side(
        Vertical(
            Tagged(f'SSIM = {app}',
                Tagged(f'PSNR = {psnr}',
                    Horizontal(
                        Vertical(Tagged('Img0', 0)),
                        Vertical(Tagged('Img1', 1)),
                    )
                )
            )
        ),
        img0, img1
)

show(img2, persist=True)

