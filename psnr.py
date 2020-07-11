import numpy as np

def PSNR(a, b):
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return 100
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    return psnr
