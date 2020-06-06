import cv2
from display_utils import *
from scipy.signal import convolve2d
from scipy.ndimage import maximum_filter, minimum_filter
import sys

KERNEL_SIZE=3

kernels = {
    'average': lambda x: np.ones((x,x)),
    'highpass-0': np.array([-1,-1,-1,-1,9,-1,-1,-1,-1]).reshape((3,3)),
    'emboss-e': np.array([0,0,0,1,0,-1,0,0,0]).reshape((3,3)),
    'emboss-nw': np.array([0,0,1,0,0,0,-1,0,0]).reshape((3,3)),
    'emboss-se': np.array([-1,-1,1,-1,-2,1,1,1,1]).reshape((3,3)),
    'emboss-s': np.array([-1,-1,-1,1,-2,1,1,1,1]).reshape((3,3)),
    'emboss-n': np.array([1,1,1,1,-2,1,-1,-1,-1]).reshape((3,3)),
    'emboss-ne': np.array([1,1,1,1,-2,-1,1,-1,-1]).reshape((3,3)),
    'emboss-w': np.array([1,1,-1,1,-2,-1,1,1,-1]).reshape((3,3)),
    'emboss-sw': np.array([1,-1,-1,1,-2,-1,1,1,1]).reshape((3,3)),
    'laplacian-0': np.array([0,-1,0,-1,4,-1,0,-1,0]).reshape((3,3)),
    'laplacian-1': np.array([-1,-1,-1,-1,8,-1,-1,-1,-1]).reshape((3,3)),
    'laplacian-2': np.array([1,-2,1,-2,4,-2,1,-2,1]).reshape((3,3)),
    'laplacian-ortho': np.array([0,-1,0,0,2,0,0,-1,0]).reshape((3,3)),
    'laplacian-horiz': np.array([0,0,0,-1,2,-1,0,0,0]).reshape((3,3)),
    'laplacian-RL': np.array([-1,0,0,0,2,0,0,0,-1]).reshape((3,3)),
    'laplacian-LR': np.array([0,0,-1,0,2,0,-1,0,0]).reshape((3,3)),
    'laplacian_sharpen-sharpen':
                    np.array([0,-1,0,-1,5,-1,0,-1,0]).reshape((3,3)),
    'laplacian_sharpen-unsharpen':
                    np.array([1,1,1,1,-7,1,1,1,1]).reshape((3,3)),
    'sobel-xr': np.array([-1,0,1,-2,0,2,-1,0,1]).reshape((3,3)),
    'sobel-xl': np.array([1,0,-1,2,0,-2,1,0,-1]).reshape((3,3)),
    'sobel-yu': np.array([1,2,1,0,0,0,-1,-2,-1]).reshape((3,3)),
    'sobel-yd': np.array([-1,-2,-1,0,0,0,1,2,1]).reshape((3,3)),
    'roberts-x':np.array([0,0,0,0,1,0,0,0,-1]).reshape((3,3)),
    'roberts-y':np.array([0,0,0,1,0,0,0,-1,0]).reshape((3,3)),
    'gaussian': np.array([1,4,7,4,1,4,16,26,16,4,7,26,41,26,7,4,16,26,16,4,1,4,7,4,1]).reshape((5,5)),
}

img = cv2.imread(sys.argv[1] if len(sys.argv) > 1 else 'test.png')
img = (img.sum(axis=2)/3).astype(np.uint8)

other_img = cv2.imread(sys.argv[2] if len(sys.argv) > 2 else 'skelly.png')
other_img = (other_img.sum(axis=2)/3).astype(np.uint8)

def nth_power(img,n):
    m = np.max(img)
    filtered = np.power(img / m, n)
    return np.uint8(scale(filtered))

def apply_kernel(kern,to=img):
    res = convolve2d(to, kern, mode='same')
    return res

def exaverage(kernel=KERNEL_SIZE, img=img):
    return apply_kernel(kernels['average'](kernel), img)/kernel

def average(*args):
    return scale(exaverage(*args))

def minf():
    return minimum_filter(img, size=KERNEL_SIZE, mode='nearest')

def maxf():
    return maximum_filter(img, size=KERNEL_SIZE, mode='nearest')

def gaussian():
    return scale(apply_kernel(kernels['gaussian']).astype(np.float) / 273)

def highpass(idt):
    return scale(apply_kernel(kernels[f'highpass-{idt}']))

def emboss(direction):
    return scale(apply_kernel(kernels[f'emboss-{direction}']))

def exlaplacian(idt,img=img):
    return np.abs(apply_kernel(kernels[f'laplacian-{idt}'], img))

def laplacian(*args):
    return scale(exlaplacian(*args))

def laplacian_sharp(idt, img=img):
    return clip(apply_kernel(kernels[f'laplacian_sharpen-{idt}'], img))

def exsobel(img=img):
    dx = apply_kernel(kernels['sobel-xr'], img)
    dy = apply_kernel(kernels['sobel-yu'], img)
    return np.abs(dx) + np.abs(dy)

def sobel(*args):
    return scale(exsobel(*args))

def roberts():
    return scale(
            np.abs(apply_kernel(kernels['roberts-x'])) +
            np.abs(apply_kernel(kernels['roberts-y'])))


def clip(thing):
    return np.clip(thing, 0, 255).astype(np.uint8)

def scale(thing):
    thing = np.copy(thing.astype(np.float))
    thing -= np.min(thing)
    thing /= np.max(thing)
    thing *= 255
    return thing.astype(np.uint8)

layout = Vertical(
        Tagged('Original', 0),
        Tagged('Averaged (3x3 kernel)', 1),
        Tagged('Max (3x3 window)', 2),
        Tagged('Min (3x3 window)', 3),
        Tagged('"HighPass"', 4),
        Tagged('Gaussian', 33),
        Tagged('Embossed',
            Vertical(
                Horizontal(
                    Tagged('E', 5),
                    Tagged('NW', 6),
                ),
                Horizontal(
                    Tagged('S', 7),
                    Tagged('SE', 8),
                ),
                Horizontal(
                    Tagged('N', 9),
                    Tagged('NE', 10),
                ),
                Horizontal(
                    Tagged('W', 11),
                    Tagged('SW', 12)
                )
            )
        ),
        Tagged('Laplacian',
            Vertical(
                Horizontal(
                    Tagged('V1', 13),
                    Tagged('V2', 14),
                ),
                Horizontal(
                    Tagged('V3', 15),
                    Tagged('', 16)
                ),
                Horizontal(
                    Tagged('Or', 17),
                    Tagged('Ho', 18)
                ),
                Horizontal(
                    Tagged('LR', 19),
                    Tagged('RL', 20)
                )
            )
        ),
        Tagged('Laplacian Sharpen',
            Vertical(
                Horizontal(
                    Tagged('Sharpen', 21),
                    Tagged('Unsharpen', 22),
                )
            )
        ),
        Tagged('Sobel', 23),
        Tagged('Robert\'s', 24),
        Tagged('Mixed Operations',
            Horizontal(
                Vertical(
                    Horizontal(
                        Vertical(
                            Tagged('(a) Original', 25),
                        ),
                        Vertical(
                            Tagged('(b) Laplacian', 26),
                        ),
                    ),
                    Horizontal(
                        Vertical(
                            Tagged('(c) Laplacian-Sharpened', 27),
                        ),
                        Vertical(
                            Tagged('(d) Sobel', 28),
                        )
                    ),
                ),
                Spacer(),
                Vertical(
                    Horizontal(
                        Vertical(
                            Tagged('(e) 5x5 Averaged d', 29),
                        ),
                        Vertical(
                            Tagged('(f) c*e', 30),
                        )
                    ),
                    Horizontal(
                        Vertical(
                            Tagged('(g) a+f Sharpened', 31),
                        ),
                        Vertical(
                            Tagged('(h) Powerlaw to g', 32),
                        )
                    ),
                )
            )
        )
    )

a = other_img
b = exlaplacian(0, a)
c = a + b
d = exsobel(a)
e = exaverage(5, d)
f = scale(np.abs(c * e))
g = clip(a.astype(np.float) + f)
h = nth_power(g, 0.5)

img2 = side_by_side(
    layout,
    img, average(), minf(), maxf(), highpass(0),
    emboss('e'), emboss('nw'), emboss('s'), emboss('se'),
    emboss('n'), emboss('ne'), emboss('w'), emboss('sw'),
    laplacian(0), laplacian(1), laplacian(2), empty(img.shape),
    laplacian('ortho'), laplacian('horiz'), laplacian('LR'),
    laplacian('RL'),
    laplacian_sharp('sharpen'), laplacian_sharp('unsharpen'),
    sobel(), roberts(),
    # Skelly
    a,scale(b),clip(c),clip(d),scale(e),f,g,scale(h),
    # gauss
    gaussian()
)

cv2.imshow('Out', img2)

wait_for_key('q', lambda: cv2.waitKey(0), cv2.destroyAllWindows)
