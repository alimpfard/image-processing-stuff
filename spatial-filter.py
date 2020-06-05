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
}

img = cv2.imread(sys.argv[1] if len(sys.argv) > 1 else 'test.png')
img = (img.sum(axis=2)/3).astype(np.uint8)

other_img = cv2.imread(sys.argv[2] if len(sys.argv) > 2 else 'skelly.png')
other_img = (other_img.sum(axis=2)/3).astype(np.uint8)

def apply_kernel(kern,to=img):
    res = convolve2d(to, kern, mode='same')
    return res

def average(kernel=KERNEL_SIZE, img=img):
    return scale(apply_kernel(kernels['average'](kernel), img))

def minf():
    return minimum_filter(img, size=KERNEL_SIZE, mode='nearest')

def maxf():
    return maximum_filter(img, size=KERNEL_SIZE, mode='nearest')

def highpass(idt):
    return scale(apply_kernel(kernels[f'highpass-{idt}']))

def emboss(direction):
    return scale(apply_kernel(kernels[f'emboss-{direction}']))

def laplacian(idt,img=img):
    return scale(apply_kernel(kernels[f'laplacian-{idt}'], img))

def laplacian_sharp(idt, img=img):
    return np.clip(apply_kernel(kernels[f'laplacian_sharpen-{idt}'], img), 0, 255).astype(np.uint8)

def sobel(img=img):
    return scale(
            apply_kernel(kernels['sobel-xr'],
                apply_kernel(kernels['sobel-yu'],
                    apply_kernel(kernels['sobel-xl'],
                        apply_kernel(kernels['sobel-yd'], img)))))

def roberts():
    return scale(
            apply_kernel(kernels['roberts-x'],
                apply_kernel(kernels['roberts-y'])))


def clip(thing):
    return np.clip(thing, 0, 255).astype(np.uint8)

def scale(thing):
    thing = np.copy(thing.astype(np.float))
    thing -= np.min(thing)
    thing /= np.max(thing)
    thing *= 255
    return thing.astype(np.uint8)


img2 = side_by_side(
    Vertical(
        Tagged('Original', 0),
        Tagged('Averaged (3x3 kernel)', 1),
        Tagged('Max (3x3 window)', 2),
        Tagged('Min (3x3 window)', 3),
        Tagged('"HighPass"', 4),
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
    ),
    img, average(), minf(), maxf(), highpass(0),
    emboss('e'), emboss('nw'), emboss('s'), emboss('se'),
    emboss('n'), emboss('ne'), emboss('w'), emboss('sw'),
    laplacian(0), laplacian(1), laplacian(2), empty(img.shape),
    laplacian('ortho'), laplacian('horiz'), laplacian('LR'),
    laplacian('RL'),
    laplacian_sharp('sharpen'), laplacian_sharp('unsharpen'),
    sobel(), roberts(),
    # Skelly
    (a:=other_img), laplacian(0, other_img), (c:=laplacian_sharp('sharpen', other_img)),
    (d:=sobel(other_img)), (e:=average(5, d)), (f:=scale(c * e)), (g:=clip(a+f)), clip(g)
)

cv2.imshow('Out', img2)

wait_for_key('q', lambda: cv2.waitKey(0), cv2.destroyAllWindows)
