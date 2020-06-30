## Image Processing Stuff

All of these implementations test library functions against manually written code, so expect slow execution times.

- `bgr2gray.py`
converts an image from BGR to Gray

- `noise_denoise.py`
Noisifies an image and then applies a median filter to it

- `fourier-transform.py`
Brings an image to the frequency domain

- `fourier-filters.py`
Applies multiple FFT-based filters: {gaussian,butterworth,ideal} (high,low)-pass filters

- `homomorphic-filter.py`
Applies some homomorphic filters

- `spatial-transforms.py`
Applies multiple spatial transformations (Negative, Log, Exp, Power)

- `intensity-transforms.py`
Applies multiple spatial intensity transforms (pw-linear, bit-plane slicing)

- `histogram.py`
Renders a histogram of the provided image (actual source in `display_utils.py` in `Histogram`)

- `image-averaging.py`
Attempts to remove gaussian noise from image by averaging multiple noisy images

- `laplacian.py`
Does some laplacian stuff (Image sharpening)

- `spatial-filters.py`
Collections of all implemented spatial filters (WIP as of yet)

- `erosion-dilation.py`
Applies morphological operators "erode", "dilate" and their mixups "opening mode" and "closing mode"

- `ssim.py`
Implements SSIM

#### Utilities

- `display_utils.py`
Implements a nice image layout engine:
```py
side_by_side(LAYOUT, ...images)

# where LAYOUT is a layout descriptor:
Vertical(*elements)   # stacks elements vertically
Horizontal(*elements) # stacks elements horizontally
Spacer(SIZE)          # creates some empty space SIZE-wide (or long)
Tagged(TAG, LAYOUT, **options) # adds a caption to a given layout

[*INDEX]              # creates a multi-channel overlay of the mentioned indices
INDEX                 # copies the image referenced by images[INDEX]
```

named Matrix-like layouts can be created via `nxm_matrix_view`:
```py
view = nxm_matrix_view(range(8), ['foo', 'bar'], 2, 2)

# -> A tagged matrix view
# Vertical
#   Tagged(foo)
#     Vertical
#       Spacer(40 px)
#       Vertical
#         Horizontal
#           0
#           1
#         Horizontal
#           2
#           3
#   Tagged(bar)
#     Vertical
#       Spacer(40 px)
#       Vertical
#         Horizontal
#           4
#           5
#         Horizontal
#           6
#           7
```

