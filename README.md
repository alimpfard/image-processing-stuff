## Image Processing Stuff

All of these implementations test library functions against manually written code, so expect slow execution times.

- `bgr2gray.py`
converts an image from BGR to Gray

- `noise_denoise.py`
Noisifies an image and then applies a median filter to it

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

