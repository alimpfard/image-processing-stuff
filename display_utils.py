import numpy as np
from PIL import Image, ImageDraw, ImageFont
from itertools import chain, islice

def show(x):
    if hasattr(x, '__show__'):
        return x.__show__()
    return str(x)

class Vertical:
    def __init__(self, *elements):
        self.elements = elements

    def __str__(self):
        return 'Vertical\n  ' + "\n  ".join(x.replace('\n  ', '\n    ') for x in map(str, self.elements))

    def __show__(self):
        return '\n'.join(map(lambda x: show(x), self.elements))

class Horizontal:
    def __init__(self, *elements):
        self.elements = elements

    def __str__(self):
        return 'Horizontal\n  ' + "\n  ".join(x.replace('\n  ', '\n    ') for x in map(str, self.elements))

    def __show__(self):
        x = [show(x).split('\n') for x in self.elements]
        return '\n'.join('|'.join(x) for x in zip(*x))

class Tagged:
    def __init__(self, tag, number, origin=(0,0), fill='rgb(255,0,0)', font='/usr/share/fonts/TTF/Hack-Regular-Nerd-Font-Complete.ttf', fontsize=32):
        self.tag = tag
        self.number = number
        self.origin = origin
        self.fill = fill
        self.font = ImageFont.truetype(font, fontsize)

    def __str__(self):
        return 'Tagged({})\n  '.format(self.tag) + str(self.number).replace('\n  ', '\n    ')

    def __show__(self):
        return f'[{self.tag}]{show(self.number)}'

class Spacer:
    def __init__(self, space=3, value=255):
        self.space = space
        self.value = value

    def __str__(self):
        return f'Spacer({self.space} px)'

    def __show__(self):
        return f' '

def mkslice(cs, x):
    xs = [slice(None)] * cs
    xs[-1] = x
    return tuple(xs)

def mkselect(shape, x):
    xs = []
    for s in shape:
        xs.append(slice(0, s))

    xs.append(x)
    return tuple(xs)

def extend_to_shape(a, shape, v):
    shape = list(shape)
    for i in range(len(shape)):
        if i >= len(a.shape):
            break
        if i != v:
            shape[i] = a.shape[i]

    shape = tuple(shape)
    print(a.shape, '->', shape)
    new_a = np.ones(shape, dtype=np.uint8) * 255
    for i in range(shape[-1]):
        slc = mkselect(a.shape, i)
        new_a[slc] = a
    return new_a


def coerce_resize(a, b, v):
    if len(a.shape) > len(b.shape):
        return coerce_resize(a, extend_to_shape(b, a.shape, v), v)

    if len(a.shape) < len(b.shape):
        return coerce_resize(extend_to_shape(a, b.shape, v), b, v)

    if a.shape[v] == b.shape[v]:
        return a, b

    if a.shape[v] > b.shape[v]:
        shape = [*b.shape]
        shape[v] = a.shape[v]
        shape = tuple(shape)
        new_b = np.ones(shape, dtype=np.uint8) * 255
        start_x = a.shape[v] // 2 - b.shape[v] // 2
        shape = [slice(None)] * len(shape)
        shape[v] = slice(start_x, start_x + b.shape[v])
        shape = tuple(shape)
        new_b[shape] = b
        return a, new_b
    else:
        shape = [*a.shape]
        shape[v] = b.shape[v]
        shape = tuple(shape)
        new_a = np.ones(shape, dtype=np.uint8) * 255
        start_x = b.shape[v] // 2 - a.shape[v] // 2
        shape = [slice(None)] * len(shape)
        shape[v] = slice(start_x, start_x + a.shape[v])
        shape = tuple(shape)
        new_a[shape] = a
        print(new_a.shape, a.shape)
        return new_a,b

def side_by_side(order, *numbers):
    return _side_by_side([order], numbers, 1)

def _side_by_side(orders, numbers, alignment):
    stack = None
    for order in orders:
        if isinstance(order, Vertical):
            sbs = _side_by_side(order.elements, numbers, 1)
            if stack is not None:
                stack, sbs = coerce_resize(stack, sbs, alignment)
                if alignment == 1:
                    stack = np.vstack((stack, sbs))
                else:
                    stack = np.hstack((stack, sbs))
            else:
                stack = sbs
        elif isinstance(order, Horizontal):
            sbs = _side_by_side(order.elements, numbers, 0)
            if stack is not None:
                stack, sbs = coerce_resize(stack, sbs, alignment)
                if alignment == 1:
                    stack = np.vstack((stack, sbs))
                else:
                    stack = np.hstack((stack, sbs))
            else:
                stack = sbs
        elif isinstance(order, Spacer):
            shape = [order.space, order.space]
            if stack is not None:
                shape = [*stack.shape]
                shape[0] = order.space
                shape[1] = order.space

            sbs = np.ones(shape) * order.value
            if stack is not None:
                stack, sbs = coerce_resize(stack, sbs, alignment)
                if alignment == 1:
                    stack = np.vstack((stack, sbs))
                else:
                    stack = np.hstack((stack, sbs))
            else:
                stack = sbs
        elif isinstance(order, list):
            mkidx = lambda x: tuple([*numbers[order[0]].shape, x])
            shape = mkidx(len(order))
            sbs = np.zeros(shape, dtype=np.uint8)
            for e,i in enumerate(order):
                xslice = mkslice(len(sbs.shape), e)
                sbs[xslice] = numbers[i]

            if stack is not None:
                stack, sbs = coerce_resize(stack, sbs, 1)
                if alignment == 1:
                    stack = np.vstack((stack, sbs))
                else:
                    stack = np.hstack((stack, sbs))
            else:
                stack = sbs
        elif isinstance(order, Tagged):
            sbs = _side_by_side([order.number], numbers, alignment)
            image = Image.fromarray(sbs)
            draw = ImageDraw.Draw(image)
            draw.text(order.origin, order.tag, fill=order.fill, font=order.font)
            sbs = np.array(image)
            if stack is not None:
                stack, sbs = coerce_resize(stack, sbs, alignment)
                if alignment == 1:
                    stack = np.vstack((stack, sbs))
                else:
                    stack = np.hstack((stack, sbs))
            else:
                stack = sbs
        else:
            sbs = numbers[order]
            if stack is not None:
                stack, sbs = coerce_resize(stack, sbs, 0)
                if alignment == 1:
                    stack = np.vstack((stack, sbs))
                else:
                    stack = np.hstack((stack, sbs))
            else:
                stack = sbs
    return stack

def layout_with_names(elements, layout):
    return layout(*[Tagged(name, Vertical(Spacer(40), value)) for name,value in elements.items()])

def chunks(iterable, size):
    iterator = iter(iterable)
    for first in iterator:
        yield chain([first], islice(iterator, size - 1))

def nxm_matrix_view(indices, names, n, m):
    size = n * m
    layouts = []
    for view in chunks(indices, size):
        v_list = []
        for layout in map(list, chunks(view, m)):
            v_list.append(Horizontal(*layout))
        layouts.append(Vertical(*v_list))
    return layout_with_names({ name:layout for name, layout in zip(names, layouts) }, Vertical)

def nxn_matrix_view(indices, names, n):
    return nxm_matrix_view(indices, names, n, n)

def wait_for_key(key, waitkey, callback):
    while waitkey() & 0xff != ord(key):
        pass
    callback()

def empty(shape, value = 255):
    img = np.zeros(shape, np.uint8)
    img[mkselect(shape, 0)[:-1]] = value
    return img

def render_function(fn, shape):
    image = Image.fromarray(empty(shape))
    draw = ImageDraw.Draw(image)
    draw.line([(x, shape[1] - fn(x)) for x in range(0, shape[0])], fill=0, width=1, joint='curve')
    return np.array(image).astype(np.uint8)
