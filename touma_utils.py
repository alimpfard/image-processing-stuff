import numpy as np

def _AddGaussianNoiseToArray(array, mean, std):
    tmp = array+np.random.normal(mean, std, array.shape)
    tmpm = tmp.min()
    tmp = tmp-tmpm
    tmpm = tmp.max()
    if tmpm == 0:
        return np.uint8(tmp)
    return np.uint8((tmp*255/tmpm))

def AddGaussianNoise(img, mean, std, count=1):
    tmp= np.array(img, dtype=np.float32)
    if count<2:
        return _AddGaussianNoiseToArray(tmp, mean, std)
    else:
        ml=[]
        for i in range(count):
            ml.append(_AddGaussianNoiseToArray(tmp, mean, std))
        return ml

def AverageImage(inp, scale=True):
    tmp= None
    tmp=np.zeros(inp[0].shape, dtype=np.float32)
    for i in inp:
        tmp= tmp+i
    tmp=tmp/len(inp)
    if scale:
        tmpm= tmp.min()
        tmp= tmp-tmpm
        tmpm= tmp.max()
        if tmpm!=0:
            tmp=(tmp*255/tmpm)
    return np.uint8(tmp)
