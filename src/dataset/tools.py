import numpy as np

def minmax2xywh(minmax:np.array):
    # cls, minx, miny, maxx, maxy
    xywh = np.zeros_like(minmax)
    xywh[:, 0] = minmax[:, 0]
    xywh[:, 1] = (minmax[:, 1] + minmax[:, 3]) / 2
    xywh[:, 2] = (minmax[:, 2] + minmax[:, 4]) / 2
    xywh[:, 3] = minmax[:, 3] - minmax[:, 1]
    xywh[:, 4] = minmax[:, 4] - minmax[:, 2]
    return xywh
    
def xywh2minmax(xywh:np.array):
    # cls, cx, cy, w, h
    minmax = np.zeros_like(xywh)
    minmax[:, 0] = xywh[:, 0]
    minmax[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    minmax[:, 2] = xywh[:, 2] - xywh[:, 4] / 2
    minmax[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
    minmax[:, 4] = xywh[:, 2] + xywh[:, 4] / 2
    return minmax

def normalize_target(target:np.array, tsize:tuple):
    width, height = tsize
    normed_target = np.zeros_like(target)
    normed_target[:, 0] = target[:, 0]
    normed_target[:, [1, 3]] = np.round(target[:, [1, 3]] / width, 3)
    normed_target[:, [2, 4]] = np.round(target[:, [2, 4]] / height, 3)
    return normed_target
