import cv2
import numpy as np
from scipy import ndimage

def to_binary(img, thresh=127, use_otsu=True, invert=False):
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    if use_otsu:
        _, binimg = cv2.threshold(gray, 0, 255, flag + cv2.THRESH_OTSU)
    else:
        _, binimg = cv2.threshold(gray, int(thresh), 255, flag)
    return binimg

def label_components(binary_0_255, connectivity=4):
    bin01 = (binary_0_255 > 0).astype(np.uint8)
    structure = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=int) if connectivity == 4 else np.ones((3,3), dtype=int)
    labels, num = ndimage.label(bin01, structure=structure)
    return labels, num

def colorize_labels(labels):
    labels = labels.astype(np.int32)
    k = labels.max()
    if k <= 0:
        return np.zeros((*labels.shape, 3), dtype=np.uint8)
    rng = np.random.default_rng(12345)
    palette = rng.integers(0, 255, size=(k+1, 3), dtype=np.uint8)
    palette[0] = 0
    return palette[labels]
