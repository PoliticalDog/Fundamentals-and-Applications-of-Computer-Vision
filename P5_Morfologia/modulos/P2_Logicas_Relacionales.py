import os
import cv2
import numpy as np

# ===== Utilidades (lógica pura) =====
def ensure_same_size(a, b):
    if a is None or b is None:
        return a, b
    ha, wa = a.shape[:2]; hb, wb = b.shape[:2]
    if (ha, wa) != (hb, wb):
        b = cv2.resize(b, (wa, ha), interpolation=cv2.INTER_NEAREST)
    return a, b

def to_binary(img, thresh=127, use_otsu=False, invert=False):
    if img is None:
        return None
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    ttype = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    if use_otsu:
        _, binarized = cv2.threshold(gray, 0, 255, ttype + cv2.THRESH_OTSU)
    else:
        _, binarized = cv2.threshold(gray, int(thresh), 255, ttype)
    return binarized

# ===== Operaciones lógicas =====
def op_and(A, B): return cv2.bitwise_and(A, B)
def op_or(A, B):  return cv2.bitwise_or(A, B)
def op_xor(A, B): return cv2.bitwise_xor(A, B)
def op_not(A):    return cv2.bitwise_not(A)

# ===== Operaciones relacionales (salida 0/255) =====
def _bin_compare(mask): return np.where(mask, 255, 0).astype(np.uint8)
def op_gt(A, B):  return _bin_compare(A >  B)
def op_lt(A, B):  return _bin_compare(A <  B)
def op_eq(A, B):  return _bin_compare(A == B)
def op_neq(A, B): return _bin_compare(A != B)
