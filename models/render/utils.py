import unicodedata
import numpy as np
import cv2
from typing import List, Callable, Tuple

def is_whitespace(ch):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically control characters but we treat them
    # as whitespace since they are generally considered as such.
    if ch == " " or ch == "\t" or ch == "\n" or ch == "\r" or ord(ch) == 0:
        return True
    cat = unicodedata.category(ch)
    if cat == "Zs":
        return True
    return False

def is_punctuation(ch):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(ch)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
        (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(ch)
    if cat.startswith("P"):
        return True
    return False

def color_difference(rgb1: List, rgb2: List) -> float:
    # https://en.wikipedia.org/wiki/Color_difference#CIE76
    color1 = np.array(rgb1, dtype=np.uint8).reshape(1, 1, 3)
    color2 = np.array(rgb2, dtype=np.uint8).reshape(1, 1, 3)
    diff = cv2.cvtColor(color1, cv2.COLOR_RGB2LAB).astype(np.float32) - cv2.cvtColor(color2, cv2.COLOR_RGB2LAB).astype(np.float32)
    diff[..., 0] *= 0.392
    diff = np.linalg.norm(diff, axis=2) 
    return diff.item()

def rotate_polygons(center, polygons, rotation, new_center=None, to_int=True):
    if rotation == 0:
        return polygons
    if new_center is None:
        new_center = center
    rotation = np.deg2rad(rotation)
    s, c = np.sin(rotation), np.cos(rotation)
    polygons = polygons.astype(np.float32)

    polygons[:, 1::2] -= center[1]
    polygons[:, ::2] -= center[0]
    rotated = np.copy(polygons)
    rotated[:, 1::2] = polygons[:, 1::2] * c - polygons[:, ::2] * s
    rotated[:, ::2] = polygons[:, 1::2] * s + polygons[:, ::2] * c
    rotated[:, 1::2] += new_center[1]
    rotated[:, ::2] += new_center[0]
    if to_int:
        return rotated.astype(np.int64)
    return rotated