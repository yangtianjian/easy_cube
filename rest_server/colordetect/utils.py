import cv2
import numpy as np


def clip_blocks(img):
    feature_clip = []
    h, w, c = img.shape
    h0 = h // 3
    w0 = w // 3
    for i in range(0, 3):
        for j in range(0, 3):
            sub_img = img[i * h0: (i + 1) * h0, j * w0: (j + 1) * w0, :]
            feature_clip.append(sub_img)
    feature_clip = np.stack(feature_clip, axis=0)
    return feature_clip