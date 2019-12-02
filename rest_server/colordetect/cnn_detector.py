import os
import numpy as np
import cv2
from colordetect.cnn_model import CNNDetectorInference, get_inverse_colormap, ConvolutionalDetector, Lenet5Module
import torch

class CNNModel():

    def __init__(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cnn_model = CNNDetectorInference('./models/best_model', device)
        self._cm_inv = get_inverse_colormap()

    def predict(self, img):
        feature_clip = []
        h, w, c = img.shape
        h0 = h // 3
        w0 = w // 3
        for i in range(0, 3):
            for j in range(0, 3):
                sub_img = img[i * h0: (i + 1) * h0, j * w0: (j + 1) * w0, :]
                feature_clip.append(sub_img)
        feature_clip = np.stack(feature_clip, axis=0)
        #     feature_clip_standard = scaler.transform(feature_clip)
        predict = self.cnn_model.predict(feature_clip)
        predict_colors = [self._cm_inv[y] for y in predict]
        return predict_colors


if __name__ == '__main__':
    img = cv2.imread('./test/1.jpeg')
    model = CNNModel()
    print(model.predict(img))
