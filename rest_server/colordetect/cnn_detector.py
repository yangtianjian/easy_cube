import os
import numpy as np
import cv2
from colordetect.cnn_model import CNNDetectorInference, get_cnn_inverse_colormap, ConvolutionalDetector, Lenet5Module
import torch
from colordetect.utils import clip_blocks

class CNNModel():

    def __init__(self, model_path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cnn_model = CNNDetectorInference(model_path, device)
        self._cm_inv = get_cnn_inverse_colormap()

    def predict(self, img):
        feature_clip = clip_blocks(img)
        predict = self.cnn_model.predict(feature_clip)
        predict_colors = [self._cm_inv[y] for y in predict]
        return predict_colors