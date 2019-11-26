import os
import numpy as np
import cv2
from colordetect.cnn_model import CNNDetectorInference, get_inverse_colormap, ConvolutionalDetector, Lenet5Module
import torch

if __name__ == '__main__':

    test_files = [x for x in os.listdir('test') if x.lower().endswith('.jpeg') and not x.startswith('.')]
    test_files = sorted([os.path.join('test', x) for x in test_files])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cnn_model = CNNDetectorInference('./models/best_model', device)

    predicts = []
    for f in test_files:
        img = cv2.imread(f)
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
        predict = cnn_model.predict(feature_clip)
    #     print(predict)
        predicts.append(predict.tolist())

    colormap_inv = get_inverse_colormap()
    print(test_files)
    for x in predicts:
        predict_color = [colormap_inv[y] for y in x]
        print(predict_color)
