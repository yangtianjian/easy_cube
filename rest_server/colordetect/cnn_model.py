from sklearn.base import BaseEstimator, TransformerMixin
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
import cv2
import pandas as pd
import argparse
from colordetect.prepare_data import make_dataset
from sklearn.model_selection import GridSearchCV, ParameterGrid
import joblib
import os

from sklearn.tree import DecisionTreeClassifier


def _cal_size(i, p, k, st):
    return (i + 2 * p - k) // st + 1


def _create_preprocess_fn(d):
    def preprocess_fn(img0):
        img0 = cv2.resize(img0, (d, d), interpolation=cv2.INTER_AREA)
        return img0
    return preprocess_fn


def get_colormap():
    return {'W': 0, 'R': 1, 'G': 2, 'Y': 3, 'O': 4, 'B': 5}


def get_inverse_colormap():
    return dict([(v, k) for (k, v) in get_colormap().items()])


class Lenet5Module(nn.Module):

    def __init__(self, c, s):
        super(Lenet5Module, self).__init__()

        # --------------------hyper parameters ----------------------
        conv1_in_chn = c
        conv1_out_chn = 3 * c
        conv1_kernel_size = 5
        conv1_stride = 1
        conv1_padding = 1

        pool1_padding = 0
        pool1_kernel_size = 2
        pool1_stride = 2

        conv2_out_chn = 16
        conv2_kernel_size = 5
        conv2_stride = 1
        conv2_padding = 0

        pool2_padding = 0
        pool2_kernel_size = 2
        pool2_stride = 2

        fc1_out = 120
        fc2_out = 84

        # ---------------------const hyperparameters-----------------
        category = 6

        # ---------------------derived parameters--------------------

        conv1_fm_size = _cal_size(s, conv1_padding, conv1_kernel_size, conv1_stride)
        pool1_fm_size = _cal_size(conv1_fm_size, pool1_padding, pool1_kernel_size, pool1_stride)
        conv2_fm_size = _cal_size(pool1_fm_size, conv2_padding, conv2_kernel_size, conv2_stride)
        pool2_fm_size = _cal_size(conv2_fm_size, pool2_padding, pool2_kernel_size, pool2_stride)

        self.conv1 = nn.Sequential(
            nn.Conv2d(c, conv1_out_chn, conv1_kernel_size, conv1_stride, conv1_padding),
            nn.ReLU(),
            nn.MaxPool2d(pool1_kernel_size, pool1_stride, pool1_padding)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(conv1_out_chn, conv2_out_chn, conv2_kernel_size, conv2_stride, conv2_padding),
            nn.ReLU(),
            nn.MaxPool2d(pool2_kernel_size, pool2_stride, pool2_padding)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(conv2_out_chn * pool2_fm_size * pool2_fm_size, fc1_out),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(fc1_out, fc2_out),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(fc2_out, category)

    def forward(self, X):
        X = self.conv1(X)
        X = self.conv2(X)
        X = X.view(X.size()[0], -1)
        X = self.fc1(X)
        X = self.fc2(X)
        X = self.fc3(X)
        return X


class ConvolutionalDetector(BaseEstimator, TransformerMixin):

    # The default setting is for opencv
    def __init__(self,
                 img_size=28,
                 lr=1e-4,
                 batch_size=20,
                 epochs=30,
                 weight_decay=0.01,
                 scale=True,
                 input_format='NHWC',
                 color_format='BGR',
                 device='cpu',
                 debug=False):
        super(self.__class__, self).__init__()

        self.state = False
        self.model = None
        self.input_format = input_format
        self.color_format = color_format
        self.device = device
        self.debug = debug
        self.img_size = img_size

        # The parameters
        self.scale = scale
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay

        self.loss_curve = []
        self.acc_curve = []

    @staticmethod
    def load(file_name, device):
        with open(file_name, 'rb') as f:
            model_ = joblib.load(f)
            model_.device = device
        return model_

    def _scale(self, X, is_train):
        if self.scale:
            X = X / 255.0
        return X

    def _transform_representation(self, X):
        '''
        The representation should be NCHW and color format is rgb
        '''
        if self.input_format == 'NHWC':
            X = np.transpose(X, (0, 3, 1, 2)).copy()
        if self.color_format == 'BGR':
            X = X[:, :, :, ::-1].copy()
        return X

    # Procedures before feeding data into network
    def _preprocess(self, X, y=None, is_train=False, output_tensor=True):
        X = self._scale(X, is_train)
        X = self._transform_representation(X)
        if output_tensor:
            X = torch.FloatTensor(X)
            if y is not None:
                y = torch.LongTensor(y)
        if y is None:
            return X
        else:
            return X, y

    # Procedures before returning data to the caller
    def _postprocess(self, X, y):
        pass

    def fit(self, X, y):

        X, y = self._preprocess(X, y, is_train=True)
        dataset = TensorDataset(X, y)

        model = Lenet5Module(3, X.shape[2])
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()

        model.to(self.device, non_blocking=False)
        # print("-----------model initialized-----------------")
        optimizer = optim.AdamW(model.parameters(), weight_decay=self.weight_decay, lr=self.lr, amsgrad=True)

        for e in range(self.epochs):
            avg_acc = 0.0
            iteration = 0
            for i, data in enumerate(loader):
                inputs, labels = data
                inputs = inputs.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs).cpu()
                loss = criterion(outputs, labels)
                predicts = torch.argmax(outputs, dim=1)
                avg_acc += int((predicts == labels).sum()) / float(labels.size()[0])
                loss.backward()
                optimizer.step()
                iteration += 1
                self.loss_curve.append(loss.item())
            avg_acc /= iteration
            self.acc_curve.append(avg_acc)

        model.to('cpu')
        self.model = model
        self.state = True

    def predict_proba(self, X):
        X = self._preprocess(X, is_train=False)
        dataset = TensorDataset(X)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        probs = []
        model = self.model

        model.to(self.device)
        with torch.no_grad():
            for i, data in enumerate(loader):
                inputs = data[0]
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                prob = F.softmax(outputs, dim=1)
                probs.append(prob.cpu().numpy())
            probs = np.concatenate(probs, axis=0)
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


def get_dataset_for_cnn(dim, preshuffle=True, limit=999999999):
    colormap2 = get_colormap()
    (X_img, y) = make_dataset(label_blocks, block_dir, preshuffle=preshuffle, limit=limit, preprocess_fn=_create_preprocess_fn(dim))
    X_img = np.stack(X_img, axis=0)
    y = np.array([colormap2[q['color']] for q in y])
    return X_img, y


class CNNDetectorInference(BaseEstimator, TransformerMixin):

    def __init__(self, model_file_name, device):
        self.model = ConvolutionalDetector.load(model_file_name, device)

    def predict(self, X):
        X = np.stack([_create_preprocess_fn(self.model.img_size)(X[i, :, :, :]) for i in range(len(X))], axis=0)
        return self.model.predict(X)

    def predict_proba(self, X):
        X = _create_preprocess_fn(self.model.img_size)(X)
        return self.model.predict(X)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-bl", "--block_label", default='./label/label_block.tsv')
    ap.add_argument("-cbd", "--clipped_block_dir", default='clipped_color_blocks')
    ap.add_argument("-dim", "--block_width", default=32)
    ap.add_argument('-md', "--model_dir", default='./models')
    ap.add_argument('-lc', "--loss_curve", default=True)

    args = vars(ap.parse_args())

    d = args['block_width']
    label_blocks = args['block_label']
    block_dir = args['clipped_block_dir']
    model_dir = args['model_dir']
    output_loss_curve = args['loss_curve']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    params_grid = {
        "lr": [1e-4],
        "batch_size": [10000],
        "epochs": [200],
        "weight_decay": [0.05, 0.07, 0.09, 0.1, 0.13]
    }

    print("Reading data into memory")
    X_img, y = get_dataset_for_cnn(d)
    # X_img, y = get_dataset_for_cnn(d, preshuffle=True, limit=100)  # debug
    print("Start to search")
    # We use grid search method and 5-fold cross validation to choose the most robust model
    model = GridSearchCV(ConvolutionalDetector(device=device, img_size=d), params_grid, cv=4, scoring='accuracy', n_jobs=1)
    model.fit(X_img, y)

    # Choose the best model and dump
    best_model = model.best_estimator_
    cv_results = model.cv_results_
    with open(os.path.join(model_dir, 'cv_result.txt'), 'w') as f:
        f.write(str(cv_results))
    with open(os.path.join(model_dir, 'best_model'), 'wb') as f:
        torch.save(best_model, f)
    print("Finish. Plotting the curve")

    plt.figure(figsize=(20, 10))
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.plot(best_model.loss_curve)
    plt.savefig(os.path.join(model_dir, 'best_model_loss_curve.png'))
    plt.close()

    plt.figure(figsize=(20, 10))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(best_model.acc_curve)
    plt.savefig(os.path.join(model_dir, 'best_model_acc_curve.png'))
    plt.close()

    print("All done.")
