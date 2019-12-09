import numpy as np
from scipy.linalg import norm
import cv2
from colordetect.utils import clip_blocks
import copy
from cube_solver.solver import get_inv_face_map, get_face_map

def knn_predict(image_json_dict):
    '''
    :param image_json_dict: {
        U: []
        R: [],
        ...
    }
    For each face, there are 9 images.
    The center of these faces must be:
    We suppose that the user selects correct face. And U[4], R[4], ..., D[4] is correct
    :return:
    '''
    def img_dist(x, y):
        hx, wx, _ = x.shape
        hy, wy, _ = y.shape
        x = x[int(0.4 * hx): int(0.6 * hx), int(0.4 * wx): int(0.6 * wx), :]
        y = y[int(0.4 * hy): int(0.6 * hy), int(0.4 * wy): int(0.6 * wy), :]
        hx, wx, _ = x.shape
        hy, wy, _ = y.shape
        x = np.mean(x.reshape((hx * wx, 3)), axis=0)
        y = np.mean(y.reshape((hy * wy, 3)), axis=0)
        return norm(x - y)

    image_json_dict = copy.deepcopy(image_json_dict)

    center_sample = {}
    for k in ['U', 'L', 'F', 'D', 'R', 'B']:
        cv2.imwrite(k + '_whole.png', image_json_dict[k])
        image_json_dict[k] = clip_blocks(image_json_dict[k])
        for i in range(9):
            cv2.imwrite(k + "_" + str(i) + ".png", image_json_dict[k][i])
        center_sample[k] = image_json_dict[k][4]
        cv2.imwrite(k + '_center.png', center_sample[k])

    fm = get_face_map()
    answers = {}
    for k in ['U', 'L', 'F', 'D', 'R', 'B']:
        blocks = image_json_dict[k]
        answer = []
        for i in range(len(blocks)):
            mind = 999999999
            argd = ''
            for k2 in ['U', 'L', 'F', 'D', 'R', 'B']:
                d = img_dist(blocks[i], center_sample[k2])
                if d < mind:
                    mind = d
                    argd = k2
            answer.append(fm[argd])
        answers[k] = answer
    return answers
