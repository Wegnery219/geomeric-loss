import os
import numpy as np
import re
import pickle
import torch
import cv2


def preprocess(X):
    X = cv2.imread(X)
    X = cv2.resize(X, (224, 224))
    X = X.astype('float64')
    X -= np.mean(X)
    X = X.transpose((2, 0, 1))
    X = torch.from_numpy(X)
    return X


# Set this path to your dataset diectory
def get_data():
    directory = 'data/'
    directory_lis = os.listdir(directory)
    data_img = []
    data_pose = []
    for i in range(5):
        tr_dir = directory_lis[i]
        img = [f for f in os.listdir(os.path.join(directory, tr_dir)) if f.endswith("rgb.png")]
        for j in range(len(img)):
            imgpath = os.path.join(directory, tr_dir, img[j])
            imgdata = preprocess(imgpath)
            data_img.append(imgdata)
            number = re.findall(r"\d+\.?\d*", img[j])
            path = directory+'/'+tr_dir + '/' + number[0] + "_RT.pkl"
            f = open(path, 'rb')
            d = pickle.load(f)
            a = d[:3, :3]
            c = a[(0, 0)]+a[(1, 1)]+a[(2, 2)]
            q0 = np.sqrt(np.abs(c))
            q1 = (a[(2, 1)]-a[(1, 2)])/(4*q0)
            q2 = (a[(0, 2)]-a[(2, 0)])/(4*q0)
            q3 = (a[(1, 0)]-a[(0, 1)])/(4*q0)
            q = [q0, q1, q2, q3, d[(0, 3)], d[(1, 3)], d[(2, 3)]]
            data_pose.append(q)
    return data_img, data_pose






