import cv2
import dlib
import numpy as np
import skimage.draw
import matplotlib.pyplot as plt

from param import *
from utils.preprocessing.align_dlib import AlignDlib

align_dlib = AlignDlib(landmark_path)
detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(landmark_path)

def plot_image(img):
    plt.figure()
    plt.imshow(img)
    plt.show()
    return

def load_image(img, *args):
    img_color = cv2.imread(str(Path(img_path, *args, img)))
    img_plot  = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    img_bw    = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)
    return img_color, img_bw, img_plot

def save_image(img, fname, model):
    file = str(Path(img_path, model, fname))
    if model == '01_image':
        cv2.imwrite(file, img)
    elif model == '02_image_bw':
        cv2.imwrite(file, img)
    elif model == '03_cropped':
        cv2.imwrite(file, img)
    elif model == '04_cropped_bw':
        cv2.imwrite(file, img)
    elif model == '05_cropped_nobackground':
        cv2.imwrite(file, img)
    else:
        file = str(Path(img_path, '06_cropped_bw_nobackground', fname))
        cv2.imwrite(file, img)
    return

def crop_face(img):
    bounding_box = align_dlib.getLargestFaceBoundingBox(img)
    aligned = align_dlib.align(crop_dim, img, bounding_box, landmarkIndices=AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
    return aligned

def remove_background(img):
    rect = detector(img)[0]
    sp = predictor(img, rect)
    landmarks = np.array([[p.x, p.y] for p in sp.parts()])

    outline = landmarks[[*range(17), *range(26,16,-1)]]

    Y, X = skimage.draw.polygon(outline[:,1], outline[:,0])

    cropped_img = np.zeros(img.shape, dtype=np.uint8)
    cropped_img[Y, X] = img[Y, X]
    return cropped_img


def variance_of_laplacian(img):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(img, cv2.CV_64F).var()






