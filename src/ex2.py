#!/usr/bin/Python
# Author:   frnyb
# Date:     20201802

import cv2
import numpy as np

from math import sqrt

IMG_FILENAME = "../input/DJI_0237.JPG"

BGR_MEAN = (1, 2, 3)
BGR_STDEV = (1, 2, 3)

CIELAB_MEAN = (1, 2, 3)
CIELAB_STDEV = (1, 2, 3)

def segment_bgr(bgr_img, bgr_lowerb, bgr_upperb):
    seg_img = cv2.inRange(bgr_img, bgr_lowerb, bgr_upperb)

    return seg_img

def segment_cielab(bgr_img, lab_lowerb, lab_upperb):
    lab_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)

    seg_img = cv2.inRange(lab_img, lab_lowerb, lab_upperb)

    return seg_img

def segment_hist_backpr(targ_img_bgr, obj_img_bgr, threshold):
    targ_img_hsv = cv2.cvtColor(targ_img_bgr, cv2.COLOR_BGR2HSV)
    obj_img_hsv = cv2.cvtColor(obj_img_bgr, cv2.COLOR_BGR2HSV)

    targ_hist = cv2.calcHist([targ_img_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    obj_hist = cv2.obj_histcalcHist([obj_img_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

    cv2.normalize(obj_hist, obj_hist, 0, 255, cv2.NORM_MINMAX)
    dst = cv2.calcBackProject([targ_img_hsv], [0, 1], obj_hist, [0, 180, 0, 256], 1)

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cv2.filter2D(dst, -1, disc, dst)

    ret, thresh = cv2.threshold(dst, 50, 255, 0)
    thresh = cv2.merge((thresh, thresh, thresh))
    res = cv2.bitwise_and(targ_img_bgr, thresh)
    res = np.vstack((targ_img_bgr, thresh, res))

    return dst

def segment_dist(img_bgr, bgr_val):
    img_sub = np.zeros(img_bgr.shape)
    img_sub[:, :, 0] = np.subtract(img_bgr[:, :, 0], bgr_val[0])
    img_sub[:, :, 1] = np.subtract(img_bgr[:, :, 1], bgr_val[1])
    img_sub[:, :, 2] = np.subtract(img_bgr[:, :, 2], bgr_val[2])

    img_dist = np.sqrt(img_sub[:, :, 0]**2 + img_sub[:, :, 1]**2 + img_sub[:, :, 2]**2)

    img_dist = np.divide(img_dist, img_dist.max())

    img_dist = np.subtract(1, img_dist)

    img_dist = np.multiply(img_dist, 255)

    return img_dist

if __name__ == "__main__":
    img = cv2.imread(IMG_FILENAME)

    bgr_lowerb = (BGR_MEAN[0] - BGR_STDEV[0], BGR_MEAN[1] - BGR_STDEV[1], BGR_MEAN[2] - BGR_STDEV[2])
    bgr_upperb = (BGR_MEAN[0] + BGR_STDEV[0], BGR_MEAN[1] + BGR_STDEV[1], BGR_MEAN[2] + BGR_STDEV[2])
    img_seg_bgr = segment_bgr(img, bgr_lowerb, bgr_upperb)

    lab_lowerb = (CIELAB_MEAN[0] - CIELAB_STDEV[0], CIELAB_MEAN[1] - CIELAB_STDEV[1], CIELAB_MEAN[2] - CIELAB_STDEV[2])
    lab_upperb = (CIELAB_MEAN[0] + CIELAB_STDEV[0], CIELAB_MEAN[1] + CIELAB_STDEV[1], CIELAB_MEAN[2] + CIELAB_STDEV[2])
    img_seg_lab = segment_bgr(img, lab_lowerb, lab_upperb)

    # img_seg_hist = segment_hist_backpr()

    img_seg_dist = segment_dist(img, BGR_MEAN)
