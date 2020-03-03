#!/usr/bin/Python
"""
    Four different methods for segmenting image.
    
    Author:     frnyb
    Date:       20201802
"""

###############################################################
# Imports

import cv2
import numpy as np

from math import sqrt

from config_loader import load_config

###############################################################
# Methods

def segment_bgr(
    bgr_img,
    bgr_lowerb, 
    bgr_upperb
):
    seg_img = cv2.inRange(
        bgr_img, 
        bgr_lowerb, 
        bgr_upperb
    )

    return seg_img

def segment_cielab(
    bgr_img, 
    lab_lowerb, 
    lab_upperb
):
    lab_img = cv2.cvtColor(
        bgr_img, 
        cv2.COLOR_BGR2LAB
    )

    seg_img = cv2.inRange(
        lab_img, 
        lab_lowerb, 
        lab_upperb
    )

    return seg_img

def segment_hist_backpr(
    targ_img_bgr, 
    obj_img_bgr
):
    targ_img_hsv = cv2.cvtColor(
        targ_img_bgr, 
        cv2.COLOR_BGR2HSV
    )
    obj_img_hsv = cv2.cvtColor(
        obj_img_bgr, 
        cv2.COLOR_BGR2HSV
    )

    obj_hist = cv2.calcHist(
        [obj_img_hsv], 
        [0, 1], 
        None, 
        [180, 256], 
        [0, 180, 0, 256]
    )

    cv2.normalize(
        obj_hist,
        obj_hist, 
        0, 
        255, 
        cv2.NORM_MINMAX
    )
    dst = cv2.calcBackProject(
        [targ_img_hsv], 
        [0, 1], 
        obj_hist, 
        [0, 180, 0, 256], 
        1
    )

    disc = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, 
        (5, 5)
    )
    cv2.filter2D(
        dst, 
        -1,
        disc, 
        dst
    )

    return dst

def segment_dist(
    img_bgr, 
    bgr_val
):
    img_sub = np.zeros(img_bgr.shape)
    img_sub[:, :, 0] = np.subtract(img_bgr[:, :, 0], bgr_val[0])
    img_sub[:, :, 1] = np.subtract(img_bgr[:, :, 1], bgr_val[1])
    img_sub[:, :, 2] = np.subtract(img_bgr[:, :, 2], bgr_val[2])

    img_dist = np.sqrt(np.multiply(img_sub[:, :, 0], img_sub[:, :, 0]) + np.multiply(img_sub[:, :, 1], img_sub[:, :, 1]) + np.multiply(img_sub[:, :, 2], img_sub[:, :, 2]))

    img_dist = np.divide(img_dist, img_dist.max())

    img_dist = np.subtract(1, img_dist)

    img_dist = np.multiply(img_dist, 255)

    img_dist = img_dist.astype(int)

    return img_dist

def calc_lowerb(
    mean, 
    stdev, 
    interval_scalar
):
    return (mean[0] - interval_scalar * stdev[0], mean[1] - interval_scalar * stdev[1], mean[2] - interval_scalar * stdev[2])

def calc_upperb(
    mean, 
    stdev, 
    interval_scalar
):
    return (mean[0] + interval_scalar * stdev[0], mean[1] + interval_scalar * stdev[1], mean[2] + interval_scalar * stdev[2])

###############################################################
# Main

if __name__ == "__main__":
    config = load_config()

    img = cv2.imread(config['img_filename'])

    bgr_props = config['bgr_properties']
    bgr_mean = (bgr_props['b_mean'], bgr_props['g_mean'], bgr_props['r_mean'])
    bgr_stdev = (bgr_props['b_stdev'], bgr_props['g_stdev'], bgr_props['r_stdev'])
    
    # BGR segmentation
    img_seg_bgr = segment_bgr(
        img, 
        calc_lowerb(
            bgr_mean, 
            bgr_stdev, 
            bgr_props['lowerb_stdev_scalar']
        ), calc_upperb(
            bgr_mean, 
            bgr_stdev, 
            bgr_props['upperb_stdev_scalar']
        )
    )
    cv2.imwrite(
        "ex2_bgr_inrange.png", 
        img_seg_bgr
    )


    # LAB segmentation
    lab_props = config['lab_properties']
    lab_mean = (lab_props['l_mean'], lab_props['a_mean'], lab_props['b_mean'])
    lab_stdev = (lab_props['l_stdev'], lab_props['a_stdev'], lab_props['b_stdev'])

    img_seg_lab = segment_cielab(
        img, 
        calc_lowerb(
            lab_mean, 
            lab_stdev, 
            lab_props['lowerb_stdev_scalar']
        ), calc_upperb(
            lab_mean, 
            lab_stdev, 
            lab_props['upperb_stdev_scalar']
        )
    )
    cv2.imwrite(
        "ex2_lab_inrange.png", 
        img_seg_lab
    )


    # Hist backprj sementation
    img_obj = cv2.imread("histbckprj_object.png")
    img_seg_hist = segment_hist_backpr(
        img, 
        img_obj
    )
    ret, thresh_hist = cv2.threshold(
        img_seg_hist, 
        230, 
        255, 
        cv2.THRESH_BINARY
    )
    cv2.imwrite(
        "ex2_bgr_histbackprj.png", 
        thresh_hist
    )

    # BGR distance from mean segmentation
    img_seg_bgr_dist = segment_dist(
        img, 
        bgr_mean
    )
    cv2.imwrite(
        "ex2_bgr_dist.png", 
        img_seg_bgr_dist
    )