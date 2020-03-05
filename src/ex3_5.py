#!/usr/bin/Python
"""
    Script containing methods and main executable for exercises
    3 and 5, Comparison of counting pumpkins in a filtered segmented 
    image and a unfiltered segmented image.
    
    Author:     frnyb
    Date:       20202102
"""

###############################################################
# Imports

from math import sqrt, inf, factorial
import operator
from copy import deepcopy
from tqdm import tqdm

import cv2
import numpy as np

from config_loader import load_config
from contours import Contour, ContourCluster, cluster_contours
from ex2 import segment_bgr, calc_lowerb, calc_upperb
from ex4 import median

###############################################################
# Methods

def count_pumpkins_simple(
    img,
    contours,
    pumpkin_diameter
):
    img = deepcopy(img)

    counted_pumpkins = []

    for cnt in tqdm(contours):
        is_unique = True
        for ccnt in counted_pumpkins:
            if (cnt.distance_to(ccnt) < pumpkin_diameter):
                is_unique = False
                break

        if is_unique:
            counted_pumpkins.append(cnt)
            img = cv2.circle(
                img, 
                tuple(cnt.center), 
                int(pumpkin_diameter / 2), 
                (255, 0, 0), 
                1)

    return len(counted_pumpkins), img

def get_contours(filtered_img):
    _, cv2_cnts, _ = cv2.findContours(
        filtered_img, 
        cv2.RETR_TREE, 
        cv2.CHAIN_APPROX_SIMPLE
    )

    cnts = []

    for cnt in cv2_cnts:
        try:
            cnts.append(Contour(cnt))
        except Exception:
            pass

    return cnts

def mark_contours(
    img,
    contours,
    diameter
):
    img = deepcopy(img)

    for cnt in contours:
        img = cv2.circle(
            img, 
            tuple(cnt.center), 
            int(diameter / 2), 
            (255, 0, 0), 
            1)

    return img

def count_pumpkins_long(
    img,
    contours,
    pumpkin_diameter
):
    clusters = cluster_contours(
        contours,
        pumpkin_diameter
    )

    img_marked_long = deepcopy(img)

    n_pumpkins_estimate = 0

    for cl in clusters:
        n_pumpkins_estimate += cl.contained_pumpkins_estimate(pumpkin_diameter)

        center, radius = cl.get_circle(pumpkin_diameter)
        img_marked_long = cv2.circle(
            img_marked_long, 
            tuple(center), 
            radius, 
            (255, 0, 0), 
            1
        )
        img_marked_long = cv2.putText(
            img_marked_long, 
            str(cl.contained_pumpkins_estimate(pumpkin_diameter)), 
            tuple(center), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 0, 0),
            thickness=3
        )

        for cnt in cl.contours:
            img_marked_long = cv2.circle(
                img_marked_long, 
                tuple(cnt.center), 
                int(pumpkin_diameter / 2), 
                (0, 0, 255), 
                1)

    return n_pumpkins_estimate, img_marked_long



###############################################################
# Main

if __name__ == "__main__":
    config = load_config()

    img = cv2.imread("../input/DJI_0240.JPG")

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

    # Filtering
    img_blur = cv2.medianBlur(img_seg_bgr, 5)

    # Obtaining contours
    contours = get_contours(img_blur)

    marked_contours = mark_contours(
        img,
        contours,
        config['pumpkin_diameter']
    )

    cv2.imwrite(
        "../output/ex3_5_marked_contours_all.png",
        marked_contours
    )

    # Counting pumpkins simple
    n_pumpkins_simple, marked_pumpkins_simple = count_pumpkins_simple(
        img,
        contours,
        config['pumpkin_diameter']
    )

    print("Number of pumpkins from simple counting method: " + str(n_pumpkins_simple))

    cv2.imwrite(
        "../output/ex3_5_marked_pumpkins_simple.png",
        marked_pumpkins_simple
    )

    # Counting pumpkins long
    n_pumpkins_long, marked_pumpkins_long = count_pumpkins_long(
        img,
        contours,
        config['pumpkin_diameter']
    )

    print("Number of pumpkins from long counting method: " + str(n_pumpkins_long))

    cv2.imwrite(
        "../output/ex3_5_marked_pumpkins_long.png",
        marked_pumpkins_long
    )



