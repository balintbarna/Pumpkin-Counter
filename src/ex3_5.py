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

import ex2
from ex4 import median
from ex7 import highlight_contours
from config_loader import load_config

###############################################################
# Methods

# def find_pumpkins(thresh_img, pumpkin_diameter):
#     """
#         Returns contours of pumpkins given a thresholded image of the pumpkins.
#         Filters contours by closenes, so pumpkins aren't counted twice.
#     """
#     _, cnts, _ = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#     return contour_clustering(
#         cnts,
#         pumpkin_diameter
#     )

# def highlight_contours(
#     img, 
#     cnts, 
#     color, 
#     pumpkin_diameter
# ):
#     img = deepcopy(img)

#     for c in cnts:
#         img = cv2.circle(img, 
#         tuple(c.center), 
#         int(pumpkin_diameter / 2), 
#         color, 
#         1)

#     return img


###############################################################
# Main

if __name__ == "__main__":
    config = load_config()

    img = cv2.imread(config['img_filename'])

    bgr_props = config['bgr_properties']
    bgr_mean = (bgr_props['b_mean'], bgr_props['g_mean'], bgr_props['r_mean'])
    bgr_stdev = (bgr_props['b_stdev'], bgr_props['g_stdev'], bgr_props['r_stdev'])

    seg_img = ex2.segment_bgr(
        img, 
        ex2.calc_lowerb(
            bgr_mean, 
            bgr_stdev, 
            bgr_props['lowerb_stdev_scalar']
        ), ex2.calc_upperb(
            bgr_mean, 
            bgr_stdev, 
            bgr_props['upperb_stdev_scalar'])
        )

    # thresh_img = cv2.adaptiveThreshold(seg_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # cv2.imwrite("test.png", seg_img)

    # filtered_img = median(seg_img, 3)

    # pumpkin_dia = 20

    # cnt_clusters, distance_matrix = find_pumpkins(filtered_img, pumpkin_dia)

    # print("Drawing")
    # for cl in cnt_clusters:
    #     center, radius = cl.get_circle()
    #     img = cv2.circle(img, 
    #     tuple(center), 
    #     radius, 
    #     (255, 0, 0), 
    #     1)
    #     img = cv2.putText(img, str(cl.contained_pumpkins_estimate(pumpkin_dia)), tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

    #     for cnt in cl.contours:
    #         img = cv2.circle(img, 
    #             tuple(cnt.center), 
    #             int(pumpkin_dia / 2), 
    #             (0, 0, 255), 
    #             1)

    # cv2.imwrite("cluster_and_contours.png", img)

    # distance_matrix = distance_matrix.flatten()
    # distance_matrix = np.sort(distance_matrix)
    # distance_matrix = np.flip(distance_matrix)

    # for d in distance_matrix:
        # print(d)

    # img_cl = draw_pumpkin_clustering(img, cnts)
    # # img_cp = draw_pumpkin_clustering(img, cnts_cp)

    # img_cl = highlight_contours(img_cl, cnts, (255, 0, 0), pumpkin_dia)
    # # img_cp = highlight_contours(img_cp, cnts_cp, (255, 0, 0), pumpkin_dia)

    # cv2.imwrite("clustered.png", img_cl)
    # # cv2.imwrite("clustered_cp.png", img_cp)

    # # cv2.imwrite("marked_2.png", marked_img)




