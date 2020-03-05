#!/usr/bin/Python
"""
    description here
    
    Author:     bakuv
    Date:       
"""

###############################################################
# Imports

import cv2
# import constants as cn

###############################################################
# Methods

def median(img, ksize):
    """
    ksize should be uneven, e.g.: 9
    """
    return cv2.medianBlur(img, ksize)

###############################################################
# Main

if __name__ == '__main__':
    img = cv2.imread("ex2_bgr_inrange.png")
    blurred = median(img, 9)

    cv2.imwrite("blurred.png", blurred)