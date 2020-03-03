#!/usr/bin/Python
# Author:   frnyb
# Date:     20202102
# Function: Exposes method for highlighting contours in an original image.

###############################################################
# Imports

import cv2
from copy import deepcopy

###############################################################
# Methods

def highlight_contours(img, cnts, color, pumpkin_diameter):
    img = deepcopy(img)
    img = cv2.drawContours(img, cnts, -1, (255, 255, 255), -1)

    for c in cnts:
        try:
            m = cv2.moments(c)
            cx = int(m['m10'] / m['m00'])
            cy = int(m['m01'] / m['m00'])

            img = cv2.circle(img, (cx, cy), int(pumpkin_diameter / 2), (255, 0, 0), 1)
        except Exception:
            pass

    return img