"""
    Script containing classes Contour and ContourCluster.
    
    Author:     frnyb
    Date:       20200229
"""

###############################################################
# Imports

import cv2
import numpy as np

from math import sqrt #, inf
inf = float('inf')

###############################################################
# Classes

class Contour:
    """
        Class storing a OpenCV contour with information about the location
        of the contour center. Overloads operators for comparison, looks at
        distance from origin.
    """
    def __init__(
        self,
        cv2_contour
    ):
        self.contour = cv2_contour
        
        M = cv2.moments(self.contour)
        self.center = [int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])]
        self.distance_from_o = sqrt(self.center[0]**2 + self.center[1]**2)

    def distance_to(
        self,
        other=None,
        other_center=None
    ):
        dist = 0

        if (other is not None):
            dist = sqrt((self.center[0] - other.center[0])**2 + (self.center[1] - other.center[1])**2)
        else:
            dist =  sqrt((self.center[0] - other_center[0])**2 + (self.center[1] - other_center[1])**2)

        if dist == 0:
            return inf
        else:
            return dist

    def __gt__(
        self,
        other
    ):
        return self.distance_from_o > other.distance_from_o

    def __lt__(
        self,
        other
    ):
        return self.distance_from_o < other.distance_from_o

    def __eq__(
        self,
        other
    ):
        return self.distance_from_o == other.distance_from_o

    def __sub__(
        self,
        other
    ):
        return abs(self.distance_from_o - other.distance_from_o)

class ContourCluster:
    """
        Class representing a cluster of contours.
    """
    def __init__(
        self,
        contour
    ):
        self.contours = [contour]

    def distance_to_contour(
        self,
        contour
    ):
        min_distance = inf
        for cnt in self.contours:
            min_distance = min([min_distance, contour.distance_to(cnt)])
        return min_distance

    def distance_to_cluster(
        self,
        other
    ):
        min_distance = inf
        for cnt in self.contours:
            min_distance = min([min_distance, other.distance_to_contour(cnt)])
        return min_distance

    def merge(
        self,
        other
    ):
        self.contours = self.contours + other.contours
        return self

    def get_circle(self):
        avg_center = [0, 0]
        for cnt in self.contours:
            avg_center[0] += cnt.center[0]
            avg_center[1] += cnt.center[1]
        avg_center[0] = int(avg_center[0] / len(self.contours))
        avg_center[1] = int(avg_center[1] / len(self.contours))

        max_radius = 0
        for cnt in self.contours:
            max_radius = int(min([max_radius, cnt.distance_to(other_center=avg_center)]))

        return avg_center, max_radius

    def contained_pumpkins_estimate(
        self,
        diameter
    ):
        avg_dist = 0

        if len(self.contours) > 2:
            for i in range(len(self.contours)):
                avg_dist += min(
                    range(len(self.contours)),
                    key = lambda j: self.contours[i].distance_to(self.contours[j])
                )
            avg_dist /= len(self.contours)
        elif len(self.contours) == 2:
            avg_dist = self.contours[0].distance_to(self.contours[1])

        return int((avg_dist / diameter) * len(self.contours)) + 1