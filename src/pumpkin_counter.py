#!/usr/bin/python3
"""
    Script containing the PumpkinCounter class as well as
    the main executable code for counting pumpkins in a
    picture of a pumpkin field.
    
    Author:     frnyb, mgrov, bakuv
    Date:       20202102
"""

###############################################################
# Imports

import cv2
import numpy as np

import sys
import argparse
from copy import deepcopy
from tqdm import tqdm

from contours import Contour, ContourCluster
from config_loader import load_config

###############################################################
# Classes

class PumpkinCounter():
    def __init__(
        self,
        img_path=None,
        silent=False
    ):
        self.config = load_config()

        if (img_path != None):
            self.img = cv2.imread(img_path)
        else:
            self.img = cv2.imread(self.config['img_filename'])

        self.silent = silent

        bgr_props = self.config['bgr_properties']
        bgr_mean = (bgr_props['b_mean'], bgr_props['g_mean'], bgr_props['r_mean'])
        bgr_stdev = (bgr_props['b_stdev'], bgr_props['g_stdev'], bgr_props['r_stdev'])

        self.seg_img = self._segment_bgr(
            self.img,
            self._calc_lowerb(
                bgr_mean,
                bgr_stdev,
                bgr_props['lowerb_stdev_scalar']
            ),
            self._calc_upperb(
                bgr_mean,
                bgr_stdev,
                bgr_props['upperb_stdev_scalar']
            )
        )

        self.filtered_img = self._median(
            self.seg_img,
            self.config['filter_ksize']
        )

        self.contours = self._get_contours(
            self.filtered_img
        )

    def count_pumpkins_fast(
        self,
        marked_img_filename=None
    ):
        self.counted_pumpkins_fast = []
        if (marked_img_filename != None):
            self.img_marked_fast = deepcopy(self.img)

        for cnt in tqdm(
            self.contours, 
            disable=self.silent
        ):
            is_unique = True
            for ccnt in self.counted_pumpkins_fast:
                if (cnt.distance_to(ccnt) < self.config['pumpkin_diameter']):
                    is_unique = False
                    break

            if is_unique:
                self.counted_pumpkins_fast.append(cnt)
                if marked_img_filename != None:
                    self.img_marked_fast = cv2.circle(
                        self.img_marked_fast, 
                        tuple(cnt.center), 
                        int(self.config['pumpkin_diameter'] / 2), 
                        (0, 0, 255), 
                        1)

        if marked_img_filename != None:
            cv2.imwrite(
                marked_img_filename,
                self.img_marked_fast
            )

        return len(self.counted_pumpkins_fast)

    def count_pumpkins_long(
        self,
        marked_img_filename=None
    ):
        clusters = self._cluster_contours()

        if marked_img_filename != None:
            self.img_marked_long = deepcopy(self.img)

        n_pumpkins_estimate = 0

        for cl in clusters:
            n_pumpkins_estimate += cl.contained_pumpkins_estimate(self.config['pumpkin_diameter'])

            if marked_img_filename != None:
                center, radius = cl.get_circle()
                self.img_marked_long = cv2.circle(
                    self.img_marked_long, 
                    tuple(center), 
                    radius, 
                    (255, 0, 0), 
                    1
                )
                self.img_marked_long = cv2.putText(
                    self.img_marked_long, 
                    str(cl.contained_pumpkins_estimate(self.config['pumpkin_diameter'])), 
                    tuple(center), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 0, 0),
                    thickness=3
                )

                for cnt in cl.contours:
                    self.img_marked_long = cv2.circle(
                        self.img_marked_long, 
                        tuple(cnt.center), 
                        int(self.config['pumpkin_diameter'] / 2), 
                        (0, 0, 255), 
                        1)

        if marked_img_filename != None:
            cv2.imwrite(
                marked_img_filename,
                self.img_marked_long
            )

        return n_pumpkins_estimate

    def _segment_bgr(
        self,
        bgr_img, 
        bgr_lowerb, 
        bgr_upperb
    ):
        seg_img = cv2.inRange(bgr_img, bgr_lowerb, bgr_upperb)

        thresh_img = cv2.adaptiveThreshold(
            seg_img, 
            255, 
            cv2.ADAPTIVE_THRESH_MEAN_C, 
            cv2.THRESH_BINARY, 
            11, 
            2
        )

        return thresh_img

    def _calc_lowerb(
        self,
        mean, 
        stdev, 
        interval_scalar
    ):
        return (mean[0] - interval_scalar * stdev[0], mean[1] - interval_scalar * stdev[1], mean[2] - interval_scalar * stdev[2])

    def _calc_upperb(
        self,
        mean, 
        stdev, 
        interval_scalar
    ):
        return (mean[0] + interval_scalar * stdev[0], mean[1] + interval_scalar * stdev[1], mean[2] + interval_scalar * stdev[2])

    def _median(
        self,
        img, 
        ksize
    ):
        return cv2.medianBlur(
            img, 
            ksize
        )

    def _get_contours(
        self,
        filtered_img
    ):
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

    def _cluster_contours(self):
        max_cluster_distance = self.config['pumpkin_diameter']

        contour_clusters = []
        for cnt in self.contours:
            contour_clusters.append(ContourCluster(cnt))

        distance_matrix = np.zeros((len(contour_clusters), len(contour_clusters)))

        if not self.silent:
            print("Creating distance matrix")
        for i in tqdm(
            range(len(contour_clusters)),
            disable=self.silent
        ):
            for j in range(len(contour_clusters)):
                if (j < i):
                    distance_matrix[i, j] = contour_clusters[i].distance_to_cluster(contour_clusters[j])
                else:
                    distance_matrix[i, j] = np.inf

        counter = 0
        if not self.silent:
            print("Clustering contours")
        while True:
            if not self.silent:
                print("At number " + str(counter + 1) + " out of " + str(len(contour_clusters)) + " remaining clusters.")
            counter += 1
            (i, j) = np.unravel_index(
                distance_matrix.argmin(),
                distance_matrix.shape
            )

            distance = distance_matrix[i, j]
            if not self.silent:
                print("Distance is " + str(distance) + " pixels")

            if (distance > max_cluster_distance):
                break

            merged_cluster = contour_clusters.pop(i).merge(contour_clusters.pop(j))
            contour_clusters.append(merged_cluster)

            distance_matrix = np.delete(
                distance_matrix,
                i,
                axis = 1
            )
            distance_matrix = np.delete(
                distance_matrix,
                j,
                axis = 1
            )
            row_i = distance_matrix[i, :]
            distance_matrix = np.delete(
                distance_matrix,
                i,
                axis = 0
            )
            row_j = distance_matrix[j, :]
            distance_matrix = np.delete(
                distance_matrix,
                j,
                axis = 0
            )

            new_column = np.full(
                len(contour_clusters),
                np.inf
            )
            new_row = np.minimum(
                row_i,
                row_j
            )

            distance_matrix_temp = np.zeros((len(contour_clusters), len(contour_clusters)))

            distance_matrix_temp[:-1, :-1] = distance_matrix
            distance_matrix_temp[-1, :-1] = new_row
            distance_matrix_temp[:, -1] = new_column

            distance_matrix = distance_matrix_temp

        self.contour_clusters = contour_clusters

        return self.contour_clusters

###############################################################
# Main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count pumpkins in a given image.")

    parser.add_argument(
        '-i',
        dest='i',
        action='store',
        help="The input image path. If non given, a default path will be used."
    )
    parser.add_argument(
        '-l',
        dest='l',
        action='store_true',
        default=False,
        help="Long count. Specify to count using the clustering method. Takes longer time."
    )
    parser.add_argument(
        '-o',
        dest='o',
        action='store',
        type=str,
        help="Output image path. If specified, will store an image with the counted pumpkins marked at this location."
    )
    parser.add_argument(
        '-s',
        dest='s',
        action='store_true',
        default=False,
        help="If specified, no loading bar will be shown."
    )

    args = parser.parse_args(sys.argv[1:])

    pc = None

    if args.i != None:
        pc = PumpkinCounter(
        args.i,
        silent=args.s
        )
    else:
        pc = PumpkinCounter(silent=args.s)

    n_pumpkins = 0

    if (args.l):
        if args.o != None:
            n_pumpkins = pc.count_pumpkins_long(args.o)
            cv2.imwrite(
                pc.img_marked_long,
                args.o
            )
        else:
            n_pumpkins = pc.count_pumpkins_long()
    else:
        if args.o != None:
            n_pumpkins = pc.count_pumpkins_fast(args.o)
            cv2.imwrite(
                pc.img_marked_fast,
                args.o
            )
        else:
            n_pumpkins = pc.count_pumpkins_fast()

    print(n_pumpkins)

    