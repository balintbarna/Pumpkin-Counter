Exercise 3 - Counting blobs
___________________________

In this exercise the goal was to use the segmented image to count the number of pumpkins.
We used openCV's contour detector functions to find closed contours.
This leaves us with many overlapping results.

We need to find a criteria or a method to filter the undesirable matches.
A simple method we tried was to use the average pumpkin diameter and disregard those mathes
that have a distance to a unique match less than the average diameter.

INSERT IMAGE HERE

This method disregards a lot of real matches, and also leaves some false double matches on large targets.

The next method we tried is clustering based on a hierarchical search.
The algorithm stops when the shortest distance between two clusters is more than the usual pumpkin diameter.
Resulst can be seen on

INSERT IMAGE HERE

DRAW CONCLUSION ABOUT DIFFERENCES BETWEEN THE TWO METHODS
