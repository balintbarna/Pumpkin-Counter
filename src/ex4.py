import cv2

def median(img, ksize):
    return cv2.medianBlur(img, ksize)

