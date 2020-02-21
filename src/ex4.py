import cv2
import constants as cn


def median(img, ksize):
    """
    ksize should be uneven, e.g.: 9
    """
    return cv2.medianBlur(img, ksize)


if __name__ == '__main__':
    blurred = median(cn.original_img, 9)
    cv2.imwrite("output/blurred.jpg", blurred)