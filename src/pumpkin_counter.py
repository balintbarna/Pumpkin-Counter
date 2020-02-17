import cv2

def main():
    filename = "../input/DJI_0237.JPG"
    img = cv2.imread(filename)

    cv2.circle(img, (300, 400), 50, (255, 255, 255), -5)
    cv2.circle(img, (800, 400), 50, (255, 255, 255), -5)
    cv2.circle(img, (550, 600), 50, (255, 255, 255), -5)
    cv2.circle(img, (550, 900), 50, (255, 255, 255), -5)
    cv2.circle(img, (450, 880), 50, (255, 255, 255), -5)
    cv2.circle(img, (650, 880), 50, (255, 255, 255), -5)
    cv2.circle(img, (750, 840), 50, (255, 255, 255), -5)
    cv2.circle(img, (350, 840), 50, (255, 255, 255), -5)

    output_filename = "../output/01_annotated_image.jpg"
    cv2.imwrite(output_filename, img)

    n_pumpkins = 2
    print("%d pumpkins were found" % n_pumpkins)


main()
