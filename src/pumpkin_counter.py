#!/usr/bin/Python
# Author:   frnyb, mgrov, bakuv
# Date:     20202102

import cv2
import json

class PumkinCounter():
    def __init__(self):
        pass

    def load_config(self, json_config_file = "config.json"):
        with open(json_config_file) as f:
            self.config = json.load(f)

        return self.config

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


if __name__ == "__main__":
    # main()

    # config = {}
    # config['img_filename'] = "../input/DJI_0237.JPG"
    # config['pumpkin_bgr_properties'] = {}

    # config['pumpkin_bgr_properties']['b_mean'] = "hej"
    # config['pumpkin_bgr_properties']['g_mean'] = "hej"
    # config['pumpkin_bgr_properties']['r_mean'] = "hej"

    # config['pumpkin_bgr_properties']['b_stdev'] = "hej"
    # config['pumpkin_bgr_properties']['g_stdev'] = "hej"
    # config['pumpkin_bgr_properties']['r_stdev'] = "hej"

    # config['pumpkin_lab_properties'] = {}
    # config['pumpkin_lab_properties']['l_mean'] = "hej"
    # config['pumpkin_lab_properties']['a_mean'] = "hej"
    # config['pumpkin_lab_properties']['b_mean'] = "hej"
    # config['pumpkin_lab_properties']['l_stdev'] = "hej"
    # config['pumpkin_lab_properties']['a_stdev'] = "hej"
    # config['pumpkin_lab_properties']['b_stdev'] = "hej"

    # with open("config.json", "w") as f:
    #     json.dump(config, f)


    pc = PumkinCounter()
    config = pc.load_config()

    print(config["img_filename"])
