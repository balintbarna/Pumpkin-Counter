import constants as cn
import math

def calculate_gsd(img, height_meters, fov_deg):
    pixels = img.shape
    pwidth = pixels[1]
    pheight = pixels[0]
    print("resolution: "+str(pwidth)+"x"+str(pheight))
    alpha = math.radians(fov_deg / 2)
    width = height_meters * 2 * math.tan(alpha)
    ratio = width / pwidth
    height = ratio * pheight
    ratio *= 1000 # convert to mm/pixel
    print("image_width="+str(width)+" m; image_height="+str(height)+" m; image_ratio="+str(ratio)+" mm/pixel")
    return ratio

if __name__ == '__main__':
    gsd = calculate_gsd(cn.original_img, cn.height_meters, cn.fov)