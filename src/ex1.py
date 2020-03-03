#!/usr/bin/python
# -*- coding: utf-8 -*-

# import libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
import statistics as st


def rgb_hist(r, g, b, filename):
	fig0, (ax00, ax01, ax02) = plt.subplots(nrows=3, constrained_layout=True)

	ax00.hist(r.flatten(), bins = 255, color = 'r')
	ax00.set_title('Red channel histogram')

	ax01.hist(g.flatten(), bins = 255, color = 'g')
	ax01.set_title('Green channel histogram')

	ax02.hist(b.flatten(), bins = 255, color = 'b')
	ax02.set_title('Blue channel histogram')
	filename = "../output/ex1/" + filename
	fig0.savefig(filename)
	plt.close()

def cieLab_hist(L, a, b, filename):
	fig0, (ax00, ax01, ax02) = plt.subplots(nrows=3, constrained_layout=True)

	ax00.hist(L.flatten(), bins = 255, color = 'r')
	ax00.set_title('Lightness channel histogram')

	ax01.hist(a.flatten(), bins = 255, color = 'g')
	ax01.set_title('Green to red channel histogram')

	ax02.hist(b.flatten(), bins = 255, color = 'b')
	ax02.set_title('Blue to yellow channel histogram')

	filename = "../output/ex1/" + filename
	fig0.savefig(filename)
	plt.close()	

def sub_div_image(img):
	rows = len(img[:,1])
	cols = len(img[1,:])

	img00 = img[0:int(rows/3),0:int((cols/4))]
	img01 = img[0:int(rows/3),int((cols/4)):int(2*(cols/4))]
	img02 = img[0:int(rows/3),int(2*(cols/4)):int(3*(cols/4))]
	img03 = img[0:int(rows/3),int(3*(cols/4)):int(cols)]

	img10 = img[int(rows/3):int(2*(rows/3)),0:int((cols/4))]
	img11 = img[int(rows/3):int(2*(rows/3)),int((cols/4)):int(2*(cols/4))]
	img12 = img[int(rows/3):int(2*(rows/3)),int(2*(cols/4)):int(3*(cols/4))]
	img13 = img[int(rows/3):int(2*(rows/3)),int(3*(cols/4)):int(cols)]

	img20 = img[int(2*(rows/3)):int(rows),0:int((cols/4))]
	img21 = img[int(2*(rows/3)):int(rows),int((cols/4)):int(2*(cols/4))]
	img22 = img[int(2*(rows/3)):int(rows),int(2*(cols/4)):int(3*(cols/4))]
	img23 = img[int(2*(rows/3)):int(rows),int(3*(cols/4)):int(cols)]
	temp = np.array([[img00,img01,img02,img03],[img10,img11,img12,img13],[img20,img21,img22,img23]])

	return temp

def find_non_black(img):
	return np.where(np.logical_and(img[:,:,0] != 0.0,img[:,:,1] != 0.0,img[:,:,2] != 0.0))

def remove_black(img, index):
	channel0 = img[index[0],index[1],0]
	channel1 = img[index[0],index[1],1]
	channel2 = img[index[0],index[1],2]
	return channel0, channel1, channel2

def get_mean_grey(img):
	img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	return st.mean(img_grey.flatten().tolist())

def get_mean(channel0, channel1, channel2):
	if (len(channel0) == 0):
		return -1, -1, -1
	mean0 = st.mean(channel0.tolist())
	mean1 = st.mean(channel1.tolist())
	mean2 = st.mean(channel2.tolist())
	return mean0, mean1, mean2

def get_stdiv(channel0, channel1, channel2):
	if (len(channel0) == 0):
		return -1, -1, -1
	stdiv0 = np.sqrt(st.variance(channel0.tolist()))
	stdiv1 = np.sqrt(st.variance(channel1.tolist()))
	stdiv2 = np.sqrt(st.variance(channel2.tolist()))
	return stdiv0, stdiv1, stdiv2

def do_statistics(img, mask, basename):
	print("\n" + basename)
	# Histogram of origial image
	filename = basename + "_rgb_histogram.png"
	rgb_hist(img[:,:,2],img[:,:,1],img[:,:,0], filename)
	img_cielab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
	filename = basename + "_CIELab_histogram.png"
	cieLab_hist(img_cielab[:,:,0],img_cielab[:,:,1],img_cielab[:,:,2],filename)
	
	# Mean grey value
	print("  Mean grey value")
	print("  ", get_mean_grey(img), "\n")

	# Create masked image
	masked_img = cv2.bitwise_and(img,img, mask=mask)
	filename = "../output/ex1/" + basename + "_masked.png"
	cv2.imwrite(filename, masked_img)
	
	# Remove black
	non_black_mask = find_non_black(masked_img)
	bgr_b, bgr_g, bgr_r = remove_black(img, non_black_mask)
	cielab_L, cielab_a, cielab_b = remove_black(img_cielab, non_black_mask)

	# Histogram of masked image
	filename = basename + "_rgb_histogram_masked.png"
	rgb_hist(bgr_r, bgr_g, bgr_b, filename)
	filename = basename + "_CIELab_histogram_masked.png"
	cieLab_hist(cielab_L, cielab_a, cielab_b, filename)

	# Mean and standard diviation rgb
	print("  Mean and standard diviation bgr")
	print("  ", get_mean(bgr_b, bgr_g, bgr_r))
	print("  ", get_stdiv(bgr_b, bgr_g, bgr_r), "\n")
	
	# Mean and standard diviation cielab
	print("  Mean and standard diviation cielab")
	print("  ", get_mean(cielab_L, cielab_a, cielab_b))
	print("  ", get_stdiv(cielab_L, cielab_a, cielab_b))

if __name__ == "__main__":
	# Load image files
	filename = "../input/DJI_0240.JPG"
	mask_filename = "../input/DJI_0240_pumpkin_mask.png"
	img = cv2.imread(filename)
	mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
	do_statistics(img, mask, "DJI_0240_full")
	sub_img = sub_div_image(img)
	sub_mask = sub_div_image(mask)

	do_statistics(sub_img[0,0],sub_mask[0,0],"DJI_0240_sub00")
	do_statistics(sub_img[0,1],sub_mask[0,1],"DJI_0240_sub01")
	do_statistics(sub_img[0,2],sub_mask[0,2],"DJI_0240_sub02")
	do_statistics(sub_img[0,3],sub_mask[0,3],"DJI_0240_sub03")

	do_statistics(sub_img[1,0],sub_mask[1,0],"DJI_0240_sub10")
	do_statistics(sub_img[1,1],sub_mask[1,1],"DJI_0240_sub11")
	do_statistics(sub_img[1,2],sub_mask[1,2],"DJI_0240_sub12")
	do_statistics(sub_img[1,3],sub_mask[1,3],"DJI_0240_sub13")
	
	do_statistics(sub_img[2,0],sub_mask[2,0],"DJI_0240_sub20")
	do_statistics(sub_img[2,1],sub_mask[2,1],"DJI_0240_sub21")
	do_statistics(sub_img[2,2],sub_mask[2,2],"DJI_0240_sub22")
	do_statistics(sub_img[2,3],sub_mask[2,3],"DJI_0240_sub23")
