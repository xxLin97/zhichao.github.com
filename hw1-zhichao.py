#!/usr/bin/python3
import numpy as np
import cv2
from PIL import Image

def get_grayscale(img):
	b = img[:,:,0]
	g = img[:,:,1]
	r = img[:,:,2]
	gray = 0.114 * b + 0.587 * g + 0.299 * r
	gray = gray.astype(np.uint8)
	return gray


def process(img):
	img = get_grayscale(img)
	M = img.shape[0]
	N = img.shape[1]
	pixel_max_value = np.max(img)
	for row in range(0,M):
		for col in range(0,N):
			img[row,col] = pixel_max_value - img[row,col]
	arr = np.asarray(img)
	arr_flip = arr[::-1]
	return arr_flip

if __name__ == '__main__':
	image = cv2.imread("./face.jpeg")
	cv2.namedWindow('lzc', cv2.WINDOW_AUTOSIZE)
	cv2.imshow("original image",image)
	cv2.waitKey(2000)
	cv2.destroyAllWindows()
	cv2.imshow("grayscale",get_grayscale(image))
	cv2.waitKey(2000)
	cv2.destroyAllWindows()
	cv2.imshow("processed image",process(image))
	cv2.waitKey(2000)
	cv2.destroyAllWindows()