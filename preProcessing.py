# This module preprocesses an image to remove noise
# Just imporrt and use
import cv2
import numpy as np

div = 100 # Change if image shape is to large or small
kernel = np.ones((5, 5), np.uint8)

class PP():

	def __init__(self, img):
		self.img = img
		self.imgResized = img
		self.imgThresh = None

	def preprocess(self):
		# RP = self.img.shape[0] / div if self.img.shape[0] <= self.img.shape[1] else self.img.shape[1] / div  	# Setting resizing Parameter keeping aspect ratio constant
		# self.imgResized = cv2.resize (self.img, (self.img.shape[1] / RP, self.img.shape[0] / RP))  				# Resizing the image acc. to  the RP
		# imgGray = cv2.cvtColor(self.imgResized, cv2.COLOR_BGR2GRAY) 											# Grayscaling the image
		imgBlur = cv2.medianBlur(self.img, 5) 																	# Blurring to remove noise
		imgMorph = cv2.morphologyEx(imgBlur, cv2.MORPH_CLOSE, kernel)
		thresh, self.imgThresh = cv2.threshold(imgMorph, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)		# Threshing the image (OTSU)
		return self.imgThresh 